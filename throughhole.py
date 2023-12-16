import trimesh
import numpy as np
from potpourri3d import EdgeFlipGeodesicSolver
from tsp_solver.greedy import solve_tsp
from view_psd_data import *
import shutil
import argparse

from mesh_data_structure.halfedge_mesh import HETriMesh
from mesh_data_structure.utils import trace_boundary_edges, close_holes
import networkx as nx

class GeoPathSolverWrapper():
    def __init__(self, mesh : trimesh.Trimesh) -> None:
        v = mesh.vertices
        f = np.array(mesh.faces, dtype=np.int32)
        self.solver = EdgeFlipGeodesicSolver(v, f) # shares precomputation for repeated solves

    def solve_all2all(self, vids, return_distance_matrix=False):
        paths = {}
        distance_matrix = np.zeros((len(vids), len(vids)))
        for i in range(len(vids)):
            for j in range(i+1, len(vids)):
                v_start = vids[i]
                v_end = vids[j]
                path_pts = self.solve(v_start, v_end)
                paths[f"{i},{j}"] = path_pts
                if return_distance_matrix:
                    distance = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
                    distance = np.array(distance).sum()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        if return_distance_matrix:
            return paths, distance_matrix
        else:
            return paths

    def solve(self, v_start, v_end):
        path_pts = self.solver.find_geodesic_path(v_start, v_end)
        return path_pts

def get_open_boundary(mesh : trimesh.Trimesh):
    he_mesh = HETriMesh()
    he_mesh.init_mesh(mesh.vertices, mesh.faces)
    boundary_loops = trace_boundary_edges(he_mesh)
    return boundary_loops

def mesh_close_holes(mesh : trimesh.Trimesh):
    he_mesh = HETriMesh()
    he_mesh.init_mesh(mesh.vertices, mesh.faces)
    boundary_loops = trace_boundary_edges(he_mesh)
    ## close the hole
    he_mesh, centroid_ids = close_holes(he_mesh, boundary_loops)
    mesh = trimesh.Trimesh(he_mesh.vs, he_mesh.faces, process=False, maintain_order=True)
    return mesh, centroid_ids



def tsp_segment(mesh : trimesh.Trimesh):
    
    patch_mesh_holefilled, centroid_ids = mesh_close_holes(mesh)

    ## GeoPathSolverWrapper
    path_solver = GeoPathSolverWrapper(patch_mesh_holefilled)
    paths, distance_matrix = path_solver.solve_all2all(centroid_ids, return_distance_matrix=True)    
    ## solve TSP to find the order of throughhole
    tsp_path = solve_tsp(distance_matrix)
    ## find throughhole path
    throughhole_path = []
    for k in range(len(tsp_path)):
        i = tsp_path[k]
        j = tsp_path[(k+1)%len(tsp_path)]
        keys = [f"{i},{j}", f"{j},{i}"]
        for key in keys:
            if key in paths:
                throughhole_path.append(paths[key][1:-1])
                break
    return throughhole_path


def simple_floodfill_label_mesh(
    mesh: trimesh.Trimesh, 
    mask: list
):  
    mask = np.array(mask)
    mask_connected = []
    patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[mask,:], maintain_order=True, process=False)
    out = trimesh.graph.connected_component_labels(patch_mesh.face_adjacency)
    for i in range(out.max()+1):
        mask_connected.append((mask[out==i]).tolist())
    return mask_connected


def process_data(mesh, mask, merge_annulus=True):

    print(mesh.vertices.shape, mesh.faces.shape, len(mask))

    ## floodfild the mask
    mask_connected = []
    for i in range(len(mask)):
        m = np.array(mask[i])
        mask_connected.extend(simple_floodfill_label_mesh(mesh, m))
    mask = mask_connected

    """
    new idea:
    1 get the mask of the mesh
    2 get the boundary of each mask. These are the existing cuts
    3 For each mask that has a disk topology, compute its centroid
    4 Compute a TSP path that goes through all centroids
    5 Cut the mesh along the TSP path and the existing cuts
    """

    annulus_mask = []
    non_disk_mask = []
    list_boundaries = []
    for i in range(len(mask)):
        m = np.array(mask[i])
        he_mesh = HETriMesh()
        he_mesh.init_mesh(mesh.vertices, mesh.faces[m,:])
        tmp_boundaries = trace_boundary_edges(he_mesh)
        list_boundaries.append(tmp_boundaries)
        if len(tmp_boundaries)  == 1:
            print("disk topology ---", i)
        elif len(tmp_boundaries) > 1:
            if len(tmp_boundaries) == 2:
                print("not disk/annulus topology ---", i)
                if merge_annulus:
                    non_disk_mask.extend(m.tolist())
                else:
                    annulus_mask.extend(m.tolist())
            else:
                print("not disk topology ---", i)
                non_disk_mask.extend(m.tolist())
        else:
            print("Error: no boundary found")
            

    if len(non_disk_mask) > 0:
        non_disk_mask = simple_floodfill_label_mesh(mesh, non_disk_mask)

    if merge_annulus:
        assert len(annulus_mask) == 0
        for i in range(len(non_disk_mask)):
            m = np.array(non_disk_mask[i])
            he_mesh = HETriMesh()
            he_mesh.init_mesh(mesh.vertices, mesh.faces[m,:])
            tmp_boundaries = trace_boundary_edges(he_mesh)
            if len(tmp_boundaries) == 2:
                annulus_mask.append(m.tolist())
    else:
        if len(annulus_mask) > 0:
            annulus_mask = simple_floodfill_label_mesh(mesh, annulus_mask)
    
    print("annulus_mask", len(annulus_mask))
    print("non_disk_mask", len(non_disk_mask))
    return list_boundaries, non_disk_mask, annulus_mask


def cut_annulus(mesh, annulus_mask):
    annulus_cut = []
    for i in range(len(annulus_mask)):
        m = np.array(annulus_mask[i])
        patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[m,:])
        boundary_loops = get_open_boundary(patch_mesh)
        ## find two points on one hole, and compute the shortest path to the centroid of the other hole
        path_solver = GeoPathSolverWrapper(patch_mesh)

        b0 = []
        for b in boundary_loops[0]:
            b0.append(b[0])
        b0 = np.array(b0)
        b1 = []
        for b in boundary_loops[1]:
            b1.append(b[0])
        b1 = np.array(b1)
        p00 = b0[0]
        p01 = b0[len(b0)//2]
        p10 = b1[0]
        p11 = b1[len(b1)//2]
        cids=[p00, p01, p10, p11]
        paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)
        # for i in range(3):
        #     for j in range(i+1, 4):
        #         print(i, j, distance_matrix[i,j])
        # print(distance_matrix)
        # for k, path in paths.items():
        #     trimesh.PointCloud(path).export(os.path.join(save_dir,f"candidate_cuts_{k}.obj"))

        ## solve TSP
        ignored_links= [[0, 1], [2, 3]]
        tsp_path = solve_tsp(distance_matrix)
        # print(tsp_path)
        for k in range(len(tsp_path)):
            i = tsp_path[k]
            j = tsp_path[(k+1)%len(tsp_path)]
            keys = [f"{i},{j}", f"{j},{i}"]
            for key in keys:
                if key in paths and [i,j] not in ignored_links and [j,i] not in ignored_links:
                    # print("key", key)
                    annulus_cut.append(paths[key])
                    break
    return annulus_cut


def cut_through_holes(mesh, non_disk_mask):

    throughhole_path = []
    for i in range(len(non_disk_mask)):
        m = np.array(non_disk_mask[i], dtype=np.int32)
        patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[m,:])
        patch_mesh_holefilled, cids = mesh_close_holes(patch_mesh)

        # ## GeoPathSolverWrapper
        path_solver = GeoPathSolverWrapper(patch_mesh_holefilled)
        paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)    
        # print(distance_matrix)

        ## solve TSP
        tsp_path = solve_tsp(distance_matrix)
        # print(tsp_path)

        for k in range(len(tsp_path)):
            i = tsp_path[k]
            j = tsp_path[(k+1)%len(tsp_path)]
            keys = [f"{i},{j}", f"{j},{i}"]
            for key in keys:
                if key in paths:
                    throughhole_path.append(paths[key][1:-1])
                    break
    return throughhole_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='167', help='Input mesh path.')
    args = parser.parse_args()

    save_dir = "output_throughhole"
    if os.path.exists(save_dir):
        print("remove", save_dir)
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    ## load mesh
    shape_id = args.input
    fpath = f"./data/segmentation_data/*/{shape_id}.off"
    mesh, mask = visualize_psd_shape(fpath, fpath.replace(".off", "_labels.txt"))
    mesh.export(os.path.join(save_dir,"mesh.obj"))

    list_boundaries, non_disk_mask, annulus_mask = process_data(
        mesh, mask, merge_annulus=False)
    annulus_cuts = cut_annulus(mesh, annulus_mask)
    throughhole_path = cut_through_holes(mesh, non_disk_mask)

    cnt = 0
    for i, boundaries in enumerate(list_boundaries):
        # print(i, len(boundaries))
        for boundary in boundaries:
            boundary = np.vstack(boundary)
            # print("boundary", boundary[:,0])
            bverts = mesh.vertices[boundary[:,0]]
            trimesh.PointCloud(bverts).export(os.path.join(save_dir,f"list_boundaries_{cnt}.obj"))
            cnt += 1
    
    for i in range(len(throughhole_path)):
        path = np.vstack(throughhole_path[i])
        trimesh.PointCloud(path).export(os.path.join(save_dir,f"throughhole_path_{i}.obj"))
    
    for i in range(len(annulus_cuts)):
        path = np.vstack(annulus_cuts[i])
        trimesh.PointCloud(path).export(os.path.join(save_dir,f"annulus_cuts_{i}.obj"))