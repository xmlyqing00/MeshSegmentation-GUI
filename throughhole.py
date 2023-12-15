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

def mesh_close_holes(patch_mesh : trimesh.Trimesh):
    he_mesh = HETriMesh()
    he_mesh.init_mesh(patch_mesh.vertices, patch_mesh.faces)
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
    non_disk_mask_id = []
    non_disk_mask = []
    list_boundaries = []
    for i in range(len(mask)):
        m = np.array(mask[i])
        # patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[m,:])
        he_mesh = HETriMesh()
        he_mesh.init_mesh(mesh.vertices, mesh.faces[m,:])
        tmp_boundaries = trace_boundary_edges(he_mesh)
        list_boundaries.append(tmp_boundaries)
        if len(tmp_boundaries)  == 1:
            print("disk topology ---", i)
        else:
            non_disk_mask_id.append(i)
            non_disk_mask.extend(m.tolist())
            print("not disk topology ---", i)

    non_disk_mask = simple_floodfill_label_mesh(mesh, non_disk_mask)
    print("non_disk_mask", len(non_disk_mask))

    throughhole_path = []
    for i in range(len(non_disk_mask)):
        m = np.array(non_disk_mask[i], dtype=np.int32)
        patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[m,:])
        patch_mesh.export(os.path.join(save_dir,f"non_disk_mask_{i}.obj"))
    
        patch_mesh_holefilled, cids = mesh_close_holes(patch_mesh)
        patch_mesh_holefilled.export(os.path.join(save_dir,"patch_mesh_holefilled.obj"))

        # ## GeoPathSolverWrapper
        path_solver = GeoPathSolverWrapper(patch_mesh_holefilled)
        paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)    
        print(distance_matrix)

        ## solve TSP
        tsp_path = solve_tsp(distance_matrix)
        print(tsp_path)

        for k in range(len(tsp_path)):
            i = tsp_path[k]
            j = tsp_path[(k+1)%len(tsp_path)]
            keys = [f"{i},{j}", f"{j},{i}"]
            for key in keys:
                if key in paths:
                    throughhole_path.append(paths[key][1:-1])
                    break
        
    cnt = 0
    for i, boundaries in enumerate(list_boundaries):
        print(i, len(boundaries))
        for boundary in boundaries:
            boundary = np.vstack(boundary)
            print("boundary", boundary[:,0])
            bverts = mesh.vertices[boundary[:,0]]
            trimesh.PointCloud(bverts).export(os.path.join(save_dir,f"list_boundaries_{cnt}.obj"))
            cnt += 1
    
    for i in range(len(throughhole_path)):
        path = np.vstack(throughhole_path[i])
        trimesh.PointCloud(path).export(os.path.join(save_dir,f"throughhole_path_{i}.obj"))