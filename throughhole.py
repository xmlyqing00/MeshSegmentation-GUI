import trimesh
import numpy as np
from potpourri3d import EdgeFlipGeodesicSolver
from tsp_solver.greedy import solve_tsp
from view_psd_data import *

from mesh_data_structure.halfedge_mesh import HETriMesh
from mesh_data_structure.utils import trace_boundary_edges, close_holes

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

if __name__ == "__main__":
    
    save_dir = "output_throughhole"
    os.makedirs(save_dir, exist_ok=True)

    ## load mesh
    shape_id = "167"
    path = f"/home/lyang/yl_code/dataset/labeledDb/LabeledDB_new/*/{shape_id}.off"
    mesh, mask = visualize_psd_shape(path, path.replace(".off", "_labels.txt"))

    """
    new idea:
    1 get the mask of the mesh
    2 get the boundary of each mask. These are the existing cuts
    3 For each mask that has a disk topology, compute its centroid
    4 Compute a TSP path that goes through all centroids
    5 Cut the mesh along the TSP path and the existing cuts
    """


    ## get masked mesh
    new_mask = mask[0]
    new_mask.extend(mask[1])
    mask = np.array(new_mask)
    patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[mask,:])
    patch_mesh.export(os.path.join(save_dir,"patch_mesh.obj"))

    patch_mesh_holefilled, cids = mesh_close_holes(patch_mesh)
    patch_mesh_holefilled.export(os.path.join(save_dir,"patch_mesh_holefilled.obj"))
    # boundary_vertices = patch_mesh_holefilled.vertices[cids]
    # trimesh.PointCloud(boundary_vertices).export(f'boundary_vertices.ply')


    # ## GeoPathSolverWrapper
    path_solver = GeoPathSolverWrapper(patch_mesh_holefilled)
    paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)    
    print(distance_matrix)

    ## solve TSP
    tsp_path = solve_tsp(distance_matrix)
    print(tsp_path)

    throughhole_path = []
    for k in range(len(tsp_path)):
        i = tsp_path[k]
        j = tsp_path[(k+1)%len(tsp_path)]
        keys = [f"{i},{j}", f"{j},{i}"]
        for key in keys:
            if key in paths:
                throughhole_path.extend(paths[key])
                break
        trimesh.PointCloud(throughhole_path).export(os.path.join(save_dir,'throughhole_path.ply'))