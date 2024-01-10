from mesh_data_structure.halfedge_mesh import HETriMesh
import numpy as np
import trimesh
from potpourri3d import EdgeFlipGeodesicSolver
from tsp_solver.greedy import solve_tsp


## filters
def trace_boundary_edges(mesh: HETriMesh):
    """
    Trace boundary edges of a mesh
    """
    results = []

    ## find a first halfedge on the boundary
    visited = set()
    for hei, he in enumerate( mesh.halfedges ):
        if hei in visited:
            continue
        if -1 == he.face:
            start_he = he
            result = []
            ## trace
            while True:
                next_he_id = he.next_he
                next_he = mesh.halfedges[next_he_id]
                visited.add(next_he_id)
                result.append(mesh.he_index2directed_edge(next_he_id))
                he = next_he
                if he == start_he:
                    results.append(result)
                    # print("add new edge loop", len(results))
                    # print(result)
                    break
    return results

def close_holes(mesh: HETriMesh, boundaries: list):

    new_mesh = HETriMesh()
    vs = mesh.vs
    faces = mesh.faces


    ## compute boundary vertices average
    new_vertices = vs
    new_faces = faces
    centroid_ids = []
    for boundary_edges in boundaries:
        boundary_vertice_ids = [e[0] for e in boundary_edges]
        boundary_vertices = mesh.vs[boundary_vertice_ids]
        centroid = np.mean(boundary_vertices, axis=0)
        for e in boundary_edges:
            new_faces = np.vstack((new_faces, [e[0], e[1], len(new_vertices)]))
        new_vertices = np.vstack((new_vertices, centroid))
        centroid_ids.append(len(new_vertices)-1)
        
    new_mesh.init_mesh(new_vertices, new_faces)
    return new_mesh, centroid_ids



def convert_list_to_string(l:list):
    return ",".join([str(i) for i in l])


def compute_distance_matrix(mesh, b_close_holes=True):
    ## close the holes and compute the centroid of each hole
    if b_close_holes:
        mesh_holefilled, cids = mesh_close_holes(mesh)
        path_solver = GeoPathSolverWrapper(mesh_holefilled)
        paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)
    ## compute the shortest distance between each pair of holes
    else:
        path_solver = GeoPathSolverWrapper(mesh)
        boundary_loops = get_open_boundary(mesh)
        distance_matrix = np.zeros((len(boundary_loops), len(boundary_loops)))
        paths = {}
        for i in range(len(boundary_loops)-1):
            for j in range(i+1, len(boundary_loops)):
                b0 = np.array(boundary_loops[i], dtype=np.int32)[:,0]
                b0 = np.hstack([b0, boundary_loops[i][-1][1]])
                b1 = np.array(boundary_loops[j], dtype=np.int32)[:,0]
                b1 = np.hstack([b1, boundary_loops[j][-1][1]])
                path_pts, distance = path_solver.solve_path2path(b0, b1)
                paths[f"{i},{j}"] = path_pts
                ## 
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    return paths, distance_matrix 


class GeoPathSolverWrapper():
    def __init__(self, mesh : trimesh.Trimesh) -> None:
        v = mesh.vertices
        f = np.array(mesh.faces, dtype=np.int32)
        self.solver = EdgeFlipGeodesicSolver(v, f,) # shares precomputation for repeated solves

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
    
    def solve_vlist(self, vlist):
        ## cannot use unique, because the order is important. Please return idx
        # vlist_unique = np.unique(vlist)  
        return self.solver.find_geodesic_path_poly(vlist)
    
    def solve_path2path(self, vlist0, vlist1):
        min_distance = 1e10
        min_pair = None
        for i in range(len(vlist0)):
            for j in range(len(vlist1)):
                path_pts = self.solve(vlist0[i], vlist1[j])                
                distance = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
                distance = np.array(distance).sum()
                if distance < min_distance:
                    min_distance = distance
                    min_pair = [vlist0[i], vlist1[j]]
        path_pts = self.solve(min_pair[0], min_pair[1])
        return path_pts, min_distance


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

