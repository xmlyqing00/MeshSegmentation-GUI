from mesh_data_structure.halfedge_mesh import HETriMesh
import numpy as np

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




