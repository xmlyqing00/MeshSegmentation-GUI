import igl
import numpy as np
import trimesh

def compute_boundary_length(v, bnd):
    bnd_length = 0
    for i in range(bnd):
        bnd_length += np.linalg.norm(v[bnd[i]] - v[bnd[i+1]])
    return bnd_length

def map_to_ngon(v, bnd, crn_ids, use_ratio=True):
    list_bnd = bnd.tolist()

    ## list of boundary vertices
    list_boundary = []
    list_boundary_length = []
    for i in range(len(crn_ids)):
        bid0 = list_bnd.index(crn_ids[i]) 
        bid1 = list_bnd.index(crn_ids[(i+1)%len(crn_ids)])
        if bid0 < bid1:
            list_boundary.append(list_bnd[bid0:bid1])
            list_boundary[-1].append(list_bnd[bid1])
            ##
            bnd_length = compute_boundary_length(v, list_boundary[-1])
            list_boundary_length.append(bnd_length)
        else:
            list_boundary.append(list_bnd[bid0:] + list_bnd[:bid1])
            list_boundary[-1].append(list_bnd[bid1])
            ##
            bnd_length = compute_boundary_length(v, list_boundary[-1])
            list_boundary_length.append(bnd_length)

    # ## compute the length of each boundary
    # list_boundary_length = []
    # for bnd in list_boundary:
    #     bnd_length = 0
    #     ## open curve
    #     for i in range(len(bnd)-1):
    #         bnd_length += np.linalg.norm(v[bnd[i]] - v[bnd[i+1]])
    #     list_boundary_length.append(bnd_length)
    
    ## compute the ratio of each boundary to the circumference of a unit circle
    boundary_length = np.array(list_boundary_length)
    total_length = np.sum(boundary_length)
    boundary_ratio = np.cumsum(boundary_length) / total_length
    # print(boundary_ratio)
    
    ## compute coordinate of each point
    if use_ratio:
        r = boundary_ratio
        radius = boundary_ratio * 2 * np.pi
    else:
        r = np.linspace(0, 1, len(list_boundary))
        radius = r[1:] * 2 * np.pi
    radius = np.concatenate((np.array([0]), radius))
    endpoints_uv = np.stack((np.cos(radius), np.sin(radius))).swapaxes(0,1)

    ## compute the uv coordinates of each boundary vertex (list_boundary) as linear combination of the coordinates of the end points
    all_boundary_uv = []
    for j, boundary in enumerate(list_boundary):
        t = np.linspace(0, 1, len(boundary))
        t = t.reshape((len(boundary), 1))
        boundary_uv = t * endpoints_uv[None,(j+1)%len(list_boundary)] + (1-t) * endpoints_uv[None,j]
        all_boundary_uv.append(boundary_uv[:-1])
    bnd_uv = np.concatenate(all_boundary_uv, axis=0)
    return bnd_uv, endpoints_uv

def igl_parameterization(v, f, crn_ids):
    bnd = igl.boundary_loop(f)
    ## Map to an N-gon
    bnd_uv, _ = map_to_ngon(v, bnd, crn_ids)
    ## Harmonic parametrization for the internal vertices
    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    uv = np.concatenate((uv, np.zeros((uv.shape[0], 1))), axis=1)

    return uv

def find_vertices_with_small_edges(mesh, crn_ids, threshold=0.1):
    v = mesh.vertices
    f = mesh.faces
    uv = igl_parameterization(v, f, crn_ids)

    edges = trimesh.geometry.faces_to_edges(mesh.faces)
    print(edges)
    edge_lengths = np.linalg.norm(uv[edges[:,0]] - uv[edges[:,1]], axis=1)
    small_edges = edges[edge_lengths < threshold]
    

if __name__ == "__main__":
    i = 0
    meshfile = f"data/autoseg181/submeshes/submesh_{i}.obj"
    v, f  = igl.read_triangle_mesh(meshfile)

    crnfile = f"data/autoseg181/submeshes/corners_{i}.txt"
    crns, crn_ids = loadtxt_crns(crnfile)