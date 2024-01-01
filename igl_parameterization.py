import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import igl
from meshplot import plot, subplot, interact

def loadtxt_crns(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        crns = []
        crn_ids = []
        for line in lines[1:]:
            splits = line.split(" ")
            crn_ids.append(int(splits[0]))
            crns.append([float(x) for x in splits[1:]])
        crns = np.array(crns)
        crn_ids = np.array(crn_ids, dtype=np.int32)
    return crns, crn_ids

def map_to_ngon(v, list_bnd, crn_ids):
    # print(v.shape)
    list_boundary = []
    for i in range(len(crn_ids)):
        # cid0 = crn_ids[i]
        # cid1 = crn_ids[(i+1)%len(crn_ids)]
        # print(cid0, cid1)
        bid0 = list_bnd.index(crn_ids[i]) 
        bid1 = list_bnd.index(crn_ids[(i+1)%len(crn_ids)])
        if bid0 < bid1:
            list_boundary.append(list_bnd[bid0:bid1])
            list_boundary[-1].append(list_bnd[bid1])
        else:
            list_boundary.append(list_bnd[bid0:] + list_bnd[:bid1])
            list_boundary[-1].append(list_bnd[bid1])
    # print(list_boundary)

    ## compute the length of each boundary
    list_boundary_length = []
    for bnd in list_boundary:
        bnd_length = 0
        ## open curve
        for i in range(len(bnd)-1):
            bnd_length += np.linalg.norm(v[bnd[i]] - v[bnd[i+1]])
        list_boundary_length.append(bnd_length)
    
    ## compute the ratio of each boundary to the circumference of a unit circle
    boundary_length = np.array(list_boundary_length)
    total_length = np.sum(boundary_length)
    boundary_ratio = np.cumsum(boundary_length) / total_length
    # print(boundary_ratio)
    
    ## compute coordinate of each point
    # r = np.linspace(0, 1, len(list_boundary))
    # radius = r[1:] * 2 * np.pi
    radius = boundary_ratio * 2 * np.pi
    radius = np.concatenate((np.array([0]), radius))
    endpoints_uv = np.stack((np.cos(radius), np.sin(radius))).swapaxes(0,1)
    # endpoints_uv = endpoints_uv[:-1]

    ## compute the uv coordinates of each boundary vertex (list_boundary) as linear combination of the coordinates of the end points
    all_boundary_uv = []
    # print(len(list_boundary))
    for j, boundary in enumerate(list_boundary):
        t = np.linspace(0, 1, len(boundary))
        t = t.reshape((len(boundary), 1))
        # print(j, endpoints_uv.shape)
        boundary_uv = t * endpoints_uv[None,(j+1)%len(list_boundary)] + (1-t) * endpoints_uv[None,j]
        all_boundary_uv.append(boundary_uv[:-1])
    # print(all_boundary_uv)
    bnd_uv = np.concatenate(all_boundary_uv, axis=0)
    return bnd_uv, endpoints_uv

def parameterize_mesh(v, f, crn_ids):
    # meshfile = f"data/autoseg181/submeshes/submesh_{i}.obj"
    # crnfile = f"data/autoseg181/submeshes/corners_{i}.txt"
    # v, f  = igl.read_triangle_mesh(meshfile)
    # crns, crn_ids = loadtxt_crns(crnfile)
    bnd = igl.boundary_loop(f)
    bnd_list = bnd.tolist()

    for cid in crn_ids:
        assert cid in bnd_list

    # ## Map the boundary to a circle, preserving edge proportions
    # bnd_uv = igl.map_vertices_to_circle(v, bnd)
    ## Map to an N-gon
    bnd_uv, _ = map_to_ngon(v, bnd_list, crn_ids)
    ## Harmonic parametrization for the internal vertices
    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    uv = np.concatenate((uv, np.zeros((uv.shape[0], 1))), axis=1)
    # print("done")
    return uv, bnd_uv
