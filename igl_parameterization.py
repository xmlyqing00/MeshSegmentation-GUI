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
    list_boundary = []
    new_list_bnd = []
    for i in range(len(crn_ids)):
        bid0 = list_bnd.index(crn_ids[i])
        bid1 = list_bnd.index(crn_ids[(i+1)%len(crn_ids)])
        # bid1 = list_bnd.index(crn_ids[(i+1)%len(crn_ids)])+1
        if bid0 < bid1:
            list_boundary.append(list_bnd[bid0:bid1+1])
            new_list_bnd += list_bnd[bid0:bid1]
        else:
            list_boundary.append(list_bnd[bid0:] + list_bnd[:bid1+1])
            new_list_bnd += list_bnd[bid0:] + list_bnd[:bid1]

    ## compute the length of each boundary
    list_boundary_length = []
    for bnd in list_boundary:
        bnd_length = 0
        ## open curve
        for i in range(len(bnd)-1):
            bnd_length += np.linalg.norm(v[bnd[i]] - v[bnd[i+1]])
        list_boundary_length.append(bnd_length)
    
    ## compute the ratio of each boundary to the circumference of a unit circle
    list_boundary_length.insert(0, 0) ## add the cyclic start
    boundary_length = np.array(list_boundary_length)
    total_length = np.sum(boundary_length)
    boundary_length_ratio = boundary_length / total_length
    print("boundary ratio", boundary_length_ratio)
    boundary_cumsum_ratio = np.cumsum(boundary_length_ratio)
    # boundary_cumsum_ratio = np.cumsum(boundary_length) / total_length
    print("boundary_cumsum ratio", boundary_cumsum_ratio)
    
    ## compute coordinate of each point
    radius = boundary_cumsum_ratio * 2 * np.pi
    print("radius", radius)
    endpoints_uv = np.stack((np.cos(radius), np.sin(radius))).swapaxes(0,1)
    print("endpoints_uv", endpoints_uv)

    ## compute the uv coordinates of each boundary vertex (list_boundary) as linear combination of the coordinates of the end points
    all_boundary_uv = []
    print(len(list_boundary))
    for j, boundary in enumerate(list_boundary):

        ## TODO: replace the following equidistant sampling with arc-length sampling
        bnd_vertices = v[boundary]
        arc_length = np.linalg.norm(bnd_vertices[1:] - bnd_vertices[:-1], axis=1)
        arc_length = np.concatenate(([0], arc_length))
        cum_arc_length = np.cumsum(arc_length)
        cum_arc_length_ratio = cum_arc_length / cum_arc_length[-1]
        # print(len(boundary), cum_arc_length_ratio.shape)
        # print(cum_arc_length_ratio)
        # input()
        # t = np.linspace(0, 1, len(boundary))
        # t = t.reshape((len(boundary), 1))
        # print(j, endpoints_uv.shape)
        t = cum_arc_length_ratio.reshape((len(boundary), 1))
        boundary_uv = t * endpoints_uv[None,j+1] + (1-t) * endpoints_uv[None,j]
        boundary_uv = t * endpoints_uv[None,j+1] + (1-t) * endpoints_uv[None,j]
        # all_boundary_uv.append(boundary_uv)
        all_boundary_uv.append(boundary_uv[:-1])
    
    # input()
    bnd_uv = np.concatenate(all_boundary_uv, axis=0)
    return bnd_uv, endpoints_uv, boundary_length_ratio, new_list_bnd

def parameterize_mesh(v, f, crn_ids):
    bnd = igl.boundary_loop(f)
    print(bnd.shape)
    bnd_list = bnd.tolist()

    for cid in crn_ids:
        assert cid in bnd_list

    # ## Map the boundary to a circle, preserving edge proportions
    # bnd_uv = igl.map_vertices_to_circle(v, bnd)
    ## Map to an N-gon
    bnd_uv, endpoints, list_boundary_length, bnd_list = map_to_ngon(v, bnd_list, crn_ids)
    bnd = np.array(bnd_list, dtype=np.int64).reshape(-1,1)
    ## Harmonic parametrization for the internal vertices
    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    return uv, bnd_uv, endpoints, list_boundary_length, bnd_list
