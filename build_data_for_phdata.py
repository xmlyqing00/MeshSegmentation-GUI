import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import igl
import trimesh
from src.io_tools import find_nan_np, read_json, write_obj_file, write_json
from vedo import Plotter, Mesh, Line, Points, write, utils, show

from scipy.spatial import KDTree
from loguru import logger
import shutil
import argparse
import matplotlib
from matplotlib import cm
import networkx as nx
from PIL import Image

def sort_indices(pre_sort_indices, list_indices):
    """
    Sort the pre_sort_indices according to the list_indices
    """
    pos_list = []
    for idx in pre_sort_indices:
        pos = list_indices.index(idx)
        pos_list.append(pos)
    sorted_indices = [x for _, x in sorted(zip(pos_list, pre_sort_indices))]



def map_to_ngon(v, list_bnd, crn_ids, boundary_len_input = None):
        
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

    if boundary_len_input is None:
        boundary_length = np.array(list_boundary_length)
    else:
        boundary_length = np.array(boundary_len_input)


    total_length = np.sum(boundary_length)
    boundary_length_ratio = boundary_length / total_length
    boundary_cumsum_ratio = np.cumsum(boundary_length_ratio)
    
    ## compute coordinate of each point
    radius = boundary_cumsum_ratio * 2 * np.pi
    endpoints_uv = np.stack((np.cos(radius), np.sin(radius))).swapaxes(0,1)
    # print("endpoints_uv", endpoints_uv)

    ## compute the uv coordinates of each boundary vertex (list_boundary) as linear combination of the coordinates of the end points
    all_boundary_uv = []
    for j, boundary in enumerate(list_boundary):
        bnd_vertices = v[boundary]
        arc_length = np.linalg.norm(bnd_vertices[1:] - bnd_vertices[:-1], axis=1)
        arc_length = np.concatenate(([0], arc_length))
        cum_arc_length = np.cumsum(arc_length)
        cum_arc_length_ratio = cum_arc_length / cum_arc_length[-1]
        t = cum_arc_length_ratio.reshape((len(boundary), 1))
        boundary_uv = t * endpoints_uv[None,j+1] + (1-t) * endpoints_uv[None,j]
        boundary_uv = t * endpoints_uv[None,j+1] + (1-t) * endpoints_uv[None,j]
        # all_boundary_uv.append(boundary_uv)
        all_boundary_uv.append(boundary_uv[:-1])
    
    bnd_uv = np.concatenate(all_boundary_uv, axis=0)
    return bnd_uv, endpoints_uv, boundary_length_ratio, new_list_bnd


def parameterize_mesh(v, f, crn_ids):

    bnd = igl.boundary_loop(f)
    bnd_list = bnd.tolist()

    ## Map the boundary to a circle, preserving edge proportions
    # bnd_uv = igl.map_vertices_to_circle(v, bnd)
    out = map_to_ngon(v, bnd_list, crn_ids)
    # bnd_uv, endpoints_uv, boundary_length_ratio, new_bnd = out
    bnd_uv = out[0]
    new_bnd = out[3]
    new_bnd = np.array(new_bnd, dtype=int)

    uv = igl.harmonic(v, f, new_bnd, bnd_uv, 1)
    # try:
    #     find_nan_np(uv, "uv")
    # except AssertionError:
    #     np.savetxt("uv.txt", uv)
    #     AssertionError
    # print("crns", uv[crn_ids])
    

    return uv, bnd_uv


class PHComplex():
    def __init__(self, base_mesh, mask, threshold=0.00001, debug=False) -> None:

        self.threshold = threshold ## for connecting patches

        self.base_mesh = base_mesh
        self.mask = mask
        self.num_patches = len(mask)

        if debug:
            self.debug = True
            self.debug_dir = "debug"
            if os.path.exists(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(self.debug_dir)
        else:
            self.debug = False
            

        vedo_mesh = utils.trimesh2vedo(base_mesh)

        ## 1) get patches
        self.patches = []

        ## import color map
        cmap = matplotlib.cm.get_cmap('tab20', self.num_patches)
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        for i in range(self.num_patches):
            print(f"patch {i} {len(self.mask[i])}")
            patch = self.base_mesh.submesh([self.mask[i]], append=True)
            self.patches.append(patch)
            vedo_mesh.cellcolors[self.mask[i]] = (np.array(mapper.to_rgba(i%20))*255)[None,:]

        # vedo_mesh.cellcolors = vedo_mesh.cellcolors[:,:3]
        print(vedo_mesh.cellcolors)
        
        if self.debug:
            vedo_mesh.write(f"{self.debug_dir}/patches.ply")


    ## 2) make a scaffold mesh from the patches
    def init_scaffold_vertices(self):

        """
        HOW:
        2.1) Get the boundary of the patches
        2.2) Patch connectivity
        2.3) Find the nearest boundary points from other patches' boundary
        2.4) Identify corner points of the scaffold
        """

        ## 2.1) Get the boundary of the patches
        patch_boundaries = []
        for patch in self.patches:
            patch_bnd_idx = igl.boundary_loop(patch.faces)
            tree = KDTree(patch.vertices[patch_bnd_idx])
            patch_bnd = {
                "bnd_indices": patch_bnd_idx,
                "mesh": patch,
                "tree": tree,
            }
            patch_boundaries.append(patch_bnd)
        logger.info("patch boundaries are found")

        ## 2.2) Patch connectivity
        PatchAdj = np.ones((self.num_patches, self.num_patches))*100
        for i, patch_bnd in enumerate(patch_boundaries):
            for j, patch_bnd2 in enumerate(patch_boundaries):
                if i == j:
                    PatchAdj[i, j] = 0.0
                    continue
                dist, idx = patch_bnd2["tree"].query(patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"]])
                PatchAdj[i, j] = np.min(dist)
        np.savetxt("PatchAdj.txt", PatchAdj, fmt="%0.4f")
        logger.info("patch connectivity is found")
        
        ## 2.3) Find the nearest boundary points from the other connected patches' boundary
        for i, patch_bnd in enumerate(patch_boundaries):
            connected_patch_ids = np.where(PatchAdj[i] < self.threshold)[0]
            connected_patch_ids = connected_patch_ids[connected_patch_ids != i]

            v2p_connectivity = np.zeros((patch_bnd["bnd_indices"].shape[0], len(connected_patch_ids)))
            
            for j, pid in enumerate(connected_patch_ids):
                other = patch_boundaries[pid]
                nearest_distances, nearest_indices = other["tree"].query(patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"]])
                # nearest_distances, nearest_indices = patch_bnd["tree"].query(other["mesh"].vertices[other["bnd_indices"]])
                mask = np.where(nearest_distances < self.threshold)[0]
                v2p_connectivity[mask, j] = 1

            ## make it a closed boundary
            v2p_sum = np.sum(np.concatenate([v2p_connectivity, v2p_connectivity[0][None,:]], axis=0), axis=1)

            ## find boundary points
            bmask = np.where(v2p_sum == 0)[0]
            bmask = np.unique(np.concatenate([bmask, bmask-1, bmask+1]))
            bmask = np.clip(bmask, 0, len(v2p_sum)-1)
            v2p_sum[bmask] += 1

            ## corners as the points with connectivity changes
            ## differential v2p_sum
            diff_v2p_sum = np.diff(v2p_sum) 
            corners = np.where(diff_v2p_sum < 0)[0]
            patch_bnd["corners"] = corners

        ## 2.4) Identify corner points of the scaffold
        for i, patch_bnd in enumerate(patch_boundaries):
            corners = patch_bnd["corners"]
            true_corners_mask = np.zeros(len(corners), dtype=bool)
            connected_patch_ids = np.where(PatchAdj[i] < self.threshold)[0]
            connected_patch_ids = connected_patch_ids[connected_patch_ids != i]
            for j, pid in enumerate(connected_patch_ids):
                ## compute distance between corners
                other = patch_boundaries[pid]
                other_corners = other["corners"]
                dist = np.linalg.norm(patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"][corners][:, None]] - other["mesh"].vertices[other["bnd_indices"][other_corners]], axis=2)
                min_dist = np.min(dist, axis=1)
                true_corners_mask = np.logical_or(true_corners_mask, min_dist < self.threshold)
            true_corners = corners[true_corners_mask]
            patch_bnd["corners"] = true_corners
            true_corner_vertices = patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"][true_corners]]
            if self.debug:
                write_obj_file(f"{self.debug_dir}/patch_{i}_{len(true_corner_vertices)}.obj", patch_bnd["mesh"].vertices, patch_bnd["mesh"].faces)
                write_obj_file(f"{self.debug_dir}/true_corner_{i}_{len(true_corner_vertices)}.obj", true_corner_vertices)




def read_curvenet_file(filename, data_type, start_idx=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    
    print(lines)
    ## drop last four lines
    lines = lines[:-4]

    vertices = []
    max_length_edge = 0

    if data_type == 2:
        edge_collection = []
        edges = []
        for l in lines:
            ## vertices
            if len(l) > 0 and l[0] == 'v':
                vertices.append([float(l[1]), float(l[2]), float(l[3])])
            ## faces
            elif len(l) > 0 and l[0] == 'l':
                if len(l) - 1 > max_length_edge:
                    max_length_edge = len(l)-1
                edge_pts = [int(id)-start_idx for id in l[1:]]
                edges.append(edge_pts)
            else:
                if len(edges) > 0:
                    edge_collection.append(edges)
                    edges = []
        vertices = np.array(vertices)
    else:
        edge_collection = []
        for l in lines:
            ## vertices
            if len(l) > 0 and l[0] == 'v':
                vertices.append([float(l[1]), float(l[2]), float(l[3])])
            ## faces
            elif len(l) > 0 and l[0] == 'l':
                if len(l) - 1 > max_length_edge:
                    max_length_edge = len(l)-1
                edge_pts = [int(id)-start_idx for id in l[1:]]
                edge_collection.append(edge_pts)
        vertices = np.array(vertices)
    return vertices, edge_collection


model_dict = {
    "boat": 2,
    "espresso": 2,
    "loftbug": 1,
    "Roadster": 1, 
    "sga_torso": 1, 
    "toothpaste": 1, 
    "gamepad": 1, 
}


def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument('--curvenet', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--model_name', type=str, required=True)


    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    data_type = model_dict[args.model_name]
    curvenet_file = f"data/phdata/{args.model_name}/curvenet.obj"
    model_file = f"data/phdata/{args.model_name}/result.obj"
    texture_img = Image.open(f'./assets/uv_color.png')

    debug_dir = "phdata_output"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    if os.path.exists(f"{debug_dir}/{args.model_name}"):
        shutil.rmtree(f"{debug_dir}/{args.model_name}")
    os.makedirs(f"{debug_dir}/{args.model_name}")
    os.makedirs(f"{debug_dir}/{args.model_name}/flat_parameterization")
    os.makedirs(f"{debug_dir}/{args.model_name}/parameterization")
    os.makedirs(f"{debug_dir}/{args.model_name}/data/single")


    vertices, edge_collection = read_curvenet_file(curvenet_file, data_type, start_idx=1)
    corners = set()
    for i, edges in enumerate(edge_collection):
        edge_verts = []
        if data_type == 2:
            for e in edges:
                edge_verts.append(vertices[e[0]])
            edge_verts.append(vertices[edges[-1][1]])
            edge_verts = np.array(edge_verts)

            corners.add(edges[0][0])
            corners.add(edges[-1][-1])
        else:
            for e in edges:
                edge_verts.append(vertices[e])
            edge_verts = np.array(edge_verts)
        
            corners.add(edges[0])
            corners.add(edges[-1])
        # write_obj_file(f"{debug_dir}/edge_verts_{i}.obj", edge_verts)

    ## filter same vertices
    corner_verts = []
    for c in corners:
        v = vertices[c]
        if len(corner_verts) == 0:
            corner_verts.append(v)
        else:
            dist = np.linalg.norm(corner_verts - v[None,:], axis=1)
            if np.min(dist) > 0.000001:
                corner_verts.append(vertices[c])
    corner_verts = np.array(corner_verts)

    scene = trimesh.load(model_file, process=False, maintain_order=True)

    ## outputs
    """
    outputs of the code:
    data - single - mesh.obj
    data - cell_arc_lengths.json
    data - mask.json
    data - topology_graph.json

    data - flat_parameterization - flat_i.obj
    data - parameterization - mesh_uv_i.obj
    """

    ## data    
    output_data = {
        "cell_arc_lengths": "cell_arc_lengths.json",
        "mask": "mask.json",
        "topology_graph": "topology_graph.json",
        "mesh": "mesh.obj",
        "flat_parameterization": "flat",
        "parameterization": "mesh_uv",
    }
    topology_graph = {
        "node_ids": [],
        "cells": []
    }
    cell_arc_lengths = []
    face_mask = []
    patches = []
    patches_uv = []


    assert isinstance(scene, trimesh.Scene)
    ## colormap
    cmap = matplotlib.colormaps['tab20']
    norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    cnt = 0

    for k, g in scene.geometry.items():
        ## check if g is connected
        adjacency = trimesh.graph.face_adjacency(g.faces, g)
        ## save each component as a patch
        graph_connected = nx.Graph() 
        graph_connected.add_edges_from(g.face_adjacency) 
        groups = nx.connected_components(graph_connected)

        for c in groups:
            face_mask.append(list(c))
            patch = g.submesh([list(c)], append=True)
            patches.append(patch.copy())

            boundary = igl.boundary_loop(patch.faces)
            boundary_verts = patch.vertices[boundary]
            ## distance between boundary vertices and corners
            dist = np.linalg.norm(boundary_verts[:, None] - corner_verts[None], axis=2)
            min_dist = np.min(dist, axis=1)
            min_dist_idx = np.argmin(dist, axis=1)
            ## find corners correspioing to the boundary
            threshold = 0.000001
            boundary_corner_idx = np.where(min_dist < threshold)[0]
            
            if len(boundary_corner_idx) < 3:
                reverse_dist = np.linalg.norm(boundary_verts[:, None] - corner_verts[None], axis=2)
                reverse_min_dist = np.min(reverse_dist, axis=0)
                reverse_min_dist_idx = np.argmin(reverse_dist, axis=0)
                ## sort the reverse_min_dist_idx according to reverse_min_dist
                sorted = np.argsort(reverse_min_dist)
                boundary_corner_idx = reverse_min_dist_idx[sorted[:3]] ## get the first three
                boundary_corner_idx = np.sort(boundary_corner_idx)

                write_obj_file(f"{debug_dir}/{args.model_name}/data/patch_{cnt}_boundary.obj", boundary_verts)
                write_obj_file(f"{debug_dir}/{args.model_name}/data/patch_{cnt}_corner.obj", corner_verts)
                write_obj_file(f"{debug_dir}/{args.model_name}/data/patch_{cnt}_boundary_corner.obj", boundary_verts[boundary_corner_idx])
            


            corner_idx = min_dist_idx[boundary_corner_idx]
            topology_graph["cells"].append(corner_idx.tolist())

            # print("len of boundary vertices", len(boundary_verts))
            # print("boundary_corner_idx", boundary_corner_idx)
            # print("min_dist_idx", min_dist_idx)
            # print("min_dist", min_dist)
            # print(boundary)
            # print("corner_idx", corner_idx)
            # input()

            ## compute arc length
            boundary_verts = np.concatenate([boundary_verts, boundary_verts[0][None]], axis=0)
            boundary_lengths = np.linalg.norm(boundary_verts[1:] - boundary_verts[:-1], axis=1)
            normalized_boundary_lengths = boundary_lengths / np.sum(boundary_lengths)
            arc_lengths = [0.0]
            for i, c in enumerate(boundary_corner_idx):
                if i == len(boundary_corner_idx)-1:
                    arc_length = np.sum(normalized_boundary_lengths[boundary_corner_idx[i]:]) + np.sum(normalized_boundary_lengths[:boundary_corner_idx[0]])
                else:
                    arc_length = np.sum(normalized_boundary_lengths[boundary_corner_idx[i]:boundary_corner_idx[i+1]])
                arc_lengths.append(arc_length)            

            cell_arc_lengths.append(arc_lengths)

            ##
            uv, buv = parameterize_mesh(patch.vertices, patch.faces, boundary[boundary_corner_idx])
            patches_uv.append(uv)

            cnt += 1


    ## concatenate all the patches
    mesh = trimesh.util.concatenate(list(scene.geometry.values()))

    ## check if g is connected
    adjacency = trimesh.graph.face_adjacency(mesh.faces, mesh)
    ## save each component as a patch
    graph_connected = nx.Graph() 
    graph_connected.add_edges_from(mesh.face_adjacency) 
    groups = nx.connected_components(graph_connected)

    face_mask = []
    for c in groups:
        face_mask.append(list(c))

    ## find correspondence between corners and mesh vertices
    kdtree = KDTree(mesh.vertices)
    dist, idx = kdtree.query(corner_verts)
    topology_graph["node_ids"] = idx.tolist()

    ## normalize
    center = mesh.vertices.mean(axis=0)
    scale = np.max(np.abs(mesh.vertices))
    mesh.vertices = (mesh.vertices - center) / scale

    for cnt, patch in enumerate(patches):
        uv = patches_uv[cnt]
        ## extend the uv to 3d
        uv3d= np.concatenate([uv, np.zeros((uv.shape[0], 1))], axis=1)
        write_obj_file(f"{debug_dir}/{args.model_name}/flat_parameterization/flat_{cnt}.obj", uv3d, patch.faces)

        ## normalize patches
        patch.vertices = (patch.vertices - center) / scale
        patch.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)
        patch.export(f"{debug_dir}/{args.model_name}/parameterization/mesh_uv_{cnt}.obj")

    vedo_mesh = utils.trimesh2vedo(mesh)
    for cnt, m in enumerate(face_mask):
        vedo_mesh.cellcolors[m] = (np.array(mapper.to_rgba(cnt%20))*255)[None,:]

    ## save
    vedo_mesh.write(f"{debug_dir}/{args.model_name}/data/single/mesh.ply")
    # mesh.export(f"{debug_dir}/{args.model_name}/data/single/{output_data['mesh']}")
    ## validation
    corner_verts = mesh.vertices[topology_graph["node_ids"]]
    write_obj_file(f"{debug_dir}/{args.model_name}/data/corner_verts_on_mesh.obj", corner_verts)

    ## save data
    write_json(topology_graph, f"{debug_dir}/{args.model_name}/data/{output_data['topology_graph']}")
    write_json(cell_arc_lengths, f"{debug_dir}/{args.model_name}/data/{output_data['cell_arc_lengths']}")
    write_json(face_mask, f"{debug_dir}/{args.model_name}/data/{output_data['mask']}")

    # print("num nodes", len(topology_graph["node_ids"]))
    # ccc = set()
    # for c in topology_graph["cells"]:
    #     ccc.update(c)
    # print(ccc)
    # print("num nodes", len(ccc))

    plt = Plotter()
    for i, c in enumerate(topology_graph["cells"]):
        # write_obj_file(f"{debug_dir}/{args.model_name}/data/corner_verts_{i}.obj", corner_verts[c])
        ln = Line(corner_verts[c], c='r', closed=True, lw=5).pattern('- -', repeats=10)
        pts = Points(corner_verts[c], r=10, c='b')
        plt.add(ln)
        plt.add(pts)
    plt.show()
