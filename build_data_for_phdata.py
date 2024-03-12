import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import igl
import trimesh
from src.io_tools import find_nan_np, read_json, write_obj_file
from vedo import Mesh, write, utils

from scipy.spatial import KDTree
from loguru import logger
import shutil
import argparse
import matplotlib
from matplotlib import cm

def parameterize_mesh(v, f):

    bnd = igl.boundary_loop(f)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bnd)

    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    # try:
    #     find_nan_np(uv, "uv")
    # except AssertionError:
    #     np.savetxt("uv.txt", uv)
    #     AssertionError
        
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




def read_curvenet_file(filename, data_type):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
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
                edge_pts = [int(id) for id in l[1:]]
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
                edge_pts = [int(id) for id in l[1:]]
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

    debug_dir = "debug"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    # vertices, edge_collection = read_curvenet_file(curvenet_file, data_type)
    # print(len(vertices), len(edge_collection))
    # for i, edges in enumerate(edge_collection):
    #     edge_verts = []
    #     if data_type == 2:
    #         for e in edges:
    #             edge_verts.append(vertices[e[0]])
    #         edge_verts.append(vertices[edges[-1][1]])
    #         edge_verts = np.array(edge_verts)
    #     else:
    #         for e in edges:
    #             edge_verts.append(vertices[e])
    #         edge_verts = np.array(edge_verts)
 
    #     write_obj_file(f"{debug_dir}/edge_verts_{i}.obj", edge_verts)


    scene = trimesh.load(model_file, process=False, maintain_order=True)
    
    assert isinstance(scene, trimesh.Scene)
    cnt = 0
    for k, g in scene.geometry.items():
        ## check if g is connected
        comps = trimesh.graph.connected_components(g.edges)
        print(len(comps))
        ## save each component as a patch

        ##
        uv, buv = parameterize_mesh(g.vertices, g.faces)
        ## extend 2d uv to 3d
        uv = np.concatenate([uv, np.zeros((uv.shape[0], 1))], axis=1)

        write_obj_file(f"{debug_dir}/patch_{cnt}.obj", g.vertices, g.faces)
        write_obj_file(f"{debug_dir}/uv_{cnt}.obj", uv, g.faces)
  
        cnt += 1

    # mask = read_json(f"data/phdata/boat/mask.json")

    # ph_complex = PHComplex(base_mesh, mask, debug=True)
    # ph_complex.init_scaffold_vertices()

        


