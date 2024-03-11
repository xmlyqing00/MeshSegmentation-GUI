import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import igl
import trimesh
from src.io_tools import find_nan_np, read_json, write_obj_file
from mesh_data_structure.build_complex import ComplexBuilder

from scipy.spatial import KDTree
from loguru import logger


def parameterize_mesh(v, f):

    bnd = igl.boundary_loop(f)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(v, bnd)

    uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    try:
        find_nan_np(uv, "uv")
    except AssertionError:
        np.savetxt("uv.txt", uv)
        AssertionError
        
    return uv, bnd_uv


class PHComplex():
    def __init__(self, base_mesh, mask, threshold=0.00001) -> None:

        self.threshold = threshold ## for connecting patches

        self.base_mesh = base_mesh
        self.mask = mask
        self.num_patches = len(mask)


        ## 1) get patches
        self.patches = []
        for i in range(self.num_patches):
            print(f"patch {i} {len(self.mask[i])}")
            patch = self.base_mesh.submesh([self.mask[i]], append=True)
            self.patches.append(patch)


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

            ## corners as the points with connectivity changes
            v2p_sum = np.sum(v2p_connectivity, axis=1)
            ## differential v2p_sum
            diff_v2p_sum = np.diff(v2p_sum) 
            corners = np.where(diff_v2p_sum < 0)[0]
            corner_vertices = patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"][corners]]
            patch_bnd["corners"] = corners

            write_obj_file(f"patch_{i}_{len(corner_vertices)}.obj", patch_bnd["mesh"].vertices, patch_bnd["mesh"].faces)
            # write_obj_file(f"corner_{i}_{len(corner_vertices)}.obj", corner_vertices)

        ## 2.4) Identify corner points of the scaffold
        for i, patch_bnd in enumerate(patch_boundaries):
            print(i)
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
            true_corner_vertices = patch_bnd["mesh"].vertices[patch_bnd["bnd_indices"][true_corners]]
            write_obj_file(f"true_corner_{i}_{len(true_corner_vertices)}.obj", true_corner_vertices, patch_bnd["mesh"].faces)
            

if __name__ == "__main__":
    base_mesh = trimesh.load("data/mydata/ph_boat_3/single/mesh.obj")
    mask = read_json("data/mydata/ph_boat_3/mask.json")

    ph_complex = PHComplex(base_mesh, mask)
    ph_complex.init_scaffold_vertices()
    print("done")
        


