from typing import Any
import os
import trimesh
import numpy as np
import networkx as nx
from tsp_solver.greedy import solve_tsp
from view_psd_data import *
from scipy.spatial.distance import cdist
import shutil
import argparse
import bisect
from PIL import Image

from loguru import logger
from mesh_data_structure.halfedge_mesh import HETriMesh
from mesh_data_structure.utils import trace_boundary_edges, close_holes, compute_distance_matrix, convert_list_to_string, simple_floodfill_label_mesh, GeoPathSolverWrapper, get_open_boundary
from mesh_data_structure.components import PatchTopo, Boundary, Cut
from igl_parameterization import compute_harmonic_scalar_field
from src.mesh_tools import split_mesh, split_mesh_by_path, floodfill_label_mesh
from cut_cylinder import preprocess, trace_path_by_samples


class MeshSegmentator():

    def __init__(
            self, 
            mesh: trimesh.Trimesh,
            mask: list,
            save_dir:str = None,
            smooth_flag: bool = False,
            smooth_deg: int = 4,
            intersection_merged_threshold: float = 0.15,
            opt_iters: int = 3,
            ) -> None:
        
        self.save_dir = save_dir
        
        self.mesh = mesh

        ## proximity
        self.pq_mesh = trimesh.proximity.ProximityQuery(self.mesh)
        self.mesh_path_solver = GeoPathSolverWrapper(self.mesh)
        self.mask = mask
        self.b_close_holes = False
        self.smooth_flag = smooth_flag
        self.smooth_deg = smooth_deg
        self.opt_iters = opt_iters

        self.patch_topo_list = [] ## list of PatchTopo class objects
        self.boundary_list = [] ## list of Boundary class objects
        self.cut_list = [] ## list of Cut class objects
        self.texture_img = Image.open(f'./assets/uv_color.png')
        self.intersection_merged_threshold = intersection_merged_threshold
        logger.info(f'Vertices num: {self.mesh.vertices.shape}; Faces num: {self.mesh.faces.shape}; Masks num: {len(mask)}')


    def save_paths(self):
        for cut in self.cut_list:
            if cut.dead:
                continue
            trimesh.PointCloud(cut.points).export(os.path.join(self.save_dir,f"cut_{cut.id}.obj"))
        for boundary in self.boundary_list:
            if boundary.dead:
                continue
            trimesh.PointCloud(boundary.points).export(os.path.join(self.save_dir,f"boundary_{boundary.id}.obj"))
    

    def save_mesh(self):
        self.mesh.export(os.path.join(self.save_dir,"mesh_new.obj"))


    def query_closeest_boundary(self, queries):
        queries = np.array(queries)
        min_distances = np.ones((queries.shape[0],)) * 1e10
        min_boundary_indices = np.ones((queries.shape[0],)) * -1
        for i, b in enumerate(self.boundary_list):
            distances, _ = b.kdtree.query(queries, k=1, return_distance=True)
            distances = distances[:,0]
            # print(distances, "   ", i)
            # print(distances.shape, min_distances.shape)
            min_boundary_indices[distances < min_distances] = i
            min_distances[distances < min_distances] = distances[distances < min_distances]
            # print(min_distances)            
            # print(min_boundary_indices)
            # input()
        return np.array(min_boundary_indices, dtype=np.int32).tolist()

    
    def build_unique_boundary(self, mask_id):
        m = np.array(self.mask[mask_id])
        he_mesh = HETriMesh()
        he_mesh.init_mesh(self.mesh.vertices, self.mesh.faces[m,:])

        tmp_mesh = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        tmp_mesh.export(os.path.join('tmp', f"mask_{mask_id}.obj"))
        tmp_boundaries = trace_boundary_edges(he_mesh)

        # patch_topo = PatchTopo(mask_id)

        for i in range(len(tmp_boundaries)):
            logger.debug(f'Check boundary {i} in mask {mask_id}.')
            tmp_boundaries[i].append(tmp_boundaries[i][0])
            boundary = np.array(tmp_boundaries[i], dtype=np.int32)[:,0]
            
            ## check if this boundary has been found before
            unique_set = set(boundary)
            for j in range(len(self.boundary_list)):
                boundary_candidate_set = set(self.boundary_list[j].boundary_vertex_indices)
                unique_set = unique_set.difference(boundary_candidate_set)
            
            if len(unique_set) > 0:
                pmask = np.zeros_like(boundary)
                for pid in range(len(boundary)):
                    if boundary[pid] in unique_set:
                        pmask[pid] = True
                
                if pmask.sum() != len(boundary):
                    logger.debug('\tPart of the boundary has been found before.')
                    new_boundary = []
                    pt_num = len(boundary)
                    pid = 0
                    while pmask[pid]:
                        pid += 1

                    while not pmask[pid]:
                        pid += 1
                    
                    new_boundary.append(boundary[(pid-1+pt_num)%pt_num])
                    while pmask[pid]:
                        new_boundary.append(boundary[pid])
                        pid = (pid + 1) % pt_num
                    new_boundary.append(boundary[pid])
                    boundary = new_boundary
                else:
                    logger.debug('\tThe complete boundary is unique.')
                    boundary = boundary.tolist()

                boundary_obj =  Boundary(self.mesh.vertices[boundary])
                boundary_obj.set_boundary_vertex_indices_to_mesh(boundary)
                boundary_obj.id = len(self.boundary_list)
                self.boundary_list.append(boundary_obj)
            else:
                logger.debug('\tBoundary already exists. Skip!')


    def build_mask_structure(self, mask_id):
        m = np.array(self.mask[mask_id])
        he_mesh = HETriMesh()
        he_mesh.init_mesh(self.mesh.vertices, self.mesh.faces[m,:])

        tmp_mesh = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        tmp_mesh.export(os.path.join('tmp', f"mask_{mask_id}.obj"))
        tmp_boundaries = trace_boundary_edges(he_mesh)

        patch_topo = PatchTopo(mask_id)

        ## classify the patch type and cut
        if len(tmp_boundaries) == 1:
            logger.info(f"disk topology --- {mask_id}")
            patch_topo.set_type('disk')
        elif len(tmp_boundaries) == 2:
            logger.info(f"annulus topology --- {mask_id}")
            patch_topo.set_type('annulus')
        elif len(tmp_boundaries) > 2:
            logger.info(f"non-disk non-annulus topology --- {mask_id}")
            patch_topo.set_type('other')
        else:
            logger.error("Error: no boundary found")

        
        for i in range(len(tmp_boundaries)):
            tmp_boundaries[i].append(tmp_boundaries[i][0])
            boundary = np.array(tmp_boundaries[i], dtype=np.int32)[:,0]
            ## check if this boundary has been found before
            b_found = -1
            boundary_set = set(boundary)
            for j in range(len(self.boundary_list)):
                boundary_candidate_set = set(self.boundary_list[j].boundary_vertex_indices)
                
                if boundary_set == boundary_candidate_set:
                    b_found = self.boundary_list[j].id
                    self.boundary_list[j].add_mask_id(mask_id)
                    break

                # intersection_res = boundary_set.intersection(boundary_candidate_set)
                # if len(intersection_res) > 0:

            if b_found == -1:
                boundary_obj =  Boundary(self.mesh.vertices[boundary])
                boundary_obj.set_boundary_vertex_indices_to_mesh(boundary.tolist())
                boundary_obj.id = len(self.boundary_list)
                boundary_obj.add_mask_id(mask_id)
                self.boundary_list.append(boundary_obj)
                patch_topo.extend_boundary_ids([boundary_obj.id])
            else:
                patch_topo.extend_boundary_ids([b_found])
        
        self.patch_topo_list.append(patch_topo)
        return patch_topo.type


    def cut_through_holes(self, mask_id):
        m = np.array(self.mask[mask_id], dtype=np.int32)
        patch_mesh = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        paths, distance_matrix = compute_distance_matrix(patch_mesh, b_close_holes=False)
        # print(distance_matrix)
        ## solve TSP
        tsp_path = solve_tsp(distance_matrix)
        logger.debug(f'TSP Path: {tsp_path}')

        throughhole_paths = []
        for k in range(len(tsp_path)):
            i = tsp_path[k]
            j = tsp_path[(k+1)%len(tsp_path)]
            keys = [f"{i},{j}", f"{j},{i}"]
            for key in keys:
                if key in paths:
                    throughhole_paths.append(paths[key])
                    break
        return throughhole_paths
    

    def cut_annulus_by_two_furthest_pts(self, mask_id):

        m = np.array(self.mask[mask_id], dtype=np.int32)
        patch_mesh = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        path_solver = GeoPathSolverWrapper(patch_mesh)
        boundary_loops = get_open_boundary(patch_mesh)

        boundary_ends = []
        # find the farthest point on the boundary
        for b_idx in range(2):
            queries = np.array(boundary_loops[b_idx], dtype=np.int32)[:,0]
            v = patch_mesh.vertices[queries]
            pair_dist = cdist(v, v)
            a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)
            a0, a1 = queries[a0], queries[a1]
            boundary_ends.append([a0, a1])
        
        boundary_ends = np.array(boundary_ends)
        v = patch_mesh.vertices[boundary_ends.flatten()]
        pair_dist = cdist(v[:2], v[2:])
        dist_sum_0 = pair_dist[0, 0] + pair_dist[1, 1]
        dist_sum_1 = pair_dist[0, 1] + pair_dist[1, 0]

        a0, a1 = boundary_ends[0]
        if dist_sum_0 > dist_sum_1:
            b1, b0 = boundary_ends[1]
        else:
            b0, b1 = boundary_ends[1]
        
        path_pts0 = path_solver.solve(a0, b0)
        path_pts1 = path_solver.solve(a1, b1)
        
        throughhole_paths = [path_pts0, path_pts1]
        
        return throughhole_paths

    
    def cut_annulus_by_one_furthest_pts_and_propogation(self, mask_id):
        bids = self.patch_topo_list[mask_id].boundary_ids
        assert len(bids) == 2, "Num of boundary_loops in annulus should be 2"
        
        found_fixed_indices = False
        for i in range(2):
            if len(self.boundary_list[bids[i]].fixed_indices) == 2:
                logger.debug(f"Boundary {bids[i]} in mask {mask_id} has two fixed points. Propogate cuts from the fixed points.")
                found_fixed_indices = True
                bidx, bidx_c = bids[i], bids[1-i]
                break
        
        if not found_fixed_indices:
            logger.debug(f"Annulus {mask_id} has no fixed points. Cut by two furthest points.")
            # find the farthest point on the boundary
            bidx, bidx_c = bids[0], bids[1]
            v = self.boundary_list[bidx].points
            pair_dist = cdist(v, v)
            a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)
            a0 = self.boundary_list[bidx].boundary_vertex_indices[a0]
            a1 = self.boundary_list[bidx].boundary_vertex_indices[a1]
        else:
            a0, a1 = tuple(self.boundary_list[bidx].fixed_indices)
        
        if len(self.boundary_list[bidx_c].fixed_indices) > 0:
            logger.warning(f"Complement boundary {bidx_c} in mask {mask_id} has fixed points. Propogate cuts from the fixed points.")
        
        a0_boundary_dist = np.inf
        a1_boundary_dist = np.inf
        a0_boundary_path = None
        a1_boundary_path = None
        a0_boundary_p = None
        a1_boundary_p = None
        for p in self.boundary_list[bidx_c].boundary_vertex_indices:

            path = self.mesh_path_solver.solve(a0, p)
            dist = [np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]
            dist = np.array(dist).sum()

            if dist < a0_boundary_dist:
                a0_boundary_dist = dist
                a0_boundary_path = path
                a0_boundary_p = p

            path = self.mesh_path_solver.solve(a1, p)
            dist = [np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]
            dist = np.array(dist).sum()

            if dist < a1_boundary_dist:
                a1_boundary_dist = dist
                a1_boundary_path = path
                a1_boundary_p = p
        
        # Update boundary fixed indices
        self.boundary_list[bidx].fixed_indices.add(a0)
        self.boundary_list[bidx].fixed_indices.add(a1)
        self.boundary_list[bidx_c].fixed_indices.add(a0_boundary_p)
        self.boundary_list[bidx_c].fixed_indices.add(a1_boundary_p)

        throughhole_paths = [a0_boundary_path, a1_boundary_path]
        
        return throughhole_paths
    

    def find_closest_point_on_boundary(self, p, boundary_pts):
        d = np.linalg.norm(p - boundary_pts, axis=1)
        min_idx = np.argmin(d)
        return min_idx, d[min_idx]
        

    def cut_annulus_by_one_furthest_pts_and_scalar_field(self, mask_id):
        bids = self.patch_topo_list[mask_id].boundary_ids
        assert len(bids) == 2, "Num of boundary_loops in annulus should be 2"
        
        if mask_id == 4:
            print(' ')
        found_fixed_indices = False
        for i in range(2):
            if len(self.boundary_list[bids[i]].fixed_indices) == 2:
                logger.debug(f"Boundary {bids[i]} in mask {mask_id} has two fixed points. Propogate cuts from the fixed points.")
                found_fixed_indices = True
                bidx, bidx_c = bids[i], bids[1-i]
                break
        
        if not found_fixed_indices:
            logger.debug(f"Annulus {mask_id} has no fixed points. Cut by two furthest points.")
            # find the farthest point on the boundary
            bidx, bidx_c = bids[0], bids[1]
            v = self.boundary_list[bidx].points
            pair_dist = cdist(v, v)
            a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)
            a0 = self.boundary_list[bidx].boundary_vertex_indices[a0]
            a1 = self.boundary_list[bidx].boundary_vertex_indices[a1]
        else:
            a0, a1 = tuple(self.boundary_list[bidx].fixed_indices)
        
        if len(self.boundary_list[bidx_c].fixed_indices) > 0:
            logger.warning(f"Complement boundary {bidx_c} in mask {mask_id} has fixed points. Propogate cuts from the fixed points.")
        
        m = np.array(self.mask[mask_id])
        mesh_local = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        
        uv, vis_size, boundary_list_local = preprocess(mesh_local, None)
        
        # pair_dist = cdist(v, v)
        # a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)

        corner_ids = [a0, a1]
        local_vids = []
        local_bidx_logs = []
        for ai in corner_ids:
            bidx_local = 0
            while bidx_local < 2:
                vids = np.array(boundary_list_local[bidx_local])
                boundary_pts_local = mesh_local.vertices[vids]
                closest_id, closest_d = self.find_closest_point_on_boundary (self.mesh.vertices[ai], boundary_pts_local)
                if closest_d > 1e-7:
                    bidx_local = 1
                    uv[:, 0] = 1 - uv[:, 0]
                else:
                    local_vids.append(vids[closest_id])
                    local_bidx_logs.append(bidx_local)
                    break
            if bidx_local == 2:
                logger.error(f"Failed to find the closest point on the local mesh for vertex {ai}.")
                return None
        
        if local_bidx_logs[0] != local_bidx_logs[1]:
            logger.error(f"Failed to find the closest point on the same local boundary for vertex {corner_ids}.")
            return None
            
        cut_path, cut_path_info = trace_path_by_samples(mesh_local, uv, local_vids[0], sample_num=20)
        # cut_path_u, cut_path_info_u = trace_path_by_samples(mesh, uv, corner_ids[1], sample_num=20)
        cut_path2 = [self.mesh.vertices[a1]]
        
        for i in range(1, len(cut_path)):
    
            eid = cut_path_info[i]['eid']
            e = mesh_local.edges_unique[eid]
            faces = mesh_local.vertex_faces[e].flatten()

            round_path = [cut_path[i]]
            trace_u = cut_path_info[i]['u']
            point_cur = {
                'x': cut_path[i],
            }
            visited_eids = set()
            path2_u_dir = cut_path[i] - cut_path[i - 1]
            path2_u_dir = path2_u_dir / np.linalg.norm(path2_u_dir)

            while True:
                
                mask = faces != -1
                unique_faces = faces[mask]
                related_edges = mesh_local.faces_unique_edges[unique_faces]
                related_edges = np.unique(related_edges)
                
                candidate_list = []
                for eid in related_edges:
                    
                    if eid in visited_eids:
                        continue

                    e = mesh_local.edges_unique[eid]
                    u0 = uv[e[0], 0]
                    u1 = uv[e[1], 0]
                    if u0 > u1:
                        e = e[::-1]
                        u0, u1 = u1, u0

                    visited_eids.add(eid)

                    if u0 <= trace_u <= u1:
                        de_u = u1 - u0

                        if abs(de_u) < 1e-7:
                            r_list = [0, 1]
                        else:
                            r_list = [(trace_u - u0) / de_u]
                        
                        for r in r_list:

                            sample_x = mesh_local.vertices[e[0]] * (1-r) + mesh_local.vertices[e[1]] * r
                            d = np.linalg.norm(sample_x - point_cur['x'])
                            if d < 1e-7:
                                continue
                            
                            point_candidate = {
                                'x': sample_x,
                                'd': d,
                                'eid': eid
                            }
                            candidate_list.append(point_candidate)
                
                if len(candidate_list) == 0:
                    break
                
                candidate_list = sorted(candidate_list, key=lambda x: x['d'], reverse=True)
                point_next = candidate_list[0]

                round_path.append(point_next['x'])
                point_cur = point_next
                
                e = mesh_local.edges_unique[point_next['eid']]
                faces = mesh_local.vertex_faces[e].flatten()
            
            # round_path.append(round_path[0])
            round_path = np.array(round_path)
            round_center = round_path.mean(axis=0)
            opposite_point = round_center - cut_path[i]
            opposite_dir = opposite_point / np.linalg.norm(opposite_point)
            
            # round_path_seg = None
            p_cos_list = []
            for j in range(len(round_path)):
                p = round_path[j]
                p_dir = p - round_center
                p_dir = p_dir / np.linalg.norm(p_dir)
                p_cos = np.dot(p_dir, opposite_dir)
                p_cos_list.append(p_cos)

            p_cos_maxidx = np.argmax(p_cos_list)
            p_cos_max = -1
            mid_point = None
            for j in range(max(0, p_cos_maxidx - 3), min(p_cos_maxidx + 3, len(round_path)-1)):
                samples = np.linspace(0, 1, 10)
                for t in samples:
                    p = round_path[j] * (1-t) + round_path[j+1] * t
                    p_dir = p - round_center
                    p_dir = p_dir / np.linalg.norm(p_dir)
                    p_cos_oppo = np.dot(p_dir, opposite_dir)
                    p_dir_u = p - cut_path2[-1]
                    p_dir_u = p_dir_u / np.linalg.norm(p_dir_u)
                    p_cos_u = np.dot(p_dir_u, path2_u_dir)
                    p_cos = p_cos_oppo + p_cos_u
                    
                    if p_cos_max < p_cos:
                        p_cos_max = p_cos
                        mid_point = p
        
            cut_path2.append(mid_point)

            # round_spheres = create_spheres(round_path, radius=vis_size, color=(0,200,0))
            # starting_sphere = create_spheres(round_path[0], radius=vis_size * 1.5, color=(200,0,200))
            # last_sphere = create_spheres(round_path[-2], radius=vis_size * 1.5, color=(0,200,200))
            # mid_sphere = create_spheres(cut_path2[-1], radius=vis_size * 1.5, color=(200,0,0))  
            # vis = [round_spheres, mid_sphere, starting_sphere, last_sphere]
            # for p in range(1, len(round_path)):
            #     lines = create_lines(round_path[p-1], round_path[p], radius=vis_size / 2, color=(0,0,200))
            #     vis.append(lines)
            # vis = trimesh.util.concatenate(vis)
            
            # vis.export(str(out_dir / f'{exp_name.stem}_debug_round_{i}.obj'))
        
        vidcs = []
        closest_idx, closest_d = self.find_closest_point_on_boundary(cut_path[-1], self.boundary_list[bidx_c].points)
        vidcs.append(self.boundary_list[bidx_c].boundary_vertex_indices[closest_idx])
        
        closest_idx, closest_d = self.find_closest_point_on_boundary(cut_path2[-1], self.boundary_list[bidx_c].points)
        vidcs.append(self.boundary_list[bidx_c].boundary_vertex_indices[closest_idx])
        
        # Update boundary fixed indices
        self.boundary_list[bidx].fixed_indices.add(a0)
        self.boundary_list[bidx].fixed_indices.add(a1)
        self.boundary_list[bidx_c].fixed_indices.add(vidcs[0])
        self.boundary_list[bidx_c].fixed_indices.add(vidcs[1])

        throughhole_paths = [cut_path, cut_path2]
        
        return throughhole_paths
    

    def cut_mask(self, mask_id, annulus_cut_flag:bool = False):
        if annulus_cut_flag:
            logger.debug(f"Cut annulus mask, mask id {mask_id}")
            # cuts = self.cut_annulus_by_two_furthest_pts(mask_id)
            # cuts = self.cut_annulus_by_one_furthest_pts_and_propogation(mask_id)
            cuts = self.cut_annulus_by_one_furthest_pts_and_scalar_field(mask_id)
        else:
            logger.debug(f"Cut mask, mask id {mask_id}")
            cuts = self.cut_through_holes(mask_id)

        cut_obj_list = []
        for cut_pts in cuts:
            cut_obj = Cut(cut_pts, mask_id)
            cut_obj_list.append(cut_obj)

            cut_obj.id = len(self.cut_list)
            self.cut_list.append(cut_obj)

            ## track cuts
            self.patch_topo_list[mask_id].extend_cut_ids([cut_obj.id])

        ## find the boundary that connects to the cuts
        for cut_obj in cut_obj_list:
            queries = cut_obj.get_endpoints()
            connected_boundary_indices = self.query_closeest_boundary(queries)
            cut_obj.set_connected_boundary_indices(connected_boundary_indices)
            for boundary_id in connected_boundary_indices:
                self.boundary_list[boundary_id].add_connected_cut_indices(cut_obj.id)
        

    def align_cuts(self, boundary_obj: Boundary):
        cut_ids = boundary_obj.connected_cut_indices
        logger.debug(f"Align cuts on Boundary {boundary_obj.id}; cut ids: {cut_ids}")

        # if boundary_obj.id == 2:
            # print('')
        extended_cut_dict = {}
        boundary_len_list = [
            np.linalg.norm(boundary_obj.points[i+1] - boundary_obj.points[i]) 
            for i in range(len(boundary_obj.points)-1)
        ]
        boundary_len = np.array(boundary_len_list).sum()
        boundary_len_accum = np.array(boundary_len_list).cumsum()
        for i in range(len(cut_ids)-1):
            for j in range(i+1, len(cut_ids)):
                
                cut0 = self.cut_list[cut_ids[i]]
                cut1 = self.cut_list[cut_ids[j]]
                if cut0.mask_id == cut1.mask_id:
                   continue
                pair_cuts = [cut0.id, cut1.id] if cut0.id < cut1.id else [cut1.id, cut0.id]
                keystr = convert_list_to_string(pair_cuts)
                if keystr not in extended_cut_dict:
                    extended_cut_dict[keystr] = {}
                    extended_cut_dict[keystr]['cutpair'] = pair_cuts
                    extended_cut_dict[keystr]['distance'] = 1e10
                    extended_cut_dict[keystr]['points'] = []

                    end_points = []
                    if cut0.connected_boundary_indices[0] == boundary_obj.id:
                        end_points.extend(reversed(cut0.get_endpoints()))
                    else:
                        end_points.extend(cut0.get_endpoints())
                    
                    if cut1.connected_boundary_indices[0] == boundary_obj.id:
                        end_points.extend(cut1.get_endpoints())
                    else:
                        end_points.extend(reversed(cut1.get_endpoints()))

                    extended_cut_dict[keystr]['boundary'] = set(cut0.connected_boundary_indices).union(set(cut1.connected_boundary_indices))
                    extended_cut_dict[keystr]['boundary'].difference_update(set([boundary_obj.id]))

                    d, vertex_ids = self.pq_mesh.vertex(np.array(end_points))
                    if vertex_ids[1] != vertex_ids[2]:
                        
                        # Option1: Computing Geodesic, requires 40GB Mem
                        # path_pts = self.mesh_path_solver.solve(*(vertex_ids[1:3].tolist()))
                        # dist = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
                        # dist = np.array(dist).sum()
                        # path_mid = path_pts[len(path_pts)//2]
                        # end_points[1] = path_mid

                        # Option2: Explicitly find the mid point on the boundary
                        vpos1 = boundary_obj.boundary_vertex_indices.index(vertex_ids[1])
                        vpos2 = boundary_obj.boundary_vertex_indices.index(vertex_ids[2])
                        vpos_min = min(vpos1, vpos2)
                        vpos_max = max(vpos1, vpos2)
                        dist = boundary_len_accum[vpos_max] - boundary_len_accum[vpos_min]
                        
                        if dist < boundary_len - dist:
                            mid_pos = (vpos_min + vpos_max) // 2
                        else:
                            dist = boundary_len - dist
                            n = len(boundary_len_list)
                            mid_pos = ((n - vpos_max + vpos_min) // 2 + vpos_max) % n
                        
                        extended_cut_dict[keystr]['distance'] = dist
                        vertex_ids[1] = boundary_obj.boundary_vertex_indices[mid_pos]
                    else:
                        extended_cut_dict[keystr]['distance'] = 0

                    vertex_ids = np.delete(vertex_ids, 2, axis=0)
                    extended_cut_dict[keystr]['points'] = []
                    for vid in range(1, 3):
                        extended_cut_dict[keystr]['points'].append(
                            self.mesh_path_solver.solve(vertex_ids[vid-1], vertex_ids[vid])
                        )

        ## sort extended_cut_dict by distance key
        extended_cut_dict = sorted(extended_cut_dict.items(), key=lambda x: x[1]['distance'])
        # print(extended_cut_dict)

        ## keep the shortest half of cuts
        cnt = 0
        max_keep = len(extended_cut_dict)//2
        selected_cut = set()
        for k, v in extended_cut_dict:

            cut0_id = v['cutpair'][0]
            if self.cut_list[cut0_id].connected_boundary_indices[0] == boundary_obj.id:
                self.cut_list[cut0_id].set_points(v['points'][0][::-1])
            else:
                self.cut_list[cut0_id].set_points(v['points'][0])

            cut1_id = v['cutpair'][1]
            if self.cut_list[cut1_id].connected_boundary_indices[0] == boundary_obj.id:
                self.cut_list[cut1_id].set_points(v['points'][1])
            else:
                self.cut_list[cut1_id].set_points(v['points'][1][::-1])

            selected_cut.add(cut0_id)
            selected_cut.add(cut1_id)

            logger.info(f'Align two cuts {v["cutpair"]}')
            cnt += 1
            if cnt >= max_keep:
                break
        

    def cut_annulus_aligned(self, patch_topo:PatchTopo):
        logger.success("cut annulus aligned")

        ## find the two boundaries
        boundary_ids = patch_topo.boundary_ids
        assert len(boundary_ids) == 2, "len of boundary_ids should be 2"
        boundary0 = self.boundary_list[boundary_ids[0]]
        boundary1 = self.boundary_list[boundary_ids[1]]
        print("boundary0", boundary0.id)
        print("boundary1", boundary1.id)

        ## find the two cuts on each boundary
        cuts_on_boundary0 = []
        for cut_id in boundary0.connected_cut_indices:
            cuts_on_boundary0.append(self.cut_list[cut_id])
            print("cut_id on b0", cut_id)
        cuts_on_boundary1 = []
        for cut_id in boundary1.connected_cut_indices:
            cuts_on_boundary1.append(self.cut_list[cut_id])
            print("cut_id on b1", cut_id)

        ## find the extended cuts and smooth them
        extended_cut_dict = {}
        for cut0 in cuts_on_boundary0:
            for cut1 in cuts_on_boundary1:
                pair_cuts = [cut0.id, cut1.id] if cut0.id < cut1.id else [cut1.id, cut0.id]
                keystr = convert_list_to_string(pair_cuts)
                if keystr not in extended_cut_dict:
                    extended_cut_dict[keystr] = {}
                    extended_cut_dict[keystr]['cutpair'] = pair_cuts
                    extended_cut_dict[keystr]['distance'] = 1e10
                    extended_cut_dict[keystr]['points'] = []
        
        ## find the shortest path between the two cuts
        path_solver = GeoPathSolverWrapper(self.mesh)
        for k, v in extended_cut_dict.items():
            print()
            print("cutpair", k)
            points = []
            cut0 = self.cut_list[v['cutpair'][0]]
            cut1 = self.cut_list[v['cutpair'][1]]

            queries = [cut0.points[0], cut0.points[-1]]
            d, _ = boundary0.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] < d[1]:
                ## flip
                points.append(cut0.points[-1]) ## far
                points.append(cut0.points[0]) ## near
            else:
                ## not flip
                points.append(cut0.points[0])
                points.append(cut0.points[-1])

            queries = [cut1.points[0], cut1.points[-1]]
            d, _ = boundary1.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] > d[1]:
                ## flip
                points.append(cut1.points[-1]) ## near
                points.append(cut1.points[0]) ## far
            else:
                ## not flip
                points.append(cut1.points[0])
                points.append(cut1.points[-1])
            
            d, vertex_ids = self.pq_mesh.vertex(np.array(points))

            path_pts = path_solver.solve_vlist(vertex_ids)
            dist = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
            dist = np.array(dist).sum()
            v['distance'] = dist
            v['points'] = path_pts
            # print(vertex_ids, dist)
            # trimesh.PointCloud(path_pts).export(os.path.join(self.save_dir,f"align_cuts_{k}.obj"))

        ## sort extended_cut_dict by distance key
        extended_cut_dict = sorted(extended_cut_dict.items(), key=lambda x: x[1]['distance'])

        ## keep the shortest half of cuts
        cnt = 0
        max_keep = len(extended_cut_dict)//2
        for k, v in extended_cut_dict:
            cut = Cut(v['points'], -1)
            cut.id = len(self.cut_list)
            self.cut_list.append(cut)

            for cut_id in v['cutpair']:
                self.cut_list[cut_id].set_dead()

            cnt += 1
            if cnt >= max_keep:
                break
    
    def split_mesh_with_boundaries(self):
        for cut in self.boundary_list:
            new_mesh, new_mask = split_mesh(self.mesh, cut.points, self.mesh.faces, self.intersection_merged_threshold)
            self.mesh = new_mesh
            self.mask = new_mask

        self.mesh_path_solver = GeoPathSolverWrapper(self.mesh)
        self.pq_mesh = trimesh.proximity.ProximityQuery(self.mesh)


    def split_mesh_with_cuts(self):
        for cut in self.cut_list:
            if cut.dead:
                continue
            new_mesh, new_mask = split_mesh(self.mesh, list(cut.points), self.mesh.faces, self.intersection_merged_threshold)
            self.mesh = new_mesh
            self.mask = new_mask

        self.mesh_path_solver = GeoPathSolverWrapper(self.mesh)
        self.pq_mesh = trimesh.proximity.ProximityQuery(self.mesh)


    def smooth_boundaries(self):
        logger.info('Smooth boundaries.')
        for i in range(len(self.boundary_list)):

            # print("boundary", i, self.boundary_list[i].points)
            if len(self.boundary_list[i].fixed_indices) > 0:

                fixed_pts = self.mesh.vertices[list(self.boundary_list[i].fixed_indices)]
                print('fixed_pts', fixed_pts)

                boundary_loop_flag = False
                d = np.linalg.norm(self.boundary_list[i].points[0] - self.boundary_list[i].points[-1])
                if d < 1e-7:
                    boundary_loop_flag = True

                for _ in range(self.opt_iters):
                    
                    pt_num = len(self.boundary_list[i].points)
                    logger.debug(f'\tBoundary {i} has {pt_num} points.')
                    if pt_num < 20:
                        logger.debug(f'\t< 20. Skip smoothing.')
                        break
                    
                    d = np.linalg.norm(fixed_pts[np.newaxis, :] - self.boundary_list[i].points[:, np.newaxis], axis=-1)
                    fixed_boundary_ids = np.argmin(d, axis=0).tolist()
                    if boundary_loop_flag:
                        fixed_boundary_ids.append(fixed_boundary_ids[0] + pt_num)
                        sampled_ids = [fixed_boundary_ids[0]]
                    else:
                        sampled_ids = [0]
                        if fixed_boundary_ids[0] > 0:
                            sampled_ids.append(fixed_boundary_ids[0])

                    for fixed_idx in range(1, len(fixed_boundary_ids)):
                        fixed_id0 = fixed_boundary_ids[fixed_idx-1]
                        fixed_id1 = fixed_boundary_ids[fixed_idx]
                        sample_num = min(self.smooth_deg, fixed_id1 - fixed_id0 - 1)
                        if sample_num > 0:
                            sampled_ids_interval = np.linspace(fixed_id0, fixed_id1, sample_num, dtype=np.int32)
                            step = (fixed_id1 - fixed_id0) // (sample_num + 1)
                            sampled_ids_interval = sampled_ids_interval + np.random.randint(-step//2, step//2, sample_num)
                            sampled_ids_interval = sampled_ids_interval.tolist()
                            sampled_ids.extend(sampled_ids_interval[1:-1])
                        sampled_ids.append(fixed_id1)

                    print('xxx', sampled_ids)
                    if not boundary_loop_flag and sampled_ids[-1] != pt_num - 1:
                        sampled_ids.append(pt_num - 1)

                    sampled_ids = np.array(sampled_ids) % pt_num
                    print('sampled_ids', sampled_ids)
                    
                    sampled_pts = self.boundary_list[i].points[sampled_ids]
                    d, vertex_ids = self.pq_mesh.vertex(np.array(sampled_pts))
                    new_points = []
                    for j in range(1, len(vertex_ids)):
                        if vertex_ids[j-1] == vertex_ids[j]:
                            path_pts = [self.mesh.vertices[vertex_ids[j]]]
                        else:
                            path_pts = self.mesh_path_solver.solve(vertex_ids[j-1], vertex_ids[j])
                        if j == 1:
                            new_points.extend(path_pts) # Add the first pt to make a loop
                        else:
                            new_points.extend(path_pts[1:])

                    self.boundary_list[i].points = np.array(new_points)
            else:
                for _ in range(self.opt_iters):
                    pt_num = len(self.boundary_list[i].points)
                    if pt_num < 20:
                        logger.debug(f'\t< 20. Skip smoothing.')
                        break

                    sample_pt_num = min(max(self.smooth_deg, int(pt_num * 0.2)), pt_num)
                    logger.debug(f'\tBoundary {i} has {pt_num} points. Sample {sample_pt_num} points.')
                    sampled_ids = np.linspace(0, pt_num-1, sample_pt_num, dtype=np.int32).tolist()
                    print('init', sampled_ids)
                    for k in range(1, len(sampled_ids) - 1):
                        new_sample_id = np.random.randint(
                            int(sampled_ids[k-1] * 0.2 + sampled_ids[k] * 0.8), 
                            int(sampled_ids[k] * 0.8 + sampled_ids[k+1] * 0.2)
                        )
                        sampled_ids[k] = new_sample_id
                    print('add noise', sampled_ids)
                    
                    sampled_pts = self.boundary_list[i].points[sampled_ids]
                    d, vertex_ids = self.pq_mesh.vertex(np.array(sampled_pts))
                    new_points = []
                    for j in range(1, len(vertex_ids)):
                        if vertex_ids[j-1] == vertex_ids[j]:
                            path_pts = [self.mesh.vertices[vertex_ids[j]]]
                        else:
                            path_pts = self.mesh_path_solver.solve(vertex_ids[j-1], vertex_ids[j])
                        if j == 1:
                            new_points.extend(path_pts) # Add the first pt to make a loop
                        else:
                            new_points.extend(path_pts[1:])

                    self.boundary_list[i].points = np.array(new_points)
                # self.boundary_list[i].boundary_vertex_indices = []
                # for pt in new_points:
                #     d, vid = self.pq_mesh.vertex(pt)
                #     print(pt, d, vid)
                #     self.boundary_list[i].boundary_vertex_indices.append(vid)


    def __call__(self, b_close_holes:bool = None):
        if b_close_holes is not None:
            self.b_close_holes = b_close_holes
        
        ## floodfild the mask
        mask_connected = []
        for i in range(len(self.mask)):
            m = np.array(self.mask[i])
            mask_connected.extend(simple_floodfill_label_mesh(self.mesh, m))
        self.mask = mask_connected

        if self.smooth_flag:
            for i in range(len(self.mask)):
                self.build_unique_boundary(i)
                # self.build_mask_structure(i)

            for i in range(len(self.boundary_list)):
                bi = set(self.boundary_list[i].boundary_vertex_indices)
                for j in range(i+1, len(self.boundary_list)):
                    bj = set(self.boundary_list[j].boundary_vertex_indices)
                    inter_pids = bi.intersection(bj)
                    if len(inter_pids) == 0:
                        continue
                    
                    logger.info(f'Boundary {i} and {j} share {len(inter_pids)} points. Keep them.')

                    for inter_pid in inter_pids:
                        self.boundary_list[i].fixed_indices.add(inter_pid)
                        self.boundary_list[j].fixed_indices.add(inter_pid)

            self.smooth_boundaries()
            self.split_mesh_with_boundaries()

            # Recompute the mask
            picked_pt_ids = []
            for boundary in self.boundary_list:
                _, vids = self.pq_mesh.vertex(boundary.points)
                picked_pt_ids.append(vids.tolist())

            logger.info(f'Add boundaries {len(self.boundary_list)} for floodfill.')
            # convert boundary edges to mesh edges
            self.mask = floodfill_label_mesh(self.mesh, set(), picked_pt_ids)


        print('Num of mask', len(self.mask))
        cut_mesh_flag = True
        align_flag = True
        if cut_mesh_flag:
            # Build boundary and mask patches, Cut masks
            self.boundary_list = []
            self.patch_topo_list = []
            for i in range(len(self.mask)):
                mask_type = self.build_mask_structure(i)
                print(i, self.patch_topo_list[i].extend_boundary_ids)
                if mask_type == 'other':
                    self.cut_mask(i)
                elif mask_type == 'annulus':
                    self.cut_mask(i, annulus_cut_flag=True)

            # Align endpoints

            if align_flag:
                for boundary_obj in self.boundary_list:
                    # if boundary_obj.id != 6:
                        # continue
                    mask_ids = list(boundary_obj.mask_ids)
                    if len(mask_ids) != 2:
                        logger.warning(f'A boundary doesn\'t connect to two patches! {mask_ids}. Skip!')
                        continue            
                    if self.patch_topo_list[mask_ids[0]].type == 'disk' or self.patch_topo_list[mask_ids[1]].type == 'disk':
                        logger.info('One patch is a disk. Skip!')
                        continue
                    else:
                        logger.info(f'Align a {self.patch_topo_list[mask_ids[0]].type} and {self.patch_topo_list[mask_ids[1]].type}')
                        if self.patch_topo_list[mask_ids[0]].type == 'other':
                            print("mask_ids[0]", mask_ids[0])
                        self.align_cuts(boundary_obj)

            self.split_mesh_with_cuts()

            ## 
            picked_pt_ids = []
            for cut in self.cut_list:
                if cut.dead:
                    continue
                _, vids = self.pq_mesh.vertex(cut.points)
                picked_pt_ids.append(vids.tolist())
            for boundary in self.boundary_list:
                _, vids = self.pq_mesh.vertex(boundary.points)
                picked_pt_ids.append(vids.tolist())

            logger.info(f'Add cuts {len(self.cut_list)} and boundaries {len(self.boundary_list)} for floodfill.')
            ## convert cut edges to mesh edges
            self.mask = floodfill_label_mesh(self.mesh, set(), picked_pt_ids)


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

    segmentor = MeshSegmentator(mesh, mask, save_dir)
    segmentor(b_close_holes=False)
    segmentor.save_paths()
    segmentor.save_mesh()
