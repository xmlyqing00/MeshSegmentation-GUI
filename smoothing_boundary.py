from mesh_data_structure.halfedge_mesh import HETriMesh
import trimesh
from shutil import copyfile
import numpy as np
import networkx as nx
from PIL import Image

import argparse
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import time

from copy import deepcopy

def write_obj_file(filename, V, F=None, C=None, N=None, vid_start=1):
    with open(filename, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        
        if N is not None:
            for Ni in N:
                f.write(f"vn {Ni[0]} {Ni[1]} {Ni[2]}\n")
                  
        if F is not None:
            for Fi in F:
                f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start}\n")


class LlyodRelax():

    def __init__(self, mesh, num_iter=10):
        self.num_iter = num_iter

        ## data
        self.vertices = mesh.vertices ## to be updated
        self.vertices = np.concatenate([self.vertices, np.zeros((1, 3))], axis=0) ## last one is dummy
        
        ## one ring neighbors
        g = nx.from_edgelist(mesh.edges_unique)

        # one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]
        one_ring = np.zeros((len(self.vertices), 50), dtype=np.int32) -1  ## last one (-1) is dummy
        max_val = 0
        for i in range(len(mesh.vertices)):
            one_ring[i, :len(list(g[i].keys()))] = list(g[i].keys())
            max_val = max(len(list(g[i].keys())), max_val)

        self.one_ring = one_ring[:, :max_val]
        self.one_ring_count = np.sum(self.one_ring != -1, axis=-1) ## last one (-1) is dummy
        self.one_ring_count[-1] = 1 ## last one (-1) is dummy

        ## boundary vertices
        unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        bvids = np.unique(unique_edges)
        self.constraint_ids = bvids

    def set_vertices(self, vertices):

        if vertices.shape[0] != self.vertices.shape[0] - 1:
            print(vertices.shape, self.vertices.shape)
            raise ValueError("vertices shape not match")
        
        self.vertices = vertices
        self.vertices = np.concatenate([self.vertices, np.zeros((1, 3))], axis=0)

    def get_vertices(self):
        return self.vertices[:-1]
    
    def set_fixed_vertices(self, fixed_vertices):
        constraint_ids = list(self.constraint_ids) + list(fixed_vertices)
        self.constraint_ids = np.array(constraint_ids, dtype=np.int32)
    
    def _lloyd_relax(self):
        old_vertices = self.vertices
        new_vertices = np.zeros_like(old_vertices)
        new_vertices = old_vertices[self.one_ring].sum(axis=1) 
        new_vertices /= self.one_ring_count[:, None]
        new_vertices[self.constraint_ids] = old_vertices[self.constraint_ids]
        # displacement = np.linalg.norm(new_vertices - old_vertices, axis=-1)
        self.vertices = new_vertices
        return new_vertices[:-1]
    
    def run(self, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iter
        for i in range(num_iters):
            self._lloyd_relax()

def laplacian_smooth_mesh(mesh:trimesh.Trimesh, boundary_vertex_ids:list, num_iter=10):

    ## get 1-ring
    g = nx.from_edgelist(mesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]


    ## get k-ring vertices of boundary vertices
    boundary_vertex_ids = np.unique(boundary_vertex_ids) ## remove duplicated
    vertex_list = boundary_vertex_ids
    neighborhood_size = 3
    for i in range(neighborhood_size):
        ## get 1-ring neighbors of vertices in vertex_set
        neighbor_set = set()
        for v in vertex_list:
            neighbor_set.update(one_ring[v])
        vertex_list = list(neighbor_set)

    faces_set = set()
    vertex2faces = trimesh.geometry.vertex_face_indices(len(mesh.vertices), mesh.faces, mesh.faces_sparse)
    for v in vertex_list:
        faces = vertex2faces[v]
        faces_set.update(faces)

    faces_set.remove(-1)
    face_list = list(faces_set)
    faces = mesh.faces[face_list]
    vertex_indices = np.unique(faces)

    ## replace vertices in faces with new indices
    new_faces = np.zeros_like(faces) - 1
    new_boundary_vertex_ids = []
    for i, v in enumerate(vertex_indices):
        new_faces[faces == v] = i
        if v in boundary_vertex_ids:
            new_boundary_vertex_ids.append(i)

    ## make a boundary mesh for smoothing
    new_vertices = mesh.vertices[vertex_indices]
    boundary_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False, maintain_order=True)
    boundary_mesh.export("boundary_mesh.obj")

    smoother = LlyodRelax(boundary_mesh, num_iter=3)
    smoother.set_fixed_vertices(new_boundary_vertex_ids)
    smoother.run()
    updated_vertices = smoother.get_vertices()

    ## projected to original mesh
    pq_mesh = trimesh.proximity.ProximityQuery(mesh)
    closest, _, _ = pq_mesh.on_surface(updated_vertices)

    # ## update the original mesh with the smoothed boundary vertices
    all_vertices = mesh.vertices
    all_vertices[vertex_indices] = closest
    outmesh = trimesh.Trimesh(vertices=all_vertices, faces=mesh.faces, process=False, maintain_order=True)
    return outmesh


"""
boundary_curves: list of list of vertex ids
boundary_curve: a list of sorted vertex ids
"""
def boundary_resampling(mesh:trimesh.Trimesh, boundary_curves:list, is_closed=False):

    ## for visualization
    resampled_vis = [] 

    for bcurve in boundary_curves:
        sorted_vertices = mesh.vertices[bcurve]
        distance = np.linalg.norm(sorted_vertices - np.roll(sorted_vertices, 1, axis=0), axis=-1)
        cum_distance = np.cumsum(distance, axis=0)
        cum_distance = cum_distance / cum_distance[-1] ## normalize to 0, 1
        resampled_vertices = np.zeros_like(sorted_vertices)
        resampled_vertices[0] = sorted_vertices[0]
        resampled_vertices[-1] = sorted_vertices[-1]

        ## laplacian smoothing to get a better parameterization
        t = np.zeros_like(cum_distance)
        t[0] = 0
        t[-1] = 1    
        t[1:-1] = (cum_distance[2:] + cum_distance[:-2]) / 2

        # ## uniform parameterization
        # t = np.linspace(0, 1, len(sorted_vertices))
        # t = t.reshape(cum_distance.shape)
        
        ## resampling by linear interpolation 
        for i, ti in enumerate(t):
            if i == 0 or i == len(t) - 1:
                continue
            idx = np.where(cum_distance > ti)[0][0]
            idx0 = idx - 1
            idx1 = idx
            t0 = cum_distance[idx0]
            t1 = cum_distance[idx1]
            w0 = (t1 - ti) / (t1 - t0)
            w1 = (ti - t0) / (t1 - t0)
            resampled_vertices[i] = w0 * sorted_vertices[idx0] + w1 * sorted_vertices[idx1]
        mesh.vertices[bcurve[1:-1]] = resampled_vertices[1:-1]

        ## for visualization
        resampled_vis.extend(resampled_vertices)

    updated = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, maintain_order=True)
    return updated, resampled_vis
          



def edge_flip2(mesh:trimesh.Trimesh, boundary_vertex_ids:list):

    boundary_vertex_ids = np.unique(boundary_vertex_ids) ## remove duplicated

    faces_set = set()
    vertex2faces = trimesh.geometry.vertex_face_indices(len(mesh.vertices), mesh.faces, mesh.faces_sparse)
    for v in boundary_vertex_ids:
        faces = vertex2faces[v]
        faces_set.update(faces)

    faces_set.remove(-1)
    face_list = list(faces_set)
    faces = mesh.faces[face_list]
    vertex_indices = np.unique(faces)

    ## replace vertices in faces with new indices
    new_faces = np.zeros_like(faces) - 1
    new_boundary_vertex_ids = []
    face_id_map = {}
    face_id_reverse_map = {}
    for i, v in enumerate(vertex_indices):
        new_faces[faces == v] = i
        face_id_map[v] = i
        face_id_reverse_map[i] = v
        if v in boundary_vertex_ids:
            new_boundary_vertex_ids.append(i)

    ## make a boundary mesh for edge_flip
    new_vertices = mesh.vertices[vertex_indices]
    boundary_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False, maintain_order=True)

    ## flip edges
    flipped_boundary_mesh = edge_flip(boundary_mesh, new_boundary_vertex_ids)

    ## replace the faces in the original mesh
    new_faces = flipped_boundary_mesh.faces
    for i in range(len(new_faces)):
        for j in range(3):
            new_faces[i, j] = face_id_reverse_map[new_faces[i, j]]
    mesh.faces[face_list] = new_faces
    return mesh

def edge_flip3(segmented_mesh: trimesh.Trimesh, boundary_pts: list):
    f_adj_list = segmented_mesh.face_adjacency
    fe_adj_list = np.sort(segmented_mesh.face_adjacency_edges, axis=1)

    flip_edge_list = []
    num_fe = len(fe_adj_list)
    for i in tqdm(range(num_fe), desc='Edge flip'):
        both_on_boundary = True
        f_adj = f_adj_list[i]
        fe_adj = fe_adj_list[i]
        for fe_adj_pt in fe_adj:
            if fe_adj_pt not in boundary_pts:
                both_on_boundary = False
        
        ## skip edge flip if both endpoints are on the boundary
        if both_on_boundary:
            continue
        
        pt_on_edge = fe_adj
        pt_all = np.unique(segmented_mesh.faces[f_adj].flatten())
        pt_off_edge = np.setdiff1d(pt_all, pt_on_edge, assume_unique=True)

        angle_on_edge_sum = 0
        for pt_id in pt_on_edge:
            pt = segmented_mesh.vertices[pt_id]
            neighbor_pts = segmented_mesh.vertices[pt_off_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_on_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))

        angle_off_edge_sum = 0
        for pt_id in pt_off_edge:
            pt = segmented_mesh.vertices[pt_id]
            neighbor_pts = segmented_mesh.vertices[pt_on_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_off_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))

        if angle_on_edge_sum * 1.5 < angle_off_edge_sum:
            flip_edge_list.append({
                'f_adj': f_adj,
                'angle_ratio': angle_on_edge_sum / angle_off_edge_sum,
                'pt_all': pt_all,
                'pt_on_edge': pt_on_edge,
            })
    
    logger.info(f'Flip edge number: {len(flip_edge_list)}')
    
    flip_edge_list = sorted(flip_edge_list, key=lambda x: x['angle_ratio'], reverse=True)
    
    faces = np.asarray(segmented_mesh.faces)
    face_mask = np.ones(len(faces), dtype=bool)
    for flip_edge in tqdm(flip_edge_list, desc='Edge flip'):
        fid0 = flip_edge['f_adj'][0]
        fid1 = flip_edge['f_adj'][1]
        if face_mask[fid0] and face_mask[fid1]:
            face_mask[fid0] = False
            face_mask[fid1] = False

            pt_rest = np.setdiff1d(flip_edge['pt_all'], faces[fid0], assume_unique=True)
            m = faces[fid0] == flip_edge['pt_on_edge'][0]
            faces[fid0][m] = pt_rest[0]

            pt_rest = np.setdiff1d(flip_edge['pt_all'], faces[fid1], assume_unique=True)
            m = faces[fid1] == flip_edge['pt_on_edge'][1]
            faces[fid1][m] = pt_rest[0]

    segmented_mesh.faces = faces

    return segmented_mesh

def edge_flip(segmented_mesh: trimesh.Trimesh, boundary_pts: list):
    
    f_adj_list = segmented_mesh.face_adjacency
    fe_adj_list = np.sort(segmented_mesh.face_adjacency_edges, axis=1)

    flip_edge_list = []
    for f_adj, fe_adj in zip(f_adj_list, fe_adj_list):
        endpoint_on_boundary = []
        for fe_adj_pt in fe_adj:
            if fe_adj_pt in boundary_pts:
                endpoint_on_boundary.append(True)
            else:
                endpoint_on_boundary.append(False)
        
        if np.sum(endpoint_on_boundary) != 1:
            continue
        
        pt_on_edge = fe_adj
        pt_all = np.unique(segmented_mesh.faces[f_adj].flatten())
        pt_off_edge = np.setdiff1d(pt_all, pt_on_edge, assume_unique=True)

        angle_on_edge_sum = 0
        for pt_id in pt_on_edge:
            pt = segmented_mesh.vertices[pt_id]
            neighbor_pts = segmented_mesh.vertices[pt_off_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_on_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))

        angle_off_edge_sum = 0
        for pt_id in pt_off_edge:
            pt = segmented_mesh.vertices[pt_id]
            neighbor_pts = segmented_mesh.vertices[pt_on_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_off_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))

        if angle_on_edge_sum * 1.5 < angle_off_edge_sum:
            flip_edge_list.append({
                'f_adj': f_adj,
                'angle_ratio': angle_on_edge_sum / angle_off_edge_sum,
                'pt_all': pt_all,
                'pt_on_edge': pt_on_edge,
            })
    
    logger.info(f'Flip edge number: {len(flip_edge_list)}')
    
    flip_edge_list = sorted(flip_edge_list, key=lambda x: x['angle_ratio'], reverse=True)
    
    faces = np.asarray(segmented_mesh.faces)
    face_mask = np.ones(len(faces), dtype=bool)
    for flip_edge in tqdm(flip_edge_list, desc='Edge flip'):
        fid0 = flip_edge['f_adj'][0]
        fid1 = flip_edge['f_adj'][1]
        if face_mask[fid0] and face_mask[fid1]:
            face_mask[fid0] = False
            face_mask[fid1] = False

            pt_rest = np.setdiff1d(flip_edge['pt_all'], faces[fid0], assume_unique=True)
            m = faces[fid0] == flip_edge['pt_on_edge'][0]
            faces[fid0][m] = pt_rest[0]

            pt_rest = np.setdiff1d(flip_edge['pt_all'], faces[fid1], assume_unique=True)
            m = faces[fid1] == flip_edge['pt_on_edge'][1]
            faces[fid1][m] = pt_rest[0]

    segmented_mesh.faces = faces

    return segmented_mesh


def refinement(
        segmented_mesh: trimesh.Trimesh, 
        boundary_curves: list, 
        iters: int = 0, 
        resample_boundary: bool = False,
        edge_flip_flag: bool = False, 
        laplician_flag: bool=False,
    ):

    boundary_vertex_ids = []
    for bcurve in boundary_curves:
        boundary_vertex_ids.extend(bcurve)

    viz_list = []
    for iter_idx in range(iters):

        logger.info(f'Iteration {iter_idx + 1} / {iters}')

        if resample_boundary:
            ## NO NEED TO RE-TRACE THE BOUNDARY POINTS; THE ORDER SHOULD MAINTAIN
            t0 = time.time()
            segmented_mesh, _ = boundary_resampling(segmented_mesh, boundary_curves) ## boundary vertex ids
            t1 = time.time()
            logger.info(f'boundary_resampling time: {t1 - t0}')
            segmented_mesh.export(f'boundary_resampled_mesh_{iter_idx}.obj')
            # import matplotlib
            # from matplotlib import cm
            # norm = matplotlib.colors.Normalize(0, 1, clip=True)
            # mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
            # colors = np.linspace(0,1,len(resampled_vertices))
            # colors = mapper.to_rgba(colors)[:,:3]
            # write_obj_file(f"resampled_boundary_{iter_idx}.obj", resampled_vertices, C=colors)

        if edge_flip_flag:
            t0 = time.time()
            segmented_mesh = edge_flip2(segmented_mesh, boundary_vertex_ids)
            # segmented_mesh = edge_flip(segmented_mesh, boundary_vertex_ids)
            t1 = time.time()
            logger.info(f'edge_flip time: {t1 - t0}')
            segmented_mesh.export(f'edgeflip_mesh_{iter_idx}.obj')

        if laplician_flag:
            t0 = time.time()
            segmented_mesh = laplacian_smooth_mesh(segmented_mesh, boundary_vertex_ids, num_iter=3)
            t1 = time.time()
            logger.info(f'laplacian_smooth_mesh time: {t1 - t0}')
            segmented_mesh.export(f'smoothed_mesh_{iter_idx}.obj')

    t0 = time.time()
    segmented_mesh = edge_flip3(segmented_mesh, boundary_vertex_ids)
    t1 = time.time()
    logger.info(f'edge_flip3 time: {t1 - t0}')

    return segmented_mesh, viz_list

def trace_boundary_curves(boundary_edges):

    ## find intersection points (those appear more than twice)
    vertex_count = {}
    bifurcating_vertex_ids = []
    for edge in boundary_edges:
        if edge[0] not in vertex_count:
            vertex_count[edge[0]] = 0
        if edge[1] not in vertex_count:
            vertex_count[edge[1]] = 0
        vertex_count[edge[0]] += 1
        vertex_count[edge[1]] += 1

        if vertex_count[edge[0]] > 2 and edge[0] not in bifurcating_vertex_ids:
            bifurcating_vertex_ids.append(edge[0])
        if vertex_count[edge[1]] > 2 and edge[1] not in bifurcating_vertex_ids:
            bifurcating_vertex_ids.append(edge[1])

    # bifurcating_count = {}
    # for vid in bifurcating_vertex_ids:
    #     bifurcating_count[vid] = vertex_count[vid]
    # print(bifurcating_count)

    boundary_curves = []
    boundary_edges = list(boundary_edges)
    while len(boundary_edges) > 0:

        ## initialize a curve        
        curve = []
        for edge in boundary_edges:
            if edge[0] in bifurcating_vertex_ids:
                boundary_edges.remove(edge)
                curve = [edge[0], edge[1]]
                break
            elif edge[1] in bifurcating_vertex_ids:
                boundary_edges.remove(edge)
                curve = [edge[1], edge[0]]
                break
        ## trace the curve until encountering a bifurcating vertex
        while True:
            for edge in boundary_edges:
                if curve[-1] == edge[0]:
                    boundary_edges.remove(edge)
                    curve.append(edge[1])
                    break
                elif curve[-1] == edge[1]:
                    boundary_edges.remove(edge)
                    curve.append(edge[0])
                    break
            if curve[-1] in bifurcating_vertex_ids:
                break
        
        ## store
        boundary_curves.append(curve)

    return boundary_curves, bifurcating_vertex_ids
    

def apply_mask(segmented_mesh: trimesh.Trimesh, mask: list):

    face_patches = -1 + np.zeros(len(segmented_mesh.faces), dtype=np.int32)
    for i, seg in enumerate(mask):
        for fid in seg:
            face_patches[fid] = i
    
    group_num = len(mask)
    for i in range(len(face_patches)):
        if face_patches[i] == -1:
            logger.info(f'Found a single face. Face id: {i}')
            face_patches[i] = group_num
            group_num += 1
            mask.append([i])

    f_adj = segmented_mesh.face_adjacency
    fe_adj = np.sort(segmented_mesh.face_adjacency_edges, axis=1)

    boundary_edges = set()
    boundary_pts = set()
    for i in range(len(f_adj)):
        if face_patches[f_adj[i][0]] != face_patches[f_adj[i][1]]:
            boundary_pts.add(fe_adj[i][0])
            boundary_pts.add(fe_adj[i][1])
            boundary_edges.add((fe_adj[i][0], fe_adj[i][1]))
    
    logger.info(f'Patch number: {len(mask)}')
    logger.info(f'Boundary point number: {len(boundary_pts)}')

    return boundary_edges, boundary_pts

    # unique_edges = segmented_mesh.edges[trimesh.grouping.group_rows(segmented_mesh.edges_sorted, require_count=1)]
    # logger.info(f'Edge number of open boundary: {unique_edges.shape[0]}')

def import_mesh_mask(outdir: Path):
    segmented_mesh_path = outdir / 'segmented_mesh.ply'
    segmented_mesh = trimesh.load(segmented_mesh_path, process=False, maintain_order=True)
    
    mask_path = outdir / 'mask.json'
    with open(mask_path, 'r') as f:
        mask = json.load(f)

    return segmented_mesh, mask

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--outdir', type=str, default='./output', help='Output directory.')
    parser.add_argument('--iters', type=int, default=1, help='Refinement iterations.')
    parser.add_argument('--boundary-resample', action='store_true', help='Resample the boundary.')
    parser.add_argument('--edge-flip', action='store_true', help='Flip the edge.')
    parser.add_argument('--laplacian', action='store_true', help='Apply laplacian smooth.')
    args = parser.parse_args()    
    logger.info(f'Arguments: {args}')

    outdir = Path(args.outdir)
    segmented_mesh, mask = import_mesh_mask(outdir)

    color_img = np.asarray(Image.open('assets/cm_tab20.png').convert('RGBA'))
    cmap = []
    for i in range(20):
        c = 25 * (i % 20) + 10
        cmap.append(color_img[20, c ])

    for group_idx, group in enumerate(mask):
        segmented_mesh.visual.face_colors[group] = cmap[group_idx % 20]

    boundary_edges, boundary_pts = apply_mask(segmented_mesh, mask)
    boundary_curves, bifurcating_vertex_ids = trace_boundary_curves(boundary_edges)

    # write_obj_file("bifurcating_vertex_ids.obj", segmented_mesh.vertices[bifurcating_vertex_ids])
    # for i, bcurve in enumerate(boundary_curves):
    #     write_obj_file(f"boundary_curve_{i}.obj", segmented_mesh.vertices[bcurve])

    smoothed_mesh, viz_list = refinement(
        segmented_mesh, boundary_curves, 
        args.iters, args.boundary_resample, args.edge_flip, args.laplacian
    )
    segmented_mesh.export(f'segmented_mesh_final.ply')
    # export_path = str(outdir / f'segmented_mesh_smoothed_{args.iters}.ply')
    # if args.edge_flip:
    #     export_path = export_path.replace('.ply', '_edge_flip.ply')
    # if args.laplacian:
    #     export_path = export_path.replace('.ply', '_laplician.ply')
    # smoothed_mesh.export(str(export_path))
    # logger.info(f'Smoothed mesh exported to {export_path}')

    # if len(viz_list) > 0:
    #     if len(viz_list) == 1:
    #         viz_mesh = viz_list[0]
    #     else:
    #         viz_mesh = trimesh.util.concatenate(viz_list)
    #     viz_mesh_path = export_path.replace('.ply', '_modification.ply')
    #     viz_mesh.export(str(viz_mesh_path))
    #     logger.info(f'Visualization mesh exported to {viz_mesh_path}')
