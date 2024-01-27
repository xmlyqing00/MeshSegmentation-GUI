import trimesh
import argparse
import json
import numpy as np
from random import shuffle
from pathlib import Path
from loguru import logger
from PIL import Image
from tqdm import tqdm
from src.utils import create_lines


def import_mesh_mask(outdir: Path):
    segmented_mesh_path = outdir / 'segmented_mesh.ply'
    segmented_mesh = trimesh.load(segmented_mesh_path, process=False, maintain_order=True)
    
    mask_path = outdir / 'mask.json'
    with open(mask_path, 'r') as f:
        mask = json.load(f)

    return segmented_mesh, mask


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


def edge_flip(segmented_mesh: trimesh.Trimesh, boundary_pts: set):
    
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
    edge_flip_viz = []
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

            pt0 = segmented_mesh.vertices[flip_edge['pt_on_edge'][0]]
            pt1 = segmented_mesh.vertices[flip_edge['pt_on_edge'][1]]
            edge_flip_viz.append(create_lines(pt0, pt1, 0.001, color=(200, 0, 0)))

    segmented_mesh.faces = faces

    return segmented_mesh, edge_flip_viz


def refinement(
        segmented_mesh: trimesh.Trimesh, boundary_pts: set, 
        iters: int = 0, edge_flip_flag: bool = False, laplician_flag=False,
    ):

    viz_list = []
    for iter_idx in range(iters):

        logger.info(f'Iteration {iter_idx + 1} / {iters}')
        
        if edge_flip_flag:
            segmented_mesh, edge_flip_viz = edge_flip(segmented_mesh, boundary_pts)
            viz_list.extend(edge_flip_viz)

        if laplician_flag:
            one_ring_pts = []
            for boundary_pt in tqdm(boundary_pts, desc='Check one ring points'):

                neighbor_pts = segmented_mesh.vertex_neighbors[boundary_pt]
                one_ring_pts.extend(neighbor_pts)

            one_ring_pts = np.unique(one_ring_pts)
            one_ring_pts = np.setdiff1d(one_ring_pts, list(boundary_pts), assume_unique=True)
            
            logger.info(f'One ring points number: {len(one_ring_pts)}')
            
            shuffle(one_ring_pts)
            for pt in tqdm(one_ring_pts, desc='Laplician smooth'):

                neighbor_pts = segmented_mesh.vertex_neighbors[pt]
                neighbor_verts = segmented_mesh.vertices[neighbor_pts]
                segmented_mesh.vertices[pt] = np.mean(neighbor_verts, axis=0)

    return segmented_mesh, viz_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--outdir', type=str, default='./output', help='Output directory.')
    parser.add_argument('--iters', type=int, default=1, help='Refinement iterations.')
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

    smoothed_mesh, viz_list = refinement(
        segmented_mesh, boundary_pts, 
        args.iters, args.edge_flip, args.laplacian
    )

    export_path = str(outdir / f'segmented_mesh_smoothed_{args.iters}.ply')
    if args.edge_flip:
        export_path = export_path.replace('.ply', '_edge_flip.ply')
    if args.laplacian:
        export_path = export_path.replace('.ply', '_laplician.ply')
    smoothed_mesh.export(str(export_path))
    logger.info(f'Smoothed mesh exported to {export_path}')

    if len(viz_list) > 0:
        if len(viz_list) == 1:
            viz_mesh = viz_list[0]
        else:
            viz_mesh = trimesh.util.concatenate(viz_list)
        viz_mesh_path = export_path.replace('.ply', '_modification.ply')
        viz_mesh.export(str(viz_mesh_path))
        logger.info(f'Visualization mesh exported to {viz_mesh_path}')
