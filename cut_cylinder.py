import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import trimesh
import argparse
import numpy as np
import json
import bisect
from scipy.spatial.distance import cdist
from src.utils import NpEncoder, create_lines, create_spheres
from igl_parameterization import compute_harmonic_scalar_field
from PIL import Image


def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, cls=NpEncoder)


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


def count_foldover_triangles(mesh, threshold):
    face_pairs = mesh.face_adjacency
    pairs = mesh.face_normals[face_pairs]
    cos_sim = np.sum(pairs[:, 0, :]*pairs[:, 1, :], axis=-1)
    ## 170 degree = 2.96705972839 rad = -0.98
    # face_pair_mask = cos_sim < -0.80 ## edge
    # flipped_faces = np.unique(face_pairs[face_pair_mask])
    face_pair_mask = cos_sim < np.cos(np.deg2rad(threshold))
    flipped_faces = np.unique(face_pairs[face_pair_mask])
    return flipped_faces

## can draw line segments
def write_line_file2(save_to, V, L, C=None, vid_start=1):
    with open(save_to, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        # f.write('s off\n')
        for Li in L:
            line = "l "
            for i in Li:
                line = f"{line}{i+vid_start} "
            f.write(line+'\n')


def trace_path_by_samples(mesh, uv, corner_id, sample_num=20):

    point_cur = {
        'x': mesh.vertices[corner_id],
        'u': uv[corner_id, 0]
    }
    cut_path = [point_cur['x']]
    cut_path_info = [point_cur]

    faces = mesh.vertex_faces[corner_id]

    while True:
        
        mask = faces != -1
        unique_faces = faces[mask]
        related_edges = mesh.faces_unique_edges[unique_faces]
        related_edges = np.unique(related_edges)
        
        sample_pts = []
        for eid in related_edges:

            e = mesh.edges_unique[eid]
            samples = np.linspace(0, 1, sample_num)
            for t in samples:
                sample_x = mesh.vertices[e[0]] * (1-t) + mesh.vertices[e[1]] * t
                sample_u = uv[e[0], 0] * (1-t) + uv[e[1], 0] * t
                
                d_x_norm = np.linalg.norm(sample_x - point_cur['x'])
                d_u_norm = sample_u - point_cur['u']

                if abs(d_x_norm) < 1e-7:
                    derivative = 0
                    continue
                else:
                    derivative = d_u_norm / d_x_norm

                sample_pts.append({
                    'x': sample_x,
                    'u': sample_u,
                    'derivative': derivative,
                    't': t,
                    'eid': eid
                })

        sample_pts = sorted(sample_pts, key=lambda x: x['derivative'], reverse=True)
        point_next = sample_pts[0]
        cut_path.append(point_next['x'])
        cut_path_info.append(point_next)

        if abs(point_next['u'] - 1) < 1e-7:
            break
        else:
            point_cur = point_next
            e = mesh.edges_unique[point_next['eid']]
            # adj_index = mesh.face_adjacency_edges_tree.query([e])[1]
            # faces = mesh.face_adjacency[adj_index]
            faces = mesh.vertex_faces[e].flatten()

    return np.array(cut_path), cut_path_info


def preprocess(mesh, texture_img):
    
    mesh_size = np.array([
        mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min(),
    ])
    vis_size = 1e-2 * mesh_size.max()
    
    uv, boundary_list = compute_harmonic_scalar_field(mesh)
    uv[:, 1] = 0
    uv_visuals = trimesh.visual.texture.TextureVisuals(
        uv=uv, 
        image=texture_img
    )
    mesh.visual = uv_visuals
    mesh.export(str(out_dir / exp_name.name))
    
    return uv, vis_size, boundary_list


def main_cut_two_paths(mesh, out_dir, texture_img, exp_name):

    uv, vis_size, boundary_list = preprocess(mesh, texture_img)
    sample_num = 20

    bidx = 0
    vids = np.array(boundary_list[bidx])
    v = mesh.vertices[vids]
    pair_dist = cdist(v, v)
    a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)

    corner_ids = [vids[a0], vids[a1]]
    for corner_id in corner_ids:

        cut_path, cut_path_info = trace_path_by_samples(mesh, uv, corner_id, sample_num=sample_num)

        vis_spheres = create_spheres(cut_path, radius=vis_size, color=(0,200,0))
        vis_lines = []
        for p in range(1, len(cut_path)):
            lines = create_lines(cut_path[p-1], cut_path[p], radius=vis_size, color=(0,0,200))
            vis_lines.append(lines)
        vis_lines = trimesh.util.concatenate(vis_lines)
        
        vis_spheres.export(str(out_dir / f'{exp_name.stem}_spheres_{corner_id}.obj'))
        vis_lines.export(str(out_dir / f'{exp_name.stem}_lines_{corner_id}.obj'))


def main_cut_one_path(mesh, out_dir, texture_img, exp_name):

    uv, vis_size, boundary_list = preprocess(mesh, texture_img)
    bidx = 0
    vids = np.array(boundary_list[bidx])
    v = mesh.vertices[vids]
    pair_dist = cdist(v, v)
    a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)

    corner_ids = [vids[a0], vids[a1]]

    cut_path, cut_path_info = trace_path_by_samples(mesh, uv, corner_ids[0], sample_num=20)
    cut_path_u, cut_path_info_u = trace_path_by_samples(mesh, uv, corner_ids[1], sample_num=20)
    cut_path2 = [mesh.vertices[vids[a1]]]
    # cut_path2 = []

    path2_idx = 1
    for i in range(1, len(cut_path)):
        
        if i == 0:
            faces = mesh.vertex_faces[corner_ids[1]].flatten()
        else:
            eid = cut_path_info[i]['eid']
            e = mesh.edges_unique[eid]
            faces = mesh.vertex_faces[e].flatten()
        # grad_deepest = cut_path[i] - cut_path[i-1]
        # grad_deepest = grad_deepest / np.linalg.norm(grad_deepest)

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
            related_edges = mesh.faces_unique_edges[unique_faces]
            related_edges = np.unique(related_edges)
            
            candidate_list = []
            for eid in related_edges:
                
                if eid in visited_eids:
                    continue

                e = mesh.edges_unique[eid]
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

                        sample_x = mesh.vertices[e[0]] * (1-r) + mesh.vertices[e[1]] * r
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
            # visited_eids.add(point_next['eid'])
            e = mesh.edges_unique[point_next['eid']]
            faces = mesh.vertex_faces[e].flatten()
        
        # round_path.append(round_path[0])
        round_path = np.array(round_path)
        round_center = round_path.mean(axis=0)
        opposite_point = round_center - cut_path[i]
        opposite_dir = opposite_point / np.linalg.norm(opposite_point)
        
        round_path_seg = None
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
        for j in range(p_cos_maxidx - 3, p_cos_maxidx + 3):
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
        # round_piecewise_lens = np.array([np.linalg.norm(round_path[i+1] - round_path[i]) for i in range(len(round_path)-1)])
        # round_piecewise_lens_cumsum = np.cumsum(round_piecewise_lens)
        # half_len = round_piecewise_lens_cumsum[-1] / 2
        # mid_point_idx = bisect.bisect_left(round_piecewise_lens_cumsum, half_len)
        # rest_len = round_piecewise_lens_cumsum[mid_point_idx] - half_len
        # r = rest_len / round_piecewise_lens[mid_point_idx]
        # mid_point = round_path[mid_point_idx] * (1-r) + round_path[mid_point_idx+1] * r
        cut_path2.append(mid_point)

        round_spheres = create_spheres(round_path, radius=vis_size, color=(0,200,0))
        starting_sphere = create_spheres(round_path[0], radius=vis_size * 1.5, color=(200,0,200))
        last_sphere = create_spheres(round_path[-2], radius=vis_size * 1.5, color=(0,200,200))
        mid_sphere = create_spheres(cut_path2[-1], radius=vis_size * 1.5, color=(200,0,0))  
        vis = [round_spheres, mid_sphere, starting_sphere, last_sphere]
        for p in range(1, len(round_path)):
            lines = create_lines(round_path[p-1], round_path[p], radius=vis_size / 2, color=(0,0,200))
            vis.append(lines)
        vis = trimesh.util.concatenate(vis)
        
        vis.export(str(out_dir / f'{exp_name.stem}_debug_round_{i}.obj'))

    cut_path_total = [cut_path, cut_path2]
    for i, cut_path in enumerate(cut_path_total):
        # print(i, cut_path)
        vis_spheres = create_spheres(cut_path, radius=vis_size, color=(0,200,0))
        vis_lines = [vis_spheres]
        for p in range(1, len(cut_path)):
            # print(p, cut_path[p-1], cut_path[p])
            lines = create_lines(cut_path[p-1], cut_path[p], radius=vis_size / 2, color=(0,0,200))
            vis_lines.append(lines)
        vis_lines = trimesh.util.concatenate(vis_lines)
        vis_lines.export(str(out_dir / f'{exp_name.stem}_cutpath_{i}.obj'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--mesh",
                        required=True,
                        type=str,
                        help="path to mesh"
                        )
    
    args = parser.parse_args()

    ## read data
    mesh = trimesh.load(args.mesh, process=False, maintain_order=True)
    texture_img = Image.open(f'./assets/uv_color.png')
    # texture_img = np.asarray(texture_img)

    ## root folder
    exp_name = Path(args.mesh)
    out_dir = Path('tmp') / exp_name.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print('out_dir', out_dir)

    ## parameterization
    main_cut_one_path(mesh, out_dir, texture_img, exp_name)
    
