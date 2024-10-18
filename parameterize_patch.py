import argparse
import igl
import numpy as np
import os
import trimesh
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from PIL import Image


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

    boundary_length_ratio = np.array([0, 0.25, 0.25, 0.25, 0.25])
    print("boundary ratio", boundary_length_ratio)
    boundary_cumsum_ratio = np.cumsum(boundary_length_ratio)
    print("boundary_cumsum ratio", boundary_cumsum_ratio)
    
    ## compute coordinate of each point
    radius = boundary_cumsum_ratio * 2 * np.pi + np.pi / 4
    endpoints_uv = np.stack((np.cos(radius), np.sin(radius))).swapaxes(0,1)

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
        all_boundary_uv.append(boundary_uv[:-1])
    
    bnd_uv = np.concatenate(all_boundary_uv, axis=0)
    return bnd_uv, endpoints_uv, boundary_length_ratio, new_list_bnd


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameterize a mesh using BEP and hormonic mapping.')
    parser.add_argument('--input', type=str, help='Input mesh file')
    parser.add_argument('--output', type=str, default='output/patch_parameterization', help='Output mesh file')
    parser.add_argument('--corners', type=int, nargs='+', help='Corner file')

    args = parser.parse_args()

    if os.sys.platform == 'linux':
        raise NotImplementedError("BPE is not supported on Linux")
    else:
        exe_path = 'E:/Sources/bpe/x64/Release/BPE.exe'

    input_mesh = Path(args.input)
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    out_dir = Path(args.output) / f'{input_mesh.stem}_{current_time}'
    out_dir.mkdir(parents=True, exist_ok=True)
    copyfile(args.input, out_dir / input_mesh.name)

    v, f = igl.read_triangle_mesh(args.input)
    bnd = igl.boundary_loop(f)

    print('v', v.shape, 'f', f.shape, 'bnd', bnd.shape)
    
    # check corner ids
    
    if args.corners:
        for cid in args.corners:
            assert cid in bnd

        corner_pts = v[args.corners]
        corner_pcd = trimesh.PointCloud(corner_pts)
        corner_pcd.export(str(out_dir / 'corner_pts.ply'))

    
    if True:
        cmd_str = f'{exe_path} {str(out_dir / input_mesh.name)}'
        os.system(cmd_str)

        bpe_mesh_v, bpe_mesh_f = igl.read_triangle_mesh(str(out_dir / f'{input_mesh.stem}_result.obj'))
        print('bpe_mesh_v', bpe_mesh_v.shape, 'bpe_mesh_f', bpe_mesh_f.shape)
        bnd_bpe = igl.boundary_loop(bpe_mesh_f)
        bnd_uv, corner_uv, list_boundary_length, new_list_bnd = map_to_ngon(bpe_mesh_v, bnd_bpe.tolist(), args.corners)
        bnd_new = np.array(new_list_bnd, dtype=np.int32).reshape(-1,1)
        uv = igl.harmonic(bpe_mesh_v, bpe_mesh_f, bnd_new, bnd_uv, 1)
    else:
        bnd_uv, corner_uv, list_boundary_length, new_list_bnd = map_to_ngon(v, bnd.tolist(), args.corners)
        bnd_new = np.array(new_list_bnd, dtype=np.int32).reshape(-1,1)
        uv = igl.harmonic(v, f, bnd_new, bnd_uv, 1)

    # save
    texture_img = Image.open(f'./assets/uv_color.png')

    input_trimesh = trimesh.Trimesh(vertices=v, faces=f, process=False, maintain_convexity=False)
    input_trimesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)
    input_trimesh.export(str(out_dir / 'mesh_uv.obj'))

    uv3d = np.concatenate((uv, np.zeros((uv.shape[0], 1))), axis=1)
    flat = trimesh.Trimesh(vertices=uv3d, faces=input_trimesh.faces, process=False, maintain_order=True)
    flat.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)
    flat.export(str(out_dir / 'flat_uv.obj'))

