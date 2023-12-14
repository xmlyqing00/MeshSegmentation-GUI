import argparse
import trimesh
import igl
import numpy as np
from pathlib import Path
from loguru import logger
from src.utils import write_obj


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Texture Transfer')
    parser.add_argument('--tex-mesh', type=str, default='data/manohand_tex/manohand_tex.obj', help='Input texture mesh path.')
    parser.add_argument('--new-mesh', type=str, default='output/manohand_tex/single/manohand_tex.ply', help='Input new mesh path.')

    args = parser.parse_args()
    logger.info(f'Arguments: {args}')

    # Load texture mesh
    assert Path(args.tex_mesh).suffix == '.obj'
    v, tc, n, f, ftc, fn = igl.read_obj(args.tex_mesh)
    logger.debug(f'Tex Mesh loaded by IGL: {v.shape}, {tc.shape}, {n.shape}, {f.shape}, {ftc.shape}, {fn.shape}')
    tex_mesh = {
        'v': v,
        'tc': tc,
        'n': n,
        'f': f,
        'ftc': ftc,
        'fn': fn
    }
    tex_mesh_trimesh = trimesh.load(
        args.tex_mesh, 
        process=False, maintain_order=True,
    )
    logger.debug(f'Tex Mesh loaded by Trimesh: {tex_mesh_trimesh.vertices.shape}, {tex_mesh_trimesh.faces.shape}')
    assert v.shape == tex_mesh_trimesh.vertices.shape
    assert f.shape == tex_mesh_trimesh.faces.shape

    # Load new mesh
    new_mesh = trimesh.load(args.new_mesh, proccess=False, maintain_order=True)
    logger.debug(f'New mesh: {new_mesh.vertices.shape}, {new_mesh.faces.shape}')
    
    # Find positions of new mesh vertices in texture space
    sqr_d, f_id, closest_v = igl.point_mesh_squared_distance(new_mesh.triangles_center, tex_mesh['v'], tex_mesh['f'])
    vs = tex_mesh['v'][tex_mesh['f'][f_id]]
    a, b, c = vs[:, 0], vs[:, 1], vs[:, 2]
    a = np.repeat(a, 3, axis=0)
    b = np.repeat(b, 3, axis=0)
    c = np.repeat(c, 3, axis=0)

    # Compute barycentric coordinates
    vspf = new_mesh.vertices[new_mesh.faces].reshape(-1, 3)
    bary_coords = igl.barycentric_coordinates_tri(
        vspf,
        np.ascontiguousarray(a),
        np.ascontiguousarray(b),
        np.ascontiguousarray(c)
    )

    # Compute new texture coordinates
    tc = tex_mesh['tc'][tex_mesh['ftc'][f_id]]
    tc = np.repeat(tc, 3, axis=0)
    v_tex = np.einsum('ijk, ij -> ik', tc, bary_coords)
    t_tex_idx = np.arange(len(v_tex)).reshape(-1, 3)

    # Compute new normals
    v_nrm = None
    t_nrm_idx = None
    if tex_mesh['n'] is not None and tex_mesh['fn'] is not None:
        n = tex_mesh['n'][tex_mesh['fn'][f_id]]
        n = np.repeat(n, 3, axis=0)
        v_nrm = np.einsum('ijk, ij -> ik', n, bary_coords)
        t_nrm_idx = np.arange(len(v_nrm)).reshape(-1, 3)

    # Create new mesh
    new_mesh_wtex_path = args.new_mesh.replace('.ply', '_with_tex.obj') 
    write_obj(
        new_mesh_wtex_path, 
        new_mesh.vertices, 
        new_mesh.faces, 
        v_nrm=v_nrm,
        v_tex=v_tex,
        t_nrm_idx=t_nrm_idx,
        t_tex_idx=t_tex_idx,
        material_img=tex_mesh_trimesh.visual.material.image,
        save_material=True
    )
