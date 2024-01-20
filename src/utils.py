import json
import numpy as np
from trimesh import primitives, util, transformations
from pathlib import Path
from loguru import logger


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()


def write_obj(
    mesh_path: str, 
    v_pos: np.array, 
    t_pos_idx: np.array, 
    v_nrm=None, 
    v_tex=None, 
    t_nrm_idx=None, 
    t_tex_idx=None, 
    material_img=None, 
    save_material=False
):

    mesh_path = Path(mesh_path)
    logger.info(f'Writing mesh: {mesh_path}')

    with open(mesh_path, 'w') as f:
        f.write('mtllib mat_texture.mtl\n')
        f.write('g default\n')

        logger.debug('    writing %d vertices' % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            logger.debug('    writing %d texcoords' % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], v[1]))

        if v_nrm is not None:
            logger.debug('    writing %d normals' % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write('s off\n')
        # f.write('g pMesh1\n')
        f.write('g mat_texture:Mesh\n')
        f.write('usemtl defaultMat\n')
        # f.write('usemtl defaultMat\n')

        # Write faces
        logger.debug('    writing %d faces' % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write('f ')
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write('\n')

    if save_material:
        mtl_file = mesh_path.parent / 'mat_texture.mtl'
        logger.info('Writing material: ', mtl_file)
        
        with open(mtl_file, 'w') as f:
            f.write('newmtl defaultMat\n')
            f.write('illum 4\n')
            f.write('Ka 0.00 0.00 0.00\n')
            f.write('Kd 0.50 0.50 0.50\n')
            f.write('Tf 1.00 1.00 1.00\n')
            f.write('Ni 1.00\n')
            f.write('map_Kd mat_texture.png\n')
            material_img.save(mesh_path.parent / 'mat_texture.png')

    logger.success('Exporting mesh finished.')


def create_lines(pt0: np.array, pt1: np.array, radius: float, color: np.array):
    
    if isinstance(pt0, list):
        pt0 = np.array(pt0)
    
    if isinstance(pt1, list):
        pt1 = np.array(pt1)
        
    if len(pt0.shape) > 1:
        meshes = []
        for p0, p1 in zip(pt0, pt1):
            meshes.append(create_lines(p0, p1, radius, color))
        return util.concatenate(meshes)

    h = np.linalg.norm(pt0 - pt1)
    stick = primitives.Cylinder(radius=radius, height=h, sections=4)
    stick.visual.vertex_colors = color

    normal = pt0 - pt1
    normal = normal / np.linalg.norm(normal)

    # default stick.direction is [0, 0, 1]
    rot_axis = np.cross(stick.direction, normal)
    if np.linalg.norm(rot_axis) < 1e-8:
        # if stick.direction and normal are parallel, pick another axis
        rot_axis = np.array([1, 0, 0])
    rot_angle = np.arccos(np.dot(stick.direction, normal))
    rot_mat = transformations.rotation_matrix(rot_angle, rot_axis, (0, 0, 0))
    trans_mat1 = transformations.translation_matrix((0, 0, h / 2))
    trans_mat2 = transformations.translation_matrix(pt1)
    transform_mat = trans_mat2 @ rot_mat @ trans_mat1
    stick.apply_transform(transform_mat)
    
    return stick


def create_spheres(pt: np.array, radius, color):
    
    if isinstance(pt, list):
        pt = np.array(pt)

    if len(pt.shape) > 1:
        meshes = []
        for p in pt:
            meshes.append(create_spheres(p, radius, color))
        return util.concatenate(meshes)

    sphere = primitives.Sphere(radius=radius, center=pt, subdivisions=1)
    sphere.visual.vertex_colors = color

    return sphere
