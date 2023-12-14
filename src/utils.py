import json
import numpy as np
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