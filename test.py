import shutil
import os
import trimesh
from src.io_tools import read_json
from vedo import Mesh
import matplotlib
import matplotlib.cm as cm
import numpy as np
from loguru import logger


# cubewithholev3_20240324_2158
# bunny_20240324_2159
# ts_spoon_20240324_2156
# ts_heelshoe_v3_20240324_2145 ## nan?
# tooth_upper_26_20240324_2142
# watering_pail2_20240324_2157 ## something wrong with the tracing
# chair2_20240324_2154
# cat_20240324_2205
# catcher_20240324_2150  ## something wrong with the tracing
# sculpt_rem_20240324_2153

shape_list = [
    # 'ts_heelshoe_v3_20240324_2145', ## nan
    # 'watering_pail2_20240324_2157',
    # 'catcher_20240324_2150'
    'chair2_20240324_2154',
    'cubewithholev3_20240324_2158',
    ]

source_folder = "data/old_data"
# for shape_folder in os.listdir(source_folder):
for shape_folder in shape_list:
    print(shape_folder)
    try:
        ## smoothing
        os.system(f'python smoothing_boundary.py --outdir output/old_data/{shape_folder} --iters 5 --boundary-resample --edge-flip --laplacian')
        # os.system(f'python smoothing_boundary.py --outdir output/old_data/{shape_folder} --iters 5 --boundary-resample')
        ## build from smoothed data
        os.system(f'python build_data.py --datadir output/old_data/{shape_folder} --use-smoothed-mesh')
        # os.system(f'python build_data.py --datadir output/old_data/{shape_folder}')
    except:
        logger.info(f'Error in {shape_folder}')
        continue

    print(f"Done with {shape_folder}\n\n\n")


# target_folder = "output/old_data"
# if not os.path.exists(target_folder):
#     os.makedirs(target_folder)

# for shape_folder in os.listdir(source_folder):
#     print(shape_folder)
#     target_shape_folder = os.path.join(target_folder, shape_folder)
#     if os.path.exists(target_shape_folder):
#         shutil.rmtree(target_shape_folder)
#     os.makedirs(target_shape_folder)

#     ## mask
#     shutil.copy(
#         os.path.join(source_folder, shape_folder, "data/mask.json"), 
#         os.path.join(target_shape_folder, "mask.json"))
#     ## mesh
#     mask = read_json(os.path.join(target_shape_folder, "mask.json"))
#     mesh = trimesh.load(
#         os.path.join(source_folder, shape_folder, "data/single/mesh.obj"), 
#         process=False, maintain_order=True)
#     vedo_mesh = Mesh([mesh.vertices, mesh.faces])

#     ## colormap
#     cmap = matplotlib.colormaps['tab20']
#     norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
#     mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
#     for i, m in enumerate(mask):
#         float_color = mapper.to_rgba(i%20)
#         vedo_mesh.cellcolors[m] = np.array(float_color)*255
#     vedo_mesh.write(os.path.join(target_shape_folder, "segmented_mesh.ply"))