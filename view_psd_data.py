import matplotlib
from matplotlib import cm
import numpy as np
import trimesh
import os
from glob import glob
import json
import argparse

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        

def read_psd_shape(file):
    file = glob(file)[0]
    mesh = trimesh.load(file, process=False, maintain_order=True)
    return mesh

def read_psd_segmask(file):
    file = glob(file)[0]
    with open(file, 'r') as f:
        lines = f.readlines()
    mask = []
    for i, line in enumerate(lines):
        if i % 2 == 1:
            mask.append([int(x)-1 for x in line.strip().split()])
    return mask

def visualize_psd_shape(shape_file, segmask_file, textured=False):
    mesh = read_psd_shape(shape_file)
    mask = read_psd_segmask(segmask_file)

    if textured:
        norm = matplotlib.colors.Normalize(0, 20, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)
        for cid, cmask in enumerate(mask):
            # print(cid, len(cmask))
            color = mapper.to_rgba(cid%20)
            color = np.array(color)*255
            color = color.astype(np.uint8)
            # color = trimesh.visual.random_color()
            # print(color)
            mesh.visual.face_colors[cmask] = color
    return mesh, mask


if __name__ == "__main__":

    folder = "/home/lyang/yl_code/dataset/labeledDb/LabeledDB_new/*"
    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='167', help='Input mesh path.')
    args = parser.parse_args()
    

    ## load mesh
    shape_id = args.input
    fpath = f"./data/segmentation_data/*/{shape_id}.off"
    
    mesh, mask = visualize_psd_shape(fpath, fpath.replace(".off", "_labels.txt"), textured=True)

    if True:
        maskpath = f"./data/segmentation_data/segmentation_results/{shape_id}.seg"
        pred_mask = np.loadtxt(maskpath, dtype=int)
        print(pred_mask.shape)
        print(mesh.vertices.shape)

    # mask_path = f"{shape_id}_mask.json"
    # with open(mask_path, 'w') as f:
    #     json.dump(mask, f, cls=NpEncoder, ensure_ascii=False, indent=4)
    # mesh.export(f"{shape_id}.obj")
    mesh.show()