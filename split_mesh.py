import argparse
import trimesh
import os
import json


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='data/manohand_0.obj', help='Input mesh path.')
    parser.add_argument('--mask', type=str, default=None, help='Input mask path.')
    parser.add_argument('--outdir', type=str, default='./output', help='Output directory.')
    args = parser.parse_args()

    print(args)

    tri_mesh = trimesh.load(args.input, maintain_order=True, process=False)
    obj_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.outdir, obj_name)
    os.makedirs(output_dir, exist_ok=True)

    # Try to load the mask
    mask = None
    if args.mask and os.path.exists(args.mask):
        with open(args.mask, 'r') as f:
            mask = json.load(f)

    # print(mask)

    sub_mesh_list = tri_mesh.submesh(mask)
    for sub_idx, sub_mesh in enumerate(sub_mesh_list):
        
        print(sub_mesh)
        
        sub_mesh_path = os.path.join(output_dir, f'sub_mesh_{sub_idx:03d}.off')
        sub_mesh.export(sub_mesh_path)
        