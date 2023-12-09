import argparse
import trimesh
import os
import json
import subprocess
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='data/manohand_0.obj', help='Input mesh path.')
    parser.add_argument('--mask', type=str, default=None, help='Input mask path.')
    parser.add_argument('--outdir', type=str, default='./output', help='Output directory.')
    args = parser.parse_args()

    print(args)

    tri_mesh = trimesh.load(args.input, maintain_order=True, process=False)
    obj_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.outdir, obj_name, 'sub_mesh')
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
        
        sub_mesh_path = os.path.join(output_dir, f'mesh_{sub_idx:03d}.off')
        sub_mesh.export(sub_mesh_path)

    # while not os.path.exists(sub_mesh_path):
    #     time.sleep(1)

    for sub_idx, sub_mesh in enumerate(sub_mesh_list):
        
        sub_mesh_path = os.path.join(output_dir, f'mesh_{sub_idx:03d}.off')
        program_name = '/mnt/e/Sources/NeuralImplicitBases/CGAL-5.6/examples/Surface_mesh_parameterization/build/discrete_conformal'
        input_name = sub_mesh_path
        output_name = sub_mesh_path.replace('mesh_', 'mesh_2d_')
        subprocess.run([program_name, input_name, output_name])