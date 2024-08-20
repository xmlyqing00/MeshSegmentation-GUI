import os
import sys
sys.path.append(os.getcwd())

from pathlib import Path
import trimesh
import argparse
import numpy as np
import json
from tsp_solver.greedy import solve_tsp
from mesh_data_structure.utils import compute_distance_matrix
from src.utils import NpEncoder, create_lines, create_spheres
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


def cut_through_holes(mesh, out_dir, exp_name):
    mesh_size = np.array([
        mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min(),
    ])
    vis_size = 1e-2 * mesh_size.max()

    paths, distance_matrix = compute_distance_matrix(mesh, b_close_holes=False)
    print(distance_matrix)
    # solve TSP
    tsp_path = solve_tsp(distance_matrix)

    throughhole_paths = []
    for k in range(len(tsp_path)):
        i = tsp_path[k]
        j = tsp_path[(k+1)%len(tsp_path)]
        keys = [f"{i},{j}", f"{j},{i}"]
        for key in keys:
            if key in paths:
                throughhole_paths.append(paths[key])
                break
    
    for i, cut_path in enumerate(throughhole_paths):
        # print(i, cut_path)
        # vis_spheres = create_spheres(cut_path, radius=vis_size, color=(0,200,0))
        vis_lines = []
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
    cut_through_holes(mesh, out_dir, exp_name)
    
