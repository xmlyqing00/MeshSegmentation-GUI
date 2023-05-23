import argparse
import os
import trimesh
import json
import numpy as np
import networkx as nx


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Find Components')
    parser.add_argument('--input', type=str, default='data/manohand_0.obj', help='Input mesh path.')
    parser.add_argument('--outdir', type=str, default='./output/find_components', help='Output path.')
    args = parser.parse_args()

    print(args)

    mesh = trimesh.load(args.input, process=False, maintain_order=True)
    
    graph = nx.from_edgelist(mesh.face_adjacency)
    mask = [list(x) for x in nx.connected_components(graph)]
    print('Number of components:', len(mask))

    obj_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.outdir, obj_name)
    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, 'mask.json')
    print('Save to', mask_path)

    with open(mask_path, 'w') as f:
        json.dump(mask, f, cls=NpEncoder, ensure_ascii=False, indent=4)
