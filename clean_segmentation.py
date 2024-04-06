import trimesh
import argparse
import shutil
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.utils import NpEncoder



def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, cls=NpEncoder)


def check_duplicate(f):
    for i in range(3):
        for j in range(i + 1, 3):
            if f[i] == f[j]:
                print('Found a face with duplicated vertices.', f)
                return True
    return False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Clean the mesh segmentation')
    parser.add_argument('--datadir', required=True, type=str, help='path to config')
    
    args = parser.parse_args()

    ## read data
    data_dir = Path(args.datadir)
    maskfile = data_dir / 'mask.json'
    
    meshfile = data_dir /'segmented_mesh.obj'
    if meshfile.exists() is False:
        meshfile = data_dir /'segmented_mesh.ply'

    mesh = trimesh.load(meshfile, process=False, maintain_order=True)
    mask = read_json(maskfile)

    ## root folder
    out_dir = data_dir

    # save old file
    shutil.copyfile(maskfile, out_dir / 'mask_raw.json')
    shutil.copyfile(meshfile, out_dir / meshfile.name.replace('.', '_raw.'))

    print('Copied the raw files.')

    fs = mesh.faces
    vs = mesh.vertices
    flabels = np.ones(len(fs), dtype=np.int32) * -1
    for i, seg in enumerate(mask):
        for fid in seg:
            flabels[fid] = i
    
    check_mask = (flabels == -1).nonzero()[0]
    
    assert len(check_mask) == 0, 'Found faces not in any segment.'

    fvalids = np.ones(len(fs), dtype=bool)

    mesh_raw = mesh.copy()
    mesh.merge_vertices()

    # for fid, f in enumerate(tqdm(mesh.faces)):
    #     res = check_duplicate(f)
    #     if res:
    #         fvalids[fid] = False

    fvalids = mesh.nondegenerate_faces()
    flabels = flabels[fvalids]
    mesh.update_faces(fvalids)

    print(mesh_raw.vertices.shape, mesh_raw.faces.shape)
    print(mesh.vertices.shape, mesh.faces.shape)
    
    new_mask = {}
    for i in range(len(mask)):
        new_mask[i] = []
    
    for vid, label in enumerate(tqdm(flabels)):
        new_mask[label].append(vid)

    new_mask_list = []
    for key, val in new_mask.items():
        new_mask_list.append(val)

    x = 4533
    fs = mesh.vertex_faces[x]
    print(x, fs)
    for fid in fs:
        if fid == -1:
            break
        print(fid, mesh.faces[fid])

    x = 3112
    fs = mesh.vertex_faces[x]
    print(x, fs)
    for fid in fs:
        if fid == -1:
            break
        print(fid, mesh.faces[fid])

    print(mesh.vertex_neighbors[x])
    
    write_json(new_mask_list, out_dir / 'mask_cleaned.json')
    mesh.export(str(meshfile).replace('mesh', 'mesh_cleaned'))

    print('Done.')

    # mesh.remove_unreferenced_vertices()
    # print(mesh.vertices.shape, mesh.faces.shape)
    

        


    # ## build complex from the base mesh and its patches (dict)
    # complex_builder = ComplexBuilder(mesh, mask)
    # graph = complex_builder.build_complex_recursive()
    # # complex_builder.save_complex(graph, root_dir)
    
    # # print(graph)
    # write_json(graph, savefolder / 'topology_graph.json')
    # shutil.copyfile(maskfile, savefolder / 'mask.json')