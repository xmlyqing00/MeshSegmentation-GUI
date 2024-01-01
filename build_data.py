import os
import sys
sys.path.append(os.getcwd())

import shutil
import trimesh
import argparse
import numpy as np
import json
from src.utils import NpEncoder
from igl_parameterization import parameterize_mesh
from mesh_data_structure.build_complex import normalize_data, ComplexBuilder
from mesh_data_structure.get_boundary_length_from_mask import get_boundary_length_from_mask

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


def count_foldover_triangles(mesh, threshold):
    face_pairs = mesh.face_adjacency
    pairs = mesh.face_normals[face_pairs]
    cos_sim = np.sum(pairs[:, 0, :]*pairs[:, 1, :], axis=-1)
    ## 170 degree = 2.96705972839 rad = -0.98
    # face_pair_mask = cos_sim < -0.80 ## edge
    # flipped_faces = np.unique(face_pairs[face_pair_mask])
    face_pair_mask = cos_sim < np.cos(np.deg2rad(threshold))
    flipped_faces = np.unique(face_pairs[face_pair_mask])
    return flipped_faces


def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    ## read data
    meshfile = f'./output/{args.model_name}/segmented_mesh.ply'
    maskfile = f'./output/{args.model_name}/mask.json'
    mesh = trimesh.load(meshfile, process=False, maintain_order=True)
    mask = read_json(maskfile)
    

    ## root folder
    root_dir = f'./data_built/{args.model_name}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    savefolder = f'{root_dir}/data'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    ## build complex from the base mesh and its patches (dict)
    complex_builder = ComplexBuilder(mesh, mask)
    graph = complex_builder.build_complex_recursive()
    # complex_builder.save_complex(graph, root_dir)
    print(graph)
    write_json(graph, os.path.join(savefolder, "topology_graph.json"))
    shutil.copyfile(maskfile, os.path.join(savefolder, 'mask.json'))
    ## cell arc lengths
    cell_arc_lengths = get_boundary_length_from_mask(mesh, mask, graph)
    write_json(cell_arc_lengths, os.path.join(savefolder, 'cell_arc_lengths.json'))
    

    ## save normalized mesh
    savemeshfolder = f'{savefolder}/single'
    if not os.path.exists(savemeshfolder):
        os.makedirs(savemeshfolder)
    save_meshfile = os.path.join(savemeshfolder, 'mesh.obj')
    mesh = normalize_data(mesh)
    mesh.export(save_meshfile)


    ## parameterization
    savepatchfolder = f'{root_dir}/parameterization'
    if not os.path.exists(savepatchfolder):
        os.makedirs(savepatchfolder)
    saveflatfolder = f'{root_dir}/flat_parameterization'
    if not os.path.exists(saveflatfolder):
        os.makedirs(saveflatfolder)
    node_ids = np.array(graph['node_ids'], dtype=np.int32)
    cells = graph['cells']
    for i in range(len(mask)):
        print(i, " -- face size in this mask: ", len(mask[i]))

        # print("cell nodes: ", cells[i])
        nodes = node_ids[cells[i]]
        # print("node ids in original mesh: ", nodes)
        corners = mesh.vertices[nodes]
        submesh = mesh.submesh([mask[i]], only_watertight=False)[0]
        # print("submesh.vertices", len(submesh.vertices))
        # print("submesh.faces", len(submesh.faces))

        ## build proximity mesh
        pq_patch = trimesh.proximity.ProximityQuery(submesh)
        crn_ids = pq_patch.vertex(corners)[1].tolist()
        corners = submesh.vertices[crn_ids]
        # print("patchmesh.vertices", len(patchmesh.vertices))
        # print("patchmesh.faces", len(patchmesh.faces))
        flipped = count_foldover_triangles(submesh, 135)
        print("flipped faces: ", len(flipped))

        ## parameterization
        uv, bnd_uv = parameterize_mesh(submesh.vertices, submesh.faces, crn_ids)
        submesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=None)
        # write_obj_file(f"corner_{i}.obj", corners)
        submesh.export(f'{savepatchfolder}/mesh_uv_{i}.obj')


        ## save parameterized mesh
        uv3d = np.concatenate((uv, np.zeros((uv.shape[0], 1))), axis=1)
        flat = trimesh.Trimesh(vertices=uv3d, faces=submesh.faces, process=False, maintain_order=True)
        flipped = count_foldover_triangles(submesh, 135)
        print("flat flipped faces: ", len(flipped))
        flat.export(f'{saveflatfolder}/flat_{i}.obj')



    


