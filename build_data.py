import os
import sys
sys.path.append(os.getcwd())

import shutil
import trimesh
import argparse
import numpy as np
import json
import igl
from src.utils import NpEncoder
from igl_parameterization import parameterize_mesh, parameterize_mesh_arap_harmonic, parameterize_mesh_with_boundary_len
from mesh_data_structure.build_complex import normalize_data, ComplexBuilder
from mesh_data_structure.get_boundary_length_from_mask import get_boundary_length_from_mask
from PIL import Image
import matplotlib
from matplotlib import cm
from pathlib import Path


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


## can draw line segments
def write_line_file2(save_to, V, L, C=None, vid_start=1):
    with open(save_to, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        # f.write('s off\n')
        for Li in L:
            line = "l "
            for i in Li:
                line = f"{line}{i+vid_start} "
            f.write(line+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--datadir",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    parser.add_argument('--use-smoothed-mesh', action='store_true', help='use smoothed mesh or not')
    parser.add_argument('--method', type=str, default='harmonic', help='parameterization method [harmonic, BPE_harmonic]')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    ## read data
    data_dir = Path(args.datadir)
    maskfile = data_dir / 'mask.json'
    if args.use_smoothed_mesh:
        meshfile = data_dir /'segmented_mesh_smoothed.obj'
        if meshfile.exists() is False:
            meshfile = data_dir /'segmented_mesh_smoothed.ply'
    else:
        meshfile = data_dir /'segmented_mesh.obj'
        if meshfile.exists() is False:
            meshfile = data_dir /'segmented_mesh.ply'
    mesh = trimesh.load(meshfile, process=False, maintain_order=True)
    mask = read_json(maskfile)

    texture_img = Image.open(f'./assets/uv_color.png')

    ## root folder
    root_dir = Path(f'./data_built_{args.method}') / data_dir.name
    root_dir.mkdir(parents=True, exist_ok=True)

    savefolder = root_dir / 'data'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    ## build complex from the base mesh and its patches (dict)
    complex_builder = ComplexBuilder(mesh, mask)
    graph = complex_builder.build_complex_recursive()
    # complex_builder.save_complex(graph, root_dir)
    
    # print(graph)
    write_json(graph, savefolder / "topology_graph.json")
    shutil.copyfile(maskfile, savefolder / 'mask.json')

    ## cell arc lengths
    cell_arc_lengths = get_boundary_length_from_mask(mesh, mask, graph)
    
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

    assert args.method in ['harmonic', 'BPE_harmonic'], "Only support harmonic and BPE_harmonic parameterization"
    if args.method == 'BPE_harmonic':

        BPE_resdir = data_dir / 'BPE'
        BPE_resdir.mkdir(parents=True, exist_ok=True)

        if os.sys.platform == 'linux':
            exe_path = '/mnt/e/Sources/bpe/x64/Release/BPE.exe'
        else:
            exe_path = 'E:/Sources/bpe/x64/Release/BPE.exe'
                
    cell_arc_lengths = []
    for i in range(len(mask)):

        nodes = node_ids[cells[i]]
        corners = mesh.vertices[nodes]
        submesh = mesh.submesh([mask[i]], only_watertight=False)[0]
        
        ## build proximity mesh
        pq_patch = trimesh.proximity.ProximityQuery(submesh)
        crn_ids = pq_patch.vertex(corners)[1].tolist()
        corners = submesh.vertices[crn_ids]
        # print("corners: ", corners)

        ## parameterization
        if args.method == 'BPE_harmonic':

            submesh_path = BPE_resdir / f'mesh_{i}.ply'
            print('submesh_path', submesh_path)
            submesh.export(submesh_path)

            cmd_str = f'{exe_path} {submesh_path}'
            print('cmd_str', cmd_str)
            os.system(cmd_str)

            submesh_para_path = BPE_resdir / f'mesh_{i}_result.obj'
            print('submesh_para_path', submesh_para_path, submesh_para_path.exists())
            submesh_para = trimesh.load(submesh_para_path, process=False, maintain_order=True)

            bnd = igl.boundary_loop(submesh.faces)
            bnd_list = bnd.tolist()
            list_boundary = []
            # new_list_bnd = []
            for crn_idx in range(len(crn_ids)):
                bid0 = bnd_list.index(crn_ids[crn_idx])
                bid1 = bnd_list.index(crn_ids[(crn_idx+1)%len(crn_ids)])
                
                if bid0 < bid1:
                    list_boundary.append(bnd_list[bid0:bid1+1])
                    # new_list_bnd += bnd_list[bid0:bid1]
                else:
                    list_boundary.append(bnd_list[bid0:] + bnd_list[:bid1+1])
                    # new_list_bnd += bnd_list[bid0:] + bnd_list[:bid1]

            ## compute the length of each boundary
            list_boundary_length = []
            for bnd in list_boundary:
                bnd_length = []
                ## open curve
                for bnd_idx in range(len(bnd)-1):
                    bnd_length.append(np.linalg.norm(submesh.vertices[bnd[bnd_idx]] - submesh.vertices[bnd[bnd_idx+1]]))
                list_boundary_length.append(bnd_length)
            
            list_boundary_length_old = list_boundary_length.copy()
            uv, bnd_uv, crn_uv, list_boundary_length, bnd_list = parameterize_mesh_with_boundary_len(submesh_para.vertices, submesh_para.faces, crn_ids, list_boundary_length)
            if np.isnan(bnd_uv[0][0]):
                print('NAN detected, re-parameterize')
                # uv, bnd_uv, crn_uv, list_boundary_length, bnd_list = parameterize_mesh_arap_harmonic(submesh.vertices, submesh.faces, crn_ids, list_boundary_length_old)
                uv, bnd_uv, crn_uv, list_boundary_length, bnd_list = parameterize_mesh(submesh.vertices, submesh.faces, crn_ids)
        
        else:
            uv, bnd_uv, crn_uv,  list_boundary_length, bnd_list = parameterize_mesh(submesh.vertices, submesh.faces, crn_ids)

        submesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=None)
        crn_uv = np.concatenate((crn_uv, np.zeros((crn_uv.shape[0], 1))), axis=1)
        bnd_uv = np.concatenate((bnd_uv, np.zeros((bnd_uv.shape[0], 1))), axis=1)
        submesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)

        submesh.export(f'{savepatchfolder}/mesh_uv_{i}.obj')
        cell_arc_lengths.append(list_boundary_length)

        crn_points = np.stack([corners, crn_uv[:-1]], axis=1)
        crn_points = crn_points.reshape(-1,3)
        corner_links = np.stack([np.arange(0, len(crn_points), 2), np.arange(1, len(crn_points), 2)], axis=-1)

        ## save parameterized mesh
        uv3d = np.concatenate((uv, np.zeros((uv.shape[0], 1))), axis=1)
        flat = trimesh.Trimesh(vertices=uv3d, faces=submesh.faces, process=False, maintain_order=True)
        flat.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_img)
        flat.export(f'{saveflatfolder}/flat_{i}.obj')


        norm = matplotlib.colors.Normalize(0, 1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        colors = np.linspace(0,1,len(bnd_list))
        colors = mapper.to_rgba(colors)[:,:3]
        # points = np.stack([submesh.vertices[bnd_list], uv3d[bnd_list]], axis=1)
        points = np.stack([submesh.vertices[bnd_list], bnd_uv], axis=1)
        colors = np.stack([colors, colors], axis=1)
        points = points.reshape(-1,3)
        colors = colors.reshape(-1,3)
        boundary_links = np.stack([np.arange(0, len(points), 2), np.arange(1, len(points), 2)], axis=-1)
        # write_line_file2(f"boundary_links_{i}.obj", points, boundary_links, colors)

        # break


    for cl in cell_arc_lengths:
        print(cl)
    write_json(cell_arc_lengths, os.path.join(savefolder, 'cell_arc_lengths.json'))

    