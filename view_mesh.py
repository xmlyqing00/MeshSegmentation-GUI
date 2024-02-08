import argparse
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm
from vedo import Plotter, Sphere, Text2D, Mesh, write, Line, utils
from src.utils import create_lines


def mesh_quality(mesh: trimesh.Trimesh, bin_num: int = 20):

    f_adj_list = mesh.face_adjacency
    fe_adj_list = np.sort(mesh.face_adjacency_edges, axis=1)

    r_list = []
    vis_bads = []
    for f_adj, fe_adj in tqdm(zip(f_adj_list, fe_adj_list)):

        pt_on_edge = fe_adj
        pt_all = np.unique(mesh.faces[f_adj].flatten())
        pt_off_edge = np.setdiff1d(pt_all, pt_on_edge, assume_unique=True)

        angle_on_edge_sum = 0
        for pt_id in pt_on_edge:
            pt = mesh.vertices[pt_id]
            neighbor_pts = mesh.vertices[pt_off_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_on_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))

        angle_off_edge_sum = 0
        for pt_id in pt_off_edge:
            pt = mesh.vertices[pt_id]
            neighbor_pts = mesh.vertices[pt_on_edge]
            neighbor_pts = neighbor_pts - pt
            neighbor_pts = neighbor_pts / np.linalg.norm(neighbor_pts, axis=1)[:, None]
            angle_off_edge_sum += np.arccos(neighbor_pts[0].dot(neighbor_pts[1]))
        
        r = angle_on_edge_sum / angle_off_edge_sum
        r_list.append(r)

        if r < 0.5:
            c = [0, 0, 200]
            if r < 0.42:
                c = [200, 0, 0]
            pts = mesh.vertices[pt_on_edge]
            bad_line = create_lines(pts[0], pts[1], 0.001, c)
            vis_bads.append(bad_line)

    r_list = np.array(r_list)
    bins = np.linspace(0, 2, bin_num)
    hist, bin_edges = np.histogram(r_list, bins=bins)
    ranges = [f'{bin_edges[i-1]:.3f} - {bin_edges[i]:.3f}' for i in range(1, len(bin_edges))]

    df = pd.DataFrame({'hist': hist}, index=ranges)

    vis_bads = trimesh.util.concatenate(vis_bads)
    vis_bads.export('bad_lines.obj')

    return df
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser('View Mesh')
    parser.add_argument('--mesh', type=str, help='Path to the mesh file')
    parser.add_argument('--fid', type=int, help='Face id')
    parser.add_argument('--vid', type=int, help='Vertex id')
    parser.add_argument('--highlight', action='store_true', help='Highlight the face')
    parser.add_argument('--quality', action='store_true', help='Show triangular quality of the mesh')

    args = parser.parse_args()
    print(args)

    if args.quality:
        print('Computing quality')
        mesh_tri = trimesh.load(args.mesh, process=False, maintain_order=True)
        df = mesh_quality(mesh_tri, bin_num=20)
        df_path = args.mesh.replace('.obj', '_quality.csv')
        df.to_csv(df_path, index=True)
        exit(0)

    plotter = Plotter(bg='white')

    mesh = Mesh(args.mesh)

    if args.fid is not None:
        mesh.cellcolors[args.fid] = [255, 0, 0, 255]
        if args.highlight:
            vs = mesh.vertices[mesh.cells[args.fid]]
            for v in vs:
                s = Sphere(v, r=0.001, c='r')
                plotter.add(s)
        # face = mesh.faces(args.fid)
    
    if args.vid is not None:
        s = Sphere(mesh.vertices[args.vid], r=0.001, c='g')
        plotter.add(s)

    
        

    plotter.add(mesh)
    plotter.show()
    plotter.close()
    