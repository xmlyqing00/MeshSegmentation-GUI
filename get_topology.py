import argparse
import trimesh
import numpy as np
import json
from pathlib import Path
from mesh_data_structure.build_complex import normalize_data, ComplexBuilder
from utils.vis_utils import build_spheres, build_cylinders


def vis_graph(graph: dict):

    cells = graph['cells']
    node_pts = np.array(graph['nodes'])
    mesh_list = []
    for cell in cells:
        new_cell = cell.copy()
        new_cell.append(cell[0])    
        cell_pts = node_pts[new_cell]

        cylinders = build_cylinders(cell_pts[:-1], cell_pts[1:], 0.02, (0, 255, 0))
        spheres = build_spheres(cell_pts[:-1], 0.03, (255, 0, 0))
        mesh_list.append(cylinders)
        mesh_list.append(spheres)
    
    mesh = trimesh.util.concatenate(mesh_list)
    return mesh


def subdivide(graph: dict):

    cells = graph['cells']
    node_pts = np.array(graph['nodes'])
    new_pt_mesh_list = []
    new_edge_mesh_list = []
    
    new_node_pts = graph['nodes']
    new_cells = []

    for cell in cells:
        center = np.mean(node_pts[cell], axis=0)
        node_num = len(cell)
        edge_pt_ids = cell.copy()
        edge_pt_ids.append(cell[0])
        edge_pts = node_pts[edge_pt_ids]
        edge_centers = (edge_pts[:-1] + edge_pts[1:]) / 2
        
        new_pt_mesh_list.append(build_spheres(center, 0.03, (0, 0, 255)))
        new_pt_mesh_list.append(build_spheres(edge_centers, 0.03, (0, 0, 255)))

        new_edge_mesh_list.append(
            build_cylinders(center[None, ...].repeat(node_num, axis=0), edge_centers, 0.02, (0, 255, 255)))

        center_id = len(new_node_pts)
        edge_center_ids = list(range(center_id + 1, center_id + len(cell) + 1))
        new_node_pts.append(center.tolist())
        new_node_pts.extend(edge_centers.tolist())

        for i in range(node_num):
            new_cell = [center_id, edge_center_ids[(i + node_num - 1) % node_num], cell[i], edge_center_ids[i],]
            new_cells.append(new_cell)

    new_pt_mesh = trimesh.util.concatenate(new_pt_mesh_list)
    new_edge_mesh = trimesh.util.concatenate(new_edge_mesh_list)

    return new_pt_mesh, new_edge_mesh, new_node_pts, new_cells


def write_quad_mesh(node_pts, cells, output_path):

    with open(output_path, 'w') as f:
        for pt in node_pts:
            f.write(f'v {pt[0]} {pt[1]} {pt[2]}\n')
        for cell in cells:
            f.write(f'f {cell[0] + 1} {cell[1] + 1} {cell[2] + 1} {cell[3] + 1}\n')
    
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get topology of a mesh')
    parser.add_argument('--input', type=str, help='Input')

    args = parser.parse_args()

    topology_path = Path(args.input)
    out_dir = Path('output') / 'ckpt_example' / topology_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, 'r') as f:
        graph = json.load(f)
    
    graph_mesh = vis_graph(graph)
    graph_mesh.export(str(out_dir / 'graph_raw.ply'))

    new_pt_mesh, new_edge_mesh, new_node_pts, new_cells = subdivide(graph)
    new_pt_mesh.export(str(out_dir / 'subdivide_pts.ply'))
    new_edge_mesh.export(str(out_dir / 'subdivide_edges.ply'))

    union_vis_mesh = trimesh.util.concatenate([graph_mesh, new_pt_mesh, new_edge_mesh])
    union_vis_mesh.export(str(out_dir / 'union_vis.ply'))

    write_quad_mesh(new_node_pts, new_cells, out_dir / 'subdivide_quad.obj')
    

    # pcd = trimesh.PointCloud(node_pts)
    # pcd.export(str(input_dir / 'node_pts.ply'))