import argparse
import trimesh
import numpy as np
import json
from pathlib import Path
from mesh_data_structure.build_complex import normalize_data, ComplexBuilder
from utils.vis_utils import build_spheres, build_cylinders


def vis_graph(cells: list, node_pts: np.array):

    mesh_list = []
    for cell in cells:
        new_cell = cell.copy()
        new_cell.append(cell[0])    
        cell_pts = node_pts[new_cell]

        cylinders = build_cylinders(cell_pts[:-1], cell_pts[1:], 0.01, (0, 255, 0))
        spheres = build_spheres(cell_pts[:-1], 0.02, (255, 0, 0))
        mesh_list.append(cylinders)
        mesh_list.append(spheres)
    
    mesh = trimesh.util.concatenate(mesh_list)
    return mesh


def subdivide(cells: list, node_pts: np.array):

    new_pt_mesh_list = []
    new_edge_mesh_list = []
    for cell in cells:
        center = np.mean(node_pts[cell], axis=0)
        edge_pt_ids = cell
        edge_pt_ids.append(cell[0])
        edge_pts = node_pts[edge_pt_ids]
        edge_centers = (edge_pts[:-1] + edge_pts[1:]) / 2
        
        new_pt_mesh_list.append(build_spheres(center, 0.02, (0, 0, 255)))
        new_pt_mesh_list.append(build_spheres(edge_centers, 0.02, (0, 0, 255)))

        new_edge_mesh_list.append(
            build_cylinders(center[None, ...].repeat(len(cell) - 1, axis=0), edge_centers, 0.01, (0, 255, 255)))

        # break

    new_pt_mesh = trimesh.util.concatenate(new_pt_mesh_list)
    new_edge_mesh = trimesh.util.concatenate(new_edge_mesh_list)

    return new_pt_mesh, new_edge_mesh


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get topology of a mesh')
    parser.add_argument('--input', type=str, help='Input')

    args = parser.parse_args()
    input_dir = Path(args.input)

    mesh = trimesh.load(str(input_dir / 'segmented_mesh.ply'), process=False, maintain_order=True)
    with open(str(input_dir / 'mask.json'), 'r') as f:
        mask = json.load(f)
    
    complex_builder = ComplexBuilder(mesh, mask)
    graph = complex_builder.build_complex_recursive()
    print(graph)

    node_pts = mesh.vertices[graph['node_ids']]

    graph_mesh = vis_graph(graph['cells'], node_pts)
    graph_mesh.export(str(input_dir / 'graph_raw.ply'))

    new_pt_mesh, new_edge_mesh = subdivide(graph['cells'], node_pts)
    new_pt_mesh.export(str(input_dir / 'new_pts.ply'))
    new_edge_mesh.export(str(input_dir / 'new_edges.ply'))

    # pcd = trimesh.PointCloud(node_pts)
    # pcd.export(str(input_dir / 'node_pts.ply'))