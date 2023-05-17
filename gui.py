import pickle
import argparse
import os
import numpy as np
import trimesh
import json
from vedo import load, Plotter, Sphere, Arrow, Text2D, Mesh, write
from potpourri3d import EdgeFlipGeodesicSolver
from mesh_tools import split_mesh, floodfill_label_mesh
from PIL import Image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()


class GUI:

    def __init__(self, mesh: Mesh, output_path: str, mask: list = None) -> None:
        
        self.picked_pts = []
        
        self.all_picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []
        self.seg_n = 0
        self.loop_flag = True

        self.output_path = output_path
        self.mesh = mesh
        self.tri_mesh = trimesh.Trimesh(
            mesh.vertices(), 
            mesh.faces(),
            process=False,
            maintain_order=True
        )

        if mask:
            f_labels = np.zeros(len(mesh.faces()), dtype=np.int32)
            for i, seg in enumerate(mask):
                for fid in seg:
                    f_labels[fid] = i + 1

            f_adj = self.tri_mesh.face_adjacency
            fe_adj = self.tri_mesh.face_adjacency_edges

            for i in range(len(f_adj)):
                if f_labels[f_adj[i][0]] != f_labels[f_adj[i][1]]:
                    self.all_picked_pts.append([fe_adj[i][0], fe_adj[i][1]])
        
        self.update_mesh_color()

    def on_mouse_click(self, event):
        # mesh = event.actor
        mesh = self.mesh
        if not mesh:
            return
        
        mouse_pt = event.picked3d

        if mouse_pt is None:
            return

        pid = mesh.closest_point(mouse_pt, return_point_id=True)
        pt = mesh.vertices()[pid]

        picked_pt = Sphere(pt, r=0.01, c='black')
        picked_pt.name = f'{self.seg_n}_{len(self.picked_pts)}'

        print(f'Picked a vertex on the mesh. ID: {picked_pt.name}, Vertex ID: {pid}. Position: {pt}')
        self.picked_pts.append({
            'pos': pt,
            'id': pid,
            'name': picked_pt.name
        })

        plt.add(picked_pt).render()

    def on_key_press(self, event):

        if event.keypress == 'f' or event.keypress == 'g':
            if event.keypress == 'g':
                self.loop_flag = True
            else:
                self.loop_flag = False

            self.compute_geodesic_path(event.keypress == 'g')
        elif event.keypress == 'c':
            self.clear_pts()
    
    def update_mesh_color(self):

        picked_pt_pos = [x['pos'] for x in self.picked_pts]
        # print(picked_pt_pos)

        if len(picked_pt_pos) > 0:
            if self.loop_flag:
                picked_pt_pos.append(picked_pt_pos[0])
            self.all_picked_pts.append(picked_pt_pos)

        groups = floodfill_label_mesh(
            self.mesh, 
            self.all_picked_pts, 
            self.tri_mesh.face_adjacency,
            self.tri_mesh.face_adjacency_edges
        )

        # f_colors = np.zeros((len(self.tri_mesh.faces), 3), np.uint8)
        print('group number', len(groups))
        self.mask_faces = []
        for group_idx, group in enumerate(groups):
            group = list(group)
            self.mask_faces.append(group)
            self.mesh.cellcolors[group] = cmap[group_idx % 20]
        
        # self.mesh.cellcolors = f_colors
        
    def compute_geodesic_path(self, loop_flag: bool = False):
        print('Compute the Geodesic path. Loop:', loop_flag)
        print(self.picked_pts)
        if len(self.picked_pts) < 2:
            print('The number of the picked points is less than 2.')
            return

        v = self.mesh.vertices()
        f = np.array(self.mesh.faces())
        # print('shapes', len(v), f.shape)
        path_solver = EdgeFlipGeodesicSolver(v, f) # shares precomputation for repeated solves

        loop_pts = self.picked_pts
        if loop_flag:
            loop_pts.append(loop_pts[0])

        new_pts = []
        for i in range(1, len(loop_pts)):
            v_start = loop_pts[i - 1]['id']
            v_end = loop_pts[i]['id']
            path_pts = path_solver.find_geodesic_path(v_start, v_end)
            print(f'{v_start} -> {v_end}:', 'Geodesic path', path_pts)
            self.geodesic_paths.append(path_pts)
            new_pts.extend(path_pts[1:-1])

            for path_id in range(1, path_pts.shape[0]):
                arrow = Arrow(path_pts[path_id - 1], path_pts[path_id], s=0.0005, c='black')
                arrow.name = f'{self.seg_n}_{len(self.arrow_names)}'
                plt.add(arrow)
                self.arrow_names.append(arrow.name)

        # seg_path = np.concatenate(self.geodesic_paths, axis=0)
        new_mesh = split_mesh(self.tri_mesh, np.array(new_pts))
        plt.remove(self.mesh)

        self.tri_mesh = new_mesh
        self.mesh = Mesh([new_mesh.vertices.tolist(), new_mesh.faces.tolist()])
        self.update_mesh_color()

        plt.add(self.mesh)

        self.save()
        plt.render()

    def clear_pts(self):
        print('Clear the picked points and paths.')

        for pt in self.picked_pts:
            plt.remove(pt['name'])
            # print(pt['name'])
        self.picked_pts = []

        for arrow_name in self.arrow_names:
            plt.remove(arrow_name)
            # print(arrow_name)
        self.arrow_names = []

        self.geodesic_paths = []

        plt.render()

    def save(self):
        self.seg_n += 1
        print(f'Save {self.seg_n} geodesic paths.', 'Ready for the next segmentation.')

        mask_path = self.output_path.replace('.obj', '_mask.json')
        with open(mask_path, 'w') as f:
            json.dump(self.mask_faces, f, cls=NpEncoder, ensure_ascii=False, indent=4)

        print(self.output_path)
        write(self.mesh, self.output_path)

        self.picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='data/manohand_0.obj', help='Input mesh path.')
    parser.add_argument('--outdir', type=str, default='./output', help='Output path.')
    args = parser.parse_args()

    print(args)

    msg = Text2D(pos='bottom-left', font="VictorMono") 
    msg.text(
        'Mouse left-click to pick vertex.\n' \
        'Press f/g to compute Geodesic path/loop.\n' \
        'Press c to clear the points.\n' \
        'Press h to see more help.'
    )

    color_img = np.asarray(Image.open('assets/cm_tab20.png').convert('RGBA'))
    cmap = []
    for i in range(20):
        c = 25 * (i % 20) + 10
        cmap.append(color_img[20, c])

    # Load the OBJ file
    mesh = load(args.input)
    obj_name = os.path.basename(args.input).split('.')[0]
    output_path = os.path.join(args.outdir, f'{obj_name}_labeled.obj')
    os.makedirs(args.outdir, exist_ok=True)

    # Try to load the mask
    mask_path = os.path.join(os.path.dirname(args.input), 'mask.json')
    mask = None
    if os.path.exists(mask_path):
        with open(mask_path, 'r') as f:
            mask = json.load(f)

    gui = GUI(mesh, output_path, mask)

    plt = Plotter(axes=8, bg='white')

    plt.add_callback('left click', gui.on_mouse_click)
    plt.add_callback('key press', gui.on_key_press)

    plt.show(mesh, msg)
    plt.close()

    
