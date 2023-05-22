import pickle
import argparse
import os
import numpy as np
import trimesh
import json
from vedo import load, Plotter, Sphere, Arrow, Text2D, Mesh, write, Line
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

    def __init__(self, mesh: Mesh, output_dir: str, plt: Plotter, mask: list = None) -> None:
        
        self.loop_flag = True
        self.merge_mode = False

        self.plt = plt
        self.output_dir = output_dir
        if mask:
            self.mask = mask
        else:
            self.mask = [[np.arange(len(mesh.faces()))]]
    
        self.picked_pts = []
        self.all_picked_pts = []
        self.arrow_objs = []
        self.patches_to_merge = []
        self.mask_history = []
        self.tri_mesh_history = []

        self.mesh = mesh
        
        self.tri_mesh = trimesh.Trimesh(
            mesh.vertices(), 
            mesh.faces(),
            process=False,
            maintain_order=True
        )
        self.plt.add(self.mesh)

        self.mesh_size = np.array([
            self.tri_mesh.vertices[:, 0].max() - self.tri_mesh.vertices[:, 0].min(),
            self.tri_mesh.vertices[:, 1].max() - self.tri_mesh.vertices[:, 1].min(),
            self.tri_mesh.vertices[:, 2].max() - self.tri_mesh.vertices[:, 2].min(),
        ])
        self.thres_nearest_pt = 2e-2 * self.mesh_size.min()
        self.point_size = 1e-2 * self.mesh_size.min()
        self.boundary_size = 1e-2 * self.mesh_size.min()
        self.shadow_dist = 0.2 * self.mesh_size.min()

        self.enable_shadow = False
        if self.enable_shadow:
            self.mesh.add_shadow('z', -self.shadow_dist)

        self.apply_mask()


    def on_mouse_click(self, event):

        # mesh = event.actor
        mouse_pt = event.picked3d

        if mouse_pt is None:
            return
        
        pid = self.mesh.closest_point(mouse_pt, return_point_id=True)
        print('Mouse click:', mouse_pt, 'Vertex ID:', pid)

        if self.merge_mode:
            self.merge_patch(pid)       
            return 
        
        mouse_pt = self.check_nearest_point(mouse_pt)
        pid = self.mesh.closest_point(mouse_pt, return_point_id=True)
        pt = self.mesh.vertices()[pid]

        picked_pt = Sphere(pt, r=self.point_size, c='black')
        picked_pt.name = f'pt_{len(self.picked_pts)}'

        print(f'Picked a vertex on the mesh. ID: {picked_pt.name}, Vertex ID: {pid}. Position: {pt}')
        self.picked_pts.append({
            'pos': pt,
            'id': pid,
            'name': picked_pt.name
        })

        self.plt.add(picked_pt).render()


    def on_key_press(self, event):

        if event.keypress == 'v' or event.keypress == 'g':
            self.stack_picked_pts(event.keypress == 'g')
        elif event.keypress == 'z':
            self.compute_geodesic_path()
        elif event.keypress == 'b':
            self.clear_last_pt()
        elif event.keypress == 'c':
            self.clear_all_pts()
        elif event.keypress == 'd':
            self.load_last_mask()
        elif event.keypress == 'm':
            self.toggle_merge_mode()
    
    
    def update_mask(self):
        
        picked_pt_pos = []
        for picked_pts in self.all_picked_pts:
            picked_pt_pos.append([x['pos'] for x in picked_pts])

        if len(picked_pt_pos) > 0:
        
            self.mask = floodfill_label_mesh(
                self.mesh, 
                self.boundary_edges,
                picked_pt_pos, 
                self.tri_mesh.face_adjacency,
                self.tri_mesh.face_adjacency_edges
            )

            self.apply_mask()
        else:
            print('No picked points. Nothing changed.')


    def apply_mask(self):

        self.face_patches = np.zeros(len(self.tri_mesh.faces), dtype=np.int32)
        for i, seg in enumerate(self.mask):
            print('patch_size:', i, len(seg))
            for fid in seg:
                self.face_patches[fid] = i

        f_adj = self.tri_mesh.face_adjacency
        fe_adj = np.sort(self.tri_mesh.face_adjacency_edges, axis=1)

        self.boundary_pts = set()
        self.boundary_edges = set()
        for i in range(len(f_adj)):
            if self.face_patches[f_adj[i][0]] != self.face_patches[f_adj[i][1]]:
                self.boundary_pts.add(fe_adj[i][0])
                self.boundary_pts.add(fe_adj[i][1])
                self.boundary_edges.add((fe_adj[i][0], fe_adj[i][1]))
        
        print('Patch number', len(self.mask))
        self.update_mesh_color()

        unique_edges = self.tri_mesh.edges[trimesh.grouping.group_rows(self.tri_mesh.edges_sorted, require_count=1)]
        print('unique edges', unique_edges.shape)
        self.boundary_pts.update(unique_edges.flatten().tolist())


    def toggle_merge_mode(self):
        self.merge_mode = not self.merge_mode
        if self.merge_mode:
            self.patches_to_merge = []
            print('Merge mode on.')
        else:
            print('Merge mode off.')


    def merge_patch(self, pid: int):
                
        fid = self.tri_mesh.vertex_faces[pid][0]    
        patch_id = self.face_patches[fid]
        if patch_id in self.patches_to_merge:
            print('This patch has been picked. Clear the selected patches.')
            self.patches_to_merge = []
            return
        
        self.patches_to_merge.append(patch_id)
        if len(self.patches_to_merge) < 2:
            return
        
        self.mask_history.append(self.mask)
        self.tri_mesh_history.append(self.tri_mesh)
        print('Merge two patches', self.patches_to_merge)

        patch_id0 = self.patches_to_merge[0]
        patch_id1 = self.patches_to_merge[1]
        self.mask[patch_id0].extend(self.mask[patch_id1])
        self.mask.pop(patch_id1)
        self.patches_to_merge = []

        self.apply_mask()
        self.save()
        

    def check_nearest_point(self, mouse_pt):

        if len(self.boundary_pts) == 0:
            return mouse_pt
        
        boundary_pts = self.tri_mesh.vertices[list(self.boundary_pts)]
        dist = np.linalg.norm(boundary_pts - mouse_pt, axis=1)
        idx = np.argmin(dist)
        if dist[idx] < self.thres_nearest_pt:
            print('Find nearest existing pt:', boundary_pts[idx], 'Distance:', dist[idx])
            return boundary_pts[idx]
        else:
            print('The nearest existing pt is too far. Distance:', dist[idx])
            return mouse_pt


    def update_mesh_color(self):

        for group_idx, group in enumerate(self.mask):

            self.mesh.cellcolors[group] = cmap[group_idx % 20]
            # if group_idx == 72:
            #     self.mesh.cellcolors[group] = (255, 0, 0, 255)
            #     f = self.tri_mesh.faces[group]
            #     print('f', f)
            #     v = self.tri_mesh.vertices[f]
            #     print('v', v)
                
            # else:
            #     self.mesh.cellcolors[group] = (0, 0, 0, 255)
            #     print('group_idx', group_idx, self.mesh.cellcolors[group[0]])

        
        self.plt.remove(self.arrow_objs)
        self.arrow_objs = []
        for eid, edge in enumerate(list(self.boundary_edges)):
            # arrow = Arrow(path_pts[path_id - 1], path_pts[path_id], s=self.boundary_size, c='black')
            arrow = Line(
                self.tri_mesh.vertices[edge[0]],
                self.tri_mesh.vertices[edge[1]],
                lw=self.boundary_size, 
                c='black'
            )
            self.arrow_objs.append(arrow)
        self.plt.add(self.arrow_objs)
        self.plt.render()
            

    def stack_picked_pts(self, loop_flag: bool = False):
        print('Stack picekd pts. Loop:', loop_flag)
        if len(self.picked_pts) < 2:
            print('The number of the picked points is less than 2.')
            return

        if loop_flag:
            self.picked_pts.append(self.picked_pts[0])

        self.all_picked_pts.append(self.picked_pts)
        self.picked_pts = []


    def compute_geodesic_path(self):
        
        if len(self.picked_pts) > 0:
            print('You have unstacked picked pts. Stack them first by press f/g.')
            print('Do nothing.')
            return

        print('Compute geodesic path.', f'Number of paths of picked pts: {len(self.all_picked_pts)}')
        v = self.mesh.vertices()
        f = np.array(self.mesh.faces())
        path_solver = EdgeFlipGeodesicSolver(v, f) # shares precomputation for repeated solves

        new_pts = []
        for picked_pts in self.all_picked_pts:
            for i in range(1, len(picked_pts)):
                v_start = picked_pts[i - 1]['id']
                v_end = picked_pts[i]['id']
                path_pts = path_solver.find_geodesic_path(v_start, v_end)
                # print(f'{v_start} -> {v_end}:', 'Geodesic path', path_pts)
                new_pts.extend(path_pts[1:-1])

        old_mesh = self.mesh
        self.mask_history.append(self.mask)
        self.tri_mesh_history.append(self.tri_mesh)

        self.tri_mesh, self.mask = split_mesh(self.tri_mesh, np.array(new_pts), self.face_patches)
        self.mesh = Mesh([self.tri_mesh.vertices.tolist(), self.tri_mesh.faces.tolist()])
        if self.enable_shadow:
            self.mesh.add_shadow('z', -self.shadow_dist)

        self.apply_mask()
        self.update_mask()
        self.plt.remove(old_mesh)
        self.plt.add(self.mesh)

        self.plt.render()
        self.save()


    def clear_last_pt(self):
        print('Clear the last picked point')
        if len(self.picked_pts) > 0:
            self.plt.remove(self.picked_pts[-1]['name'])
            self.picked_pts.pop()
            self.plt.render()


    def clear_all_pts(self):
        print('Clear the picked points')
        
        for pt in self.picked_pts:
            self.plt.remove(pt['name'])
        self.picked_pts = []

        for picked_pts in self.all_picked_pts:
            for pt in picked_pts:
                self.plt.remove(pt['name'])
        self.all_picked_pts = []
        
        self.plt.render()


    def load_last_mask(self):
        
        if len(self.mask_history) > 0:
            self.tri_mesh = self.tri_mesh_history.pop()
            self.mask = self.mask_history.pop()
            old_mesh = self.mesh
            self.mesh = Mesh([self.tri_mesh.vertices.tolist(), self.tri_mesh.faces.tolist()])
            if self.enable_shadow:
                self.mesh.add_shadow('z', -self.shadow_dist)
            self.apply_mask()
            
            self.plt.remove(old_mesh)
            self.plt.add(self.mesh)
            self.plt.render()
            self.save()

            print('Load the last mask. The number of history masks is', len(self.mask_history))
        else:
            print('This is already the first patch mask.')


    def save(self):
        print(f'Saved mask.json. Ready for the next segmentation.')

        obj_name = os.path.basename(self.output_dir)
        single_obj_dir = os.path.join(self.output_dir, 'single')
        os.makedirs(single_obj_dir, exist_ok=True)

        obj_path = os.path.join(single_obj_dir, f'{obj_name}.obj')
        mask_path = os.path.join(self.output_dir, 'mask.json')
        print('Save to', obj_path, mask_path)

        with open(mask_path, 'w') as f:
            json.dump(self.mask, f, cls=NpEncoder, ensure_ascii=False, indent=4)

        write(self.mesh, obj_path)

        self.clear_all_pts()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='data/manohand_0.obj', help='Input mesh path.')
    parser.add_argument('--mask', type=str, default=None, help='Input mask path.')
    parser.add_argument('--outdir', type=str, default='./output', help='Output path.')
    args = parser.parse_args()

    print(args)

    msg = Text2D(pos='bottom-left', font="VictorMono", s=0.6) 
    msg.text(
        'Mouse left-click to pick vertex.\n' \
        'Press v/g to stack Geodesic path/loop.\n' \
        'Press z to compute Geodesic path/loop.\n' \
        'Press b to clear the LAST picked points.\n' \
        'Press c to clear ALL picked points.\n' \
        'Press d to load the last segmentations.\n' \
        'Press m to toggle patch merging mode.\n' \
        'Press h to see more help and default features.'
    )

    color_img = np.asarray(Image.open('assets/cm_tab20.png').convert('RGBA'))
    cmap = []
    for i in range(20):
        c = 25 * (i % 20) + 10
        cmap.append(color_img[20, c])

    # Load the OBJ file
    mesh = load(args.input)
    obj_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.outdir, obj_name)
    os.makedirs(args.outdir, exist_ok=True)

    # Try to load the mask
    mask = None
    if args.mask and os.path.exists(args.mask):
        with open(args.mask, 'r') as f:
            mask = json.load(f)

    plt = Plotter(axes=0, bg='white', size=(1200, 800))
    gui = GUI(mesh, output_dir, plt, mask)  

    plt.add_callback('left click', gui.on_mouse_click)
    plt.add_callback('key press', gui.on_key_press)

    plt.add(msg)
    plt.show()

    plt.close()
