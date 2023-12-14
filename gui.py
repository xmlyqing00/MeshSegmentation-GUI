import argparse
import os
import numpy as np
import trimesh
import json
from loguru import logger
from vedo import Plotter, Sphere, Text2D, Mesh, write, Line, utils
from potpourri3d import EdgeFlipGeodesicSolver
from src.utils import NpEncoder
from src.mesh_tools import split_mesh, floodfill_label_mesh
from PIL import Image


class GUI:

    def __init__(
            self, 
            tri_mesh: trimesh.Trimesh, 
            output_dir: str, 
            plt: Plotter, 
            mask: list = None,
            close_point_merging: bool = True,
        ) -> None:
        
        self.loop_flag = True
        self.merge_mode = False
        self.close_point_merging = close_point_merging

        self.plt = plt
        self.output_dir = output_dir
        if mask:
            self.mask = mask
        else:
            self.mask = [np.arange(len(tri_mesh.faces))]
    
        self.picked_pts = []
        self.all_picked_pts = []
        self.arrow_objs = []
        self.patches_to_merge = []
        self.mask_history = []
        self.tri_mesh_history = []

        tmp_tri_mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.faces, process=False, validate=False)
        self.mesh = utils.trimesh2vedo(tmp_tri_mesh)
        self.plt.add(self.mesh)

        self.mesh_size = np.array([
            self.tri_mesh.vertices[:, 0].max() - self.tri_mesh.vertices[:, 0].min(),
            self.tri_mesh.vertices[:, 1].max() - self.tri_mesh.vertices[:, 1].min(),
            self.tri_mesh.vertices[:, 2].max() - self.tri_mesh.vertices[:, 2].min(),
        ])
        self.thres_nearest_pt = 2e-2 * self.mesh_size.min()
        self.point_size = 1e-2 * self.mesh_size.min()
        self.boundary_size = 5e-3 * self.mesh_size.max()
        self.shadow_dist = 0.2 * self.mesh_size.min()

        if not self.close_point_merging:
            self.point_size *= 0.05

        self.enable_shadow = False
        if self.enable_shadow:
            self.mesh.add_shadow('z', -self.shadow_dist)

        self.apply_mask()


    def on_mouse_click(self, event):

        # mesh = event.actor
        mouse_pt = event.picked3d

        if mouse_pt is None:
            return
        
        logger.debug(f'Mouse click: {mouse_pt}')

        if self.merge_mode:
            self.merge_patch(mouse_pt)       
            return 
        
        if self.close_point_merging:
            mouse_pt = self.check_nearest_point(mouse_pt)

        pid = self.mesh.closest_point(mouse_pt, return_point_id=True)
        pt = self.mesh.vertices()[pid]

        picked_pt = Sphere(pt, r=self.point_size, c='black')

        self.picked_pts.append({
            'pos': pt,
            'id': pid,
            'obj': picked_pt
        })

        self.plt.add(picked_pt).render()


    def on_key_press(self, event):

        if event.keypress == 'v' or event.keypress == 'g':
            self.stack_picked_pts(event.keypress == 'g')
        elif event.keypress == 's':
            self.compute_shortest_path()
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
        
        picked_pt_pid = []
        for picked_pts in self.all_picked_pts: 
            picked_pt_pid.append(
                [self.mesh.closest_point(x['pos'], return_point_id=True) for x in picked_pts]
            )

        if len(picked_pt_pid) > 0:
        
            self.mask = floodfill_label_mesh(
                self.tri_mesh, 
                self.boundary_edges,
                picked_pt_pid, 
            )

            self.apply_mask()
        else:
            print('No picked points. Nothing changed.')


    def apply_mask(self):

        self.face_patches = -1 + np.zeros(len(self.tri_mesh.faces), dtype=np.int32)
        for i, seg in enumerate(self.mask):
            for fid in seg:
                self.face_patches[fid] = i
        
        group_num = len(self.mask)
        for i in range(len(self.face_patches)):
            if self.face_patches[i] == -1:
                logger.info(f'Found a single face. Face id: {i}')
                self.face_patches[i] = group_num
                group_num += 1
                self.mask.append([i])

        f_adj = self.tri_mesh.face_adjacency
        fe_adj = np.sort(self.tri_mesh.face_adjacency_edges, axis=1)

        self.boundary_pts = set()
        self.boundary_edges = set()
        for i in range(len(f_adj)):
            if self.face_patches[f_adj[i][0]] != self.face_patches[f_adj[i][1]]:
                self.boundary_pts.add(fe_adj[i][0])
                self.boundary_pts.add(fe_adj[i][1])
                self.boundary_edges.add((fe_adj[i][0], fe_adj[i][1]))
        
        logger.info(f'Patch number: {len(self.mask)}')
        self.update_mesh_color()

        unique_edges = self.tri_mesh.edges[trimesh.grouping.group_rows(self.tri_mesh.edges_sorted, require_count=1)]
        logger.info(f'Edge number of open boundary: {unique_edges.shape[0]}')
        self.boundary_pts.update(unique_edges.flatten().tolist())


    def toggle_merge_mode(self):
        self.merge_mode = not self.merge_mode
        self.patches_to_merge = []
        if self.merge_mode:
            logger.info('Merge mode on.')
        else:
            logger.info('Merge mode off.')


    def merge_patch(self, mouse_pt: list):
        
        fid = self.mesh.closest_point(mouse_pt, return_cell_id=True)
        patch_id = self.face_patches[fid]
        logger.info(f'Selected face id {fid}. patch id {patch_id}')
        if patch_id in self.patches_to_merge:
            logger.warning('This patch has been picked. Clear the selected patches.')
            self.patches_to_merge = []
            return
        
        self.patches_to_merge.append(patch_id)
        if len(self.patches_to_merge) < 2:
            return
        
        self.mask_history.append(self.mask.copy())
        self.tri_mesh_history.append(self.tri_mesh)
        logger.success(f'Merge two patches {self.patches_to_merge}')

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
            logger.info(f'Found nearest existing pt: {boundary_pts[idx]}. Distance: {dist[idx]}')
            return boundary_pts[idx]
        else:
            logger.info(f'The nearest existing pt is too far. Distance: {dist[idx]}')
            return mouse_pt

    def update_mesh_color(self):

        for group_idx, group in enumerate(self.mask):
            self.mesh.cellcolors[group] = cmap[group_idx % 20]
           
        self.plt.remove(self.arrow_objs)
        self.arrow_objs = []
        for eid, edge in enumerate(list(self.boundary_edges)):
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
        logger.success('Stack picekd pts. Loop:', loop_flag)
        if len(self.picked_pts) < 2:
            logger.warning('The number of the picked points must be larger than 2.')
            return

        if loop_flag:
            self.picked_pts.append(self.picked_pts[0])

        self.all_picked_pts.append(self.picked_pts)
        self.picked_pts = []


    def compute_shortest_path(self):
        
        if len(self.picked_pts) > 0:
            logger.warning('You have unstacked picked pts. Stack them first by press f/g. Do nothing.')
            return

        logger.success(f'Compute the SHORTEST path. Number of paths of picked pts: {len(self.all_picked_pts)}')

        old_mesh = self.mesh
        self.mask_history.append(self.mask)
        self.tri_mesh_history.append(self.tri_mesh)

        if self.enable_shadow:
            self.mesh.add_shadow('z', -self.shadow_dist)

        self.apply_mask()
        self.update_mask()
        self.plt.remove(old_mesh)
        self.plt.add(self.mesh)

        self.plt.render()
        self.save()


    def compute_geodesic_path(self):
        
        if len(self.picked_pts) > 0:
            logger.warning('You have unstacked picked pts. Stack them first by press f/g. Do nothing.')
            return

        logger.success(f'Compute the GEODESIC path. Number of paths of picked pts: {len(self.all_picked_pts)}')
        v = self.mesh.vertices()
        # print('Compute geodesic path.', f'Number of paths of picked pts: {len(self.all_picked_pts)}')
        # v = self.mesh.vertices
        f = np.array(self.mesh.faces())
        path_solver = EdgeFlipGeodesicSolver(v, f) # shares precomputation for repeated solves

        new_pts = []
        for picked_pts in self.all_picked_pts:
            for i in range(1, len(picked_pts)):
                v_start = picked_pts[i - 1]['id']
                v_end = picked_pts[i]['id']
                # logger.debug(v_start, v_end)
                path_pts = path_solver.find_geodesic_path(v_start, v_end)
                # print(f'{v_start} -> {v_end}:', 'Geodesic path', path_pts)
                new_pts.extend(path_pts[1:-1])

        old_mesh = self.mesh
        self.mask_history.append(self.mask)
        self.tri_mesh_history.append(self.tri_mesh)

        self.tri_mesh, self.mask = split_mesh(self.tri_mesh, np.array(new_pts), self.face_patches)
        self.mesh = utils.trimesh2vedo(self.tri_mesh)
        # print('cell colors len in compute', len(self.mesh.cellcolors))
        # self.mesh = Mesh([self.tri_mesh.vertices.tolist(), self.tri_mesh.faces.tolist()])
        if self.enable_shadow:
            self.mesh.add_shadow('z', -self.shadow_dist)

        self.apply_mask()
        self.update_mask()
        self.plt.remove(old_mesh)
        self.plt.add(self.mesh)

        self.plt.render()
        self.save()


    def clear_last_pt(self):
        logger.success('Clear the last picked point')
        if len(self.picked_pts) > 0:
            self.plt.remove(self.picked_pts[-1]['obj'])
            self.picked_pts.pop()
            self.plt.render()


    def clear_all_pts(self):
        logger.success('Clear the picked points')
        
        for pt in self.picked_pts:
            self.plt.remove(pt['obj'])
        self.picked_pts = []

        for picked_pts in self.all_picked_pts:
            for pt in picked_pts:
                self.plt.remove(pt['obj'])
        self.all_picked_pts = []
        
        self.plt.render()


    def load_last_mask(self):
        
        if len(self.mask_history) > 0:
            print(len(self.mask_history[-1]))
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

            logger.success('Load the last mask. The number of history masks is', len(self.mask_history))
        else:
            logger.warning('This is already the first patch mask.')


    def save(self):

        obj_name = os.path.basename(self.output_dir)

        obj_path = os.path.join(self.output_dir, f'{obj_name}.ply')
        viz_obj_path = os.path.join(self.output_dir, f'{obj_name}_viz.ply')
        mask_path = os.path.join(self.output_dir, 'mask.json')

        with open(mask_path, 'w') as f:
            json.dump(self.mask, f, cls=NpEncoder, ensure_ascii=False, indent=4)

        self.tri_mesh.export(obj_path)
        write(self.mesh, viz_obj_path)

        self.clear_all_pts()

        logger.success(f'Saved mask and mesh to {self.output_dir}.')
        logger.info('Ready for the next segmentation.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='data/bimba_nf1M.obj', help='Input mesh path.')
    parser.add_argument('--mask', type=str, default=None, help='Input mask path.')
    parser.add_argument('--outdir', type=str, default='./output', help='Output directory.')
    parser.add_argument('--no-close-point-merging', action='store_true', help='Disable the close point merging.')
    args = parser.parse_args()    
    logger.info(f'Arguments: {args}')

    help_text = 'Mouse left-click to pick vertex.\n' \
        'Press v/g to stack path/loop of picked vertices.\n' \
        'Press z to compute Geodesic path/loop.\n' \
        'Press s to compute Shortest path/loop (no new vertex).\n' \
        'Press b to clear the LAST picked points.\n' \
        'Press c to clear ALL picked points.\n' \
        'Press d to load the last segmentations.\n' \
        'Press m to toggle patch merging mode.\n' \
        'Press h to see more help and default features.'
    logger.info(f'Keyboard shortcuts:\n{help_text}')

    msg = Text2D(pos='bottom-left', font="VictorMono", s=0.6) 
    msg.text(help_text)

    color_img = np.asarray(Image.open('assets/cm_tab20.png').convert('RGBA'))
    cmap = []
    for i in range(20):
        c = 25 * (i % 20) + 10
        cmap.append(color_img[20, c ])

    # Load the OBJ file
    tri_mesh = trimesh.load(args.input, maintain_order=True, process=False, fix_texture=False, validate=False)
    logger.info(f'Mesh vertices and faces: {tri_mesh.vertices.shape}, {tri_mesh.faces.shape}')
    obj_name = os.path.basename(args.input).split('.')[0]
    output_dir = os.path.join(args.outdir, obj_name)
    os.makedirs(args.outdir, exist_ok=True)
    logger.info(f'Output directory: {output_dir}.')

    # Try to load the mask
    mask = None
    if args.mask and os.path.exists(args.mask):
        logger.success(f'Load mask from {args.mask}')
        with open(args.mask, 'r') as f:
            mask = json.load(f)

    plt = Plotter(axes=8, bg='white', size=(1200, 800))
    if args.no_close_point_merging:
        close_point_merging = False
    else:
        close_point_merging = True

    gui = GUI(tri_mesh, output_dir, plt, mask, close_point_merging)  

    plt.add_callback('left click', gui.on_mouse_click)
    plt.add_callback('key press', gui.on_key_press)

    plt.add(msg)
    plt.show()

    plt.close()
