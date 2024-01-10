import igl
import numpy as np
import trimesh
from copy import deepcopy
from view_psd_data import *
import shutil
import argparse
from loguru import logger
import matplotlib
import matplotlib.cm as cm

from vedo import Mesh as VedoMesh
from vedo import show, Plotter, Arrows, Sphere, Spheres, Text2D, Line

from mesh_segmentor import MeshSegmentator

## igl

def trace_surface_flow(v, f):
    v1, v2, k1, k2 = igl.principal_curvature(v, f)
    h2 = 0.5 * (k1 + k2)
    avg = igl.avg_edge_length(v, f) / 2.0
    return v1, v2, h2, avg
    
    # p = plot(v, f, h2, shading={"wireframe": False}, return_plot=True)
    # p.add_lines(v + v1 * avg, v - v1 * avg, shading={"line_color": "red"})
    # p.add_lines(v + v2 * avg, v - v2 * avg, shading={"line_color": "green"})

def compute_colormap(d, type='continuous'):
    if type == 'continuous':
        norm = matplotlib.colors.Normalize(d.min(), d.max(), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    elif type == 'discrete':
        norm = matplotlib.colors.Normalize(0, 20, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)
    colors = [(r, g, b, a) for r, g, b, a in mapper.to_rgba(d)]
    return np.array(colors)*255


class AutoSegGUI:
    
    cmap = compute_colormap(np.arange(20), type='discrete')

    def __init__(
            self, 
            mesh: VedoMesh,
            mask,
            output_dir: str, 
            plt: Plotter, 
            smooth_flag: bool,
            smooth_deg: int,
        ) -> None:
        
        
        self.loop_flag = True
        
        self.init_mesh = mesh
        self.init_mask = mask
        self.init_mesh = self.update_mesh_color(self.init_mesh, self.init_mask)

        self.tri_mesh = trimesh.Trimesh(mesh.vertices, mesh.cells, process=False, maintain_order=True)

        self.mesh_size = np.array([
            self.tri_mesh.vertices[:, 0].max() - self.tri_mesh.vertices[:, 0].min(),
            self.tri_mesh.vertices[:, 1].max() - self.tri_mesh.vertices[:, 1].min(),
            self.tri_mesh.vertices[:, 2].max() - self.tri_mesh.vertices[:, 2].min(),
        ])
        self.point_size = 1e-3 * self.mesh_size.min()

        self.plt = plt
        self.output_dir = output_dir
        self.segmentor = MeshSegmentator(self.tri_mesh, mask, smooth_flag=smooth_flag, smooth_deg=smooth_deg)
        self.segmentor(b_close_holes=False)
        self.refined_mask = self.segmentor.mask
        self.refined_mesh = VedoMesh([self.segmentor.mesh.vertices, self.segmentor.mesh.faces])
        self.refined_mesh = self.update_mesh_color(self.refined_mesh, self.refined_mask)

        self.show_original = True
        self.content = []

        self.render_mesh(self.init_mesh)
        # self.render_mesh(self.refined_mesh)


    def update_mesh_color(self, mesh, mask):
        for group_idx, group in enumerate(mask):
            mesh.cellcolors[group] = self.cmap[group_idx % 20]
        return mesh


    def on_key_press(self, event):
        if event.keypress == 'u':
            self.toggle_display_mode()


    def toggle_display_mode(self):
        self.show_original = not self.show_original
        self.plt.clear()
        if self.show_original:
            self.clear_content()
            self.render_mesh(self.init_mesh)
            logger.info('Show original mesh segmentation.')
        else:
            self.render_mesh(self.refined_mesh)
            self.render_content()
            logger.info('Show refined mesh segmentation.')
        
    
    def render_principal_curvature(self):
        v = np.array(self.mesh.vertices)
        f = np.array(self.mesh.cells, dtype=np.int32)

        ## 
        v1, v2, h2, avg = trace_surface_flow(v, f)

        ## color the mesh with mean curvature
        face_h2 = np.mean(h2[f], axis=1)
        rgbd = compute_colormap(face_h2)
        print(rgbd.shape)
        mesh.cellcolors = rgbd

        ## draw principal flows
        arrows_1 = Arrows(v + v1 * avg, v - v1 * avg, s=0.5).c('red')
        arrows_2 = Arrows(v + v2 * avg, v - v2 * avg, s=0.5).c('green')
        plt.add(arrows_1)
        plt.add(arrows_2)
        plt.render()


    def render_content(self):        
        ## render cuts and boundaries
        self.content = []
        for cut in self.segmentor.cut_list:
            if cut.dead:
                continue
            self.content.append(Spheres(cut.points, c='red', r=self.point_size))
            for p_idx in range(1, len(cut.points)):
                line = Line(
                    cut.points[p_idx-1], 
                    cut.points[p_idx], 
                    c='red', 
                    lw=self.point_size
                )
                self.content.append(line)
        for boundary in self.segmentor.boundary_list:
            if boundary.dead:
                continue
            self.content.append(Spheres(boundary.points, c='blue', r=self.point_size))
            for p_idx in range(1, len(boundary.points)):
                line = Line(
                    boundary.points[p_idx-1], 
                    boundary.points[p_idx], 
                    c='blue', 
                    lw=self.point_size
                )
                self.content.append(line)

        for c in self.content:
            self.plt.add(c)
        self.plt.render()


    def render_mesh(self, mesh: VedoMesh):
        self.plt.add(mesh)
        self.plt.render()


    def clear_content(self):
        for c in self.content:
            self.plt.remove(c)
        self.content = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='167', help='Input mesh path.')
    parser.add_argument('--smooth', action='store_true', help='Smooth the boundary.')
    parser.add_argument('--smooth-deg', type=int, default=4, help='Degree of the smooth boundary.')
    args = parser.parse_args()

    logger.info(f'Arguments: {args}')

    help_text = 'Mouse left-click to drag the view.\n' \
        'Press z to refine the segmented mesh/loop.\n' \
        'Press u to toggle the displayed segmentation mask.\n' \
        'Press h to see more help and default features.'
    logger.info(f'Keyboard shortcuts:\n{help_text}')

    msg = Text2D(pos='bottom-left', font="VictorMono", s=0.6) 
    msg.text(help_text)

    save_dir = "output_throughhole"
    if os.path.exists(save_dir):
        print("remove", save_dir)
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    ## load mesh
    shape_id = args.input
    fpath = f"./data/segmentation_data/*/{shape_id}.off"
    _, mask = visualize_psd_shape(fpath, fpath.replace(".off", "_labels.txt"))
    mesh = VedoMesh(fpath)
    
    plt = Plotter(axes=8, bg='white', size=(1200, 800))
    gui = AutoSegGUI(mesh, mask, save_dir, plt, args.smooth, args.smooth_deg)

    plt.add_callback('key press', gui.on_key_press)
    plt.add(msg)
    plt.show()

    plt.close()
    
    # v = np.array(mesh.vertices)
    # f = np.array(mesh.cells, dtype=np.int32)

    # ## 
    # v1, v2, h2, avg = trace_surface_flow(v, f)

    # ## color the mesh with mean curvature
    # face_h2 = np.mean(h2[f], axis=1)
    # rgbd = compute_colormap(face_h2)
    # print(rgbd.shape)
    # mesh.cellcolors = rgbd

    # ## draw principal flows
    # arrows_1 = Arrows(v + v1 * avg, v - v1 * avg, s=0.5).c('red')
    # arrows_2 = Arrows(v + v2 * avg, v - v2 * avg, s=0.5).c('green')

    # plt = Plotter(axes=8, bg='white', size=(1200, 800))
    # plt.add(mesh)
    # plt.add(arrows_1)
    # plt.add(arrows_2)
    # plt.show()
    # plt.close()
