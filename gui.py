import pickle
import numpy as np
import trimesh
import json
from vedo import load, Plotter, Sphere, Arrow, Text2D, Mesh
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

    def __init__(self, output_path: str, mesh) -> None:
        
        self.picked_pts = []
        # self.picked_pts = [
        #     {'pos': np.array([ 0.27446488, -0.20893057,  0.07779584]), 'id': 703, 'name': '2_0'}, 
        #     {'pos': np.array([ 0.3607488 , -0.13651259,  0.05840396]), 'id': 706, 'name': '2_1'}, 
        #     {'pos': np.array([ 0.2932661 , -0.03708942,  0.09062152]), 'id': 712, 'name': '2_2'}, 
        #     {'pos': np.array([ 0.21838944, -0.19930978,  0.11183543]), 'id': 711, 'name': '2_3'}
        # ]

        # self.picked_pts = [
        #     {'pos': np.array([0.0953061 , 0.3834636 , 0.11388186]), 'id': 165, 'name': '1_0'}, 
        #     {'pos': np.array([0.1990558 , 0.38746288, 0.13386326]), 'id': 49, 'name': '1_1'}, 
        #     {'pos': np.array([0.17974353, 0.52697086, 0.08041261]), 'id': 87, 'name': '1_2'}, 
        #     {'pos': np.array([0.11824824, 0.52530175, 0.08713669]), 'id': 213, 'name': '1_3'}]

        # self.picked_pts = [
        #     {'pos': np.array([ 0.2231295 , -0.19414169,  0.08345791]), 'id': 701, 'name': '2_0'}, 
        #     {'pos': np.array([ 0.27446488, -0.20893057,  0.07779584]), 'id': 703, 'name': '2_1'}, 
        #     {'pos': np.array([ 0.34063372, -0.0659999 ,  0.11476872]), 'id': 707, 'name': '2_2'}, 
        #     {'pos': np.array([ 0.22675769, -0.03316575,  0.14041592]), 'id': 758, 'name': '2_3'}
        # ]
        # self.picked_pts = [
        #     {'pos': np.array([ 0.10138161,  0.21074224, -0.02015545]), 'id': 132, 'name': '0_0'}, 
        #     {'pos': np.array([0.19300433, 0.27309528, 0.01783872]), 'id': 170, 'name': '0_1'}, 
        #     {'pos': np.array([ 0.16398083,  0.414382  , -0.08792435]), 'id': 261, 'name': '0_2'}
        # ]
        self.all_picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []
        self.all_geodesic_paths = []
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
        if len(self.geodesic_paths) > 0:
            self.all_geodesic_paths.append(np.concatenate(self.geodesic_paths, 0))

        picked_pt_pos = [x['pos'] for x in self.picked_pts]
        print(picked_pt_pos)

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

        with open(self.output_path, 'wb') as f:
            pickle.dump(self.all_geodesic_paths, f)

        # print('mask_faces', self.mask_faces)
        with open('mask.json', 'w') as f:
            json.dump(self.mask_faces, f, cls=NpEncoder, ensure_ascii=False, indent=4)

        self.picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []


msg = Text2D(pos='bottom-left', font="VictorMono") 
msg.text(
    'Mouse left-click to pick vertex.\n' \
    'Press f/g to compute Geodesic path/loop.\n' \
    'Press c to clear the points.'
)


color_img = np.asarray(Image.open('assets/cm_tab20.png').convert('RGBA'))
cmap = []
print(color_img.shape)
for i in range(20):
    c = 25 * (i % 20) + 10
    cmap.append(color_img[20, c])

# Load the OBJ file
mesh = load('data/manohand_0.obj')
# mesh.cellcolors = [0, 0, 200]
output_path = 'geodesic_paths.pt'
gui = GUI(output_path, mesh)

plt = Plotter(axes=8, bg='white')

plt.add_callback('left click', gui.on_mouse_click)
plt.add_callback('key press', gui.on_key_press)

plt.show(mesh, msg)
plt.close()
