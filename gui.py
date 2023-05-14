import pickle
import numpy as np
import trimesh
from vedo import load, Plotter, Sphere, Arrow, Text2D, Mesh
from potpourri3d import EdgeFlipGeodesicSolver
from mesh_tools import split_mesh


class GUI:

    def __init__(self, output_path: str, mesh) -> None:
        
        self.picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []
        self.all_geodesic_paths = []
        self.seg_n = 0

        self.output_path = output_path
        self.mesh = mesh
        self.tri_mesh = trimesh.Trimesh(
            mesh.vertices(), 
            mesh.faces(),
            process=False,
            maintain_order=True
        )

    def on_mouse_click(self, event):
        mesh = event.actor
        if not mesh:
            return
        
        mouse_pt = event.picked3d
        pid = mesh.closest_point(mouse_pt, return_point_id=True)
        pt = mesh.vertices()[pid]

        picked_pt = Sphere(pt, r=0.005, c='b')
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
            self.compute_geodesic_path(event.keypress == 'g')
        elif event.keypress == 'c':
            self.clear_pts()
        
    def compute_geodesic_path(self, loop_flag: bool = False):
        print('Compute the Geodesic path. Loop:', loop_flag)
        print(self.picked_pts)

        v = self.mesh.vertices()
        f = np.array(self.mesh.faces())
        print('shapes', len(v), f.shape)
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
                arrow = Arrow(path_pts[path_id - 1], path_pts[path_id], s=0.0005, c='green')
                arrow.name = f'{self.seg_n}_{len(self.arrow_names)}'
                plt.add(arrow)
                self.arrow_names.append(arrow.name)

        # seg_path = np.concatenate(self.geodesic_paths, axis=0)
        new_mesh = split_mesh(self.tri_mesh, np.array(new_pts))
        plt.remove(self.mesh)

        self.tri_mesh = new_mesh
        self.mesh = Mesh([new_mesh.vertices.tolist(), new_mesh.faces.tolist()])
        plt.show(self.mesh)

        # self.save()
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

        self.all_geodesic_paths.append(self.geodesic_paths)
        # print(self.all_geodesic_paths)

        with open(self.output_path, 'wb') as f:
            pickle.dump(self.all_geodesic_paths, f)

        self.picked_pts = []
        self.arrow_names = []
        self.geodesic_paths = []


msg = Text2D(pos='bottom-left', font="VictorMono") 
msg.text(
    'Mouse left-click to pick vertex.\n' \
    'Press f/g to compute Geodesic path/loop.\n' \
    'Press s to save all Geodesic paths.\n' \
    'Press z to start the next segmentation.\n' \
    'Press c to clear the points.'
)

# Load the OBJ file
mesh = load('data/manohand_0.obj')
mesh.phong()
output_path = 'geodesic_paths.pt'
gui = GUI(output_path, mesh)

plt = Plotter(axes=1, bg='white')

plt.add_callback('left click', gui.on_mouse_click)
plt.add_callback('key press', gui.on_key_press)

plt.show(mesh, msg)
plt.close()
