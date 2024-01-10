
import trimesh
import numpy as np
from mesh_data_structure.utils import GeoPathSolverWrapper, get_open_boundary
from scipy.spatial.distance import cdist
from vedo import utils, show, Plotter, Arrows, Sphere, Spheres, Text2D, Line


def vis_path(path, plt, c):
    for i in range(len(path)-1):
        plt.add(Line(path[i], path[i+1], c=c))
    plt.render()


mesh = trimesh.load('tmp/mask_3.obj')
path_solver = GeoPathSolverWrapper(mesh)
boundary_loops = get_open_boundary(mesh)

assert len(boundary_loops) == 2, "Num of boundary_loops in annulus should be 2"

vedo_mesh = utils.trimesh2vedo(mesh)
plt = Plotter(bg='white', size=(1200, 800))
plt.add(vedo_mesh)
        
# find the farthest point on the boundary
bidx, bidx_c = 0, 1
vids = np.array(boundary_loops[bidx])[:, 0]
v = mesh.vertices[vids]
pair_dist = cdist(v, v)
a0, a1 = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)
a0 = vids[a0]
a1 = vids[a1]

a0_boundary_dist = np.inf
a1_boundary_dist = np.inf
a0_boundary_path = None
a1_boundary_path = None
a0_boundary_p = None
a1_boundary_p = None

vids = np.array(boundary_loops[bidx_c])[:, 0]
for p in vids:

    path = path_solver.solve(a0, p)
    dist = [np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]
    dist = np.array(dist).sum()

    vis_path(path, plt, 'r1')
    if dist < a0_boundary_dist:
        a0_boundary_dist = dist
        a0_boundary_path = path
        a0_boundary_p = p

    path = path_solver.solve(a1, p)
    dist = [np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)]
    dist = np.array(dist).sum()

    vis_path(path, plt, 'g1')
    if dist < a1_boundary_dist:
        a1_boundary_dist = dist
        a1_boundary_path = path
        a1_boundary_p = p

vis_path(a0_boundary_path, plt, 'r8')
vis_path(a1_boundary_path, plt, 'g8')

plt.show()
plt.close()