from mesh_data_structure.halfedge_mesh import HETriMesh
import trimesh
from shutil import copyfile
import numpy as np
import networkx as nx
from PIL import Image

def write_obj_file(filename, V, F=None, C=None, N=None, vid_start=1):
    with open(filename, 'w') as f:
        if C is not None:
            for Vi, Ci in zip(V, C):
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]} {Ci[0]} {Ci[1]} {Ci[2]}\n")
        else:
            for Vi in V:
                f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        
        if N is not None:
            for Ni in N:
                f.write(f"vn {Ni[0]} {Ni[1]} {Ni[2]}\n")
                  
        if F is not None:
            for Fi in F:
                f.write(f"f {Fi[0]+vid_start} {Fi[1]+vid_start} {Fi[2]+vid_start}\n")

def read_textured_obj_file(filename):
    ## read obj file
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    vertices = []
    uvcoords = []
    faces = []
    tfaces = []
    for l in lines:
        ## vertices
        if len(l) > 0 and l[0] == 'v':
            vertices.append([float(l[1]), float(l[2]), float(l[3])])
        if len(l) > 0 and l[0] == 'vt':
            uvcoords.append([float(l[1]), float(l[2])])
        ## faces
        elif len(l) > 0 and l[0] == 'f':
            faces.append([int(l[1].split('/')[0]), int(l[2].split('/')[0]), int(l[3].split('/')[0])])
            tfaces.append([int(l[1].split('/')[1]), int(l[2].split('/')[1]), int(l[3].split('/')[1])])
    vertices = np.array(vertices)
    uvcoords = np.array(uvcoords)
    faces = np.array(faces) - 1
    tfaces = np.array(tfaces) - 1
    return vertices, uvcoords, faces, tfaces

class LlyodRelax():

    def __init__(self, mesh, num_iter=10):
        self.num_iter = num_iter

        ## data
        self.vertices = mesh.vertices ## to be updated
        self.vertices = np.concatenate([self.vertices, np.zeros((1, 3))], axis=0) ## last one is dummy
        
        ## one ring neighbors
        g = nx.from_edgelist(mesh.edges_unique)

        # one_ring = [list(g[i].keys()) for i in range(len(mesh.vertices))]
        one_ring = np.zeros((len(self.vertices), 50), dtype=np.int32) -1  ## last one (-1) is dummy
        max_val = 0
        for i in range(len(mesh.vertices)):
            one_ring[i, :len(list(g[i].keys()))] = list(g[i].keys())
            max_val = max(len(list(g[i].keys())), max_val)

        self.one_ring = one_ring[:, :max_val]
        self.one_ring_count = np.sum(self.one_ring != -1, axis=-1) ## last one (-1) is dummy
        self.one_ring_count[-1] = 1 ## last one (-1) is dummy

        ## boundary vertices
        unique_edges = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
        bvids = np.unique(unique_edges)
        self.constraint_ids = bvids

    def set_vertices(self, vertices):

        if vertices.shape[0] != self.vertices.shape[0] - 1:
            print(vertices.shape, self.vertices.shape)
            raise ValueError("vertices shape not match")
        
        self.vertices = vertices
        self.vertices = np.concatenate([self.vertices, np.zeros((1, 3))], axis=0)

    def get_vertices(self):
        return self.vertices[:-1]
    
    def _lloyd_relax(self):
        old_vertices = self.vertices
        new_vertices = np.zeros_like(old_vertices)
        new_vertices = old_vertices[self.one_ring].sum(axis=1) 
        new_vertices /= self.one_ring_count[:, None]
        new_vertices[self.constraint_ids] = old_vertices[self.constraint_ids]
        # displacement = np.linalg.norm(new_vertices - old_vertices, axis=-1)
        self.vertices = new_vertices
        return new_vertices[:-1]
    
    def run(self, num_iters=None):
        if num_iters is None:
            num_iters = self.num_iter
        for i in range(num_iters):
            self._lloyd_relax()


folder = "data_built/bimba_fix4_new_4/parameterization"


for pid in range(12):
    meshfile = f'{folder}/mesh_uv_{pid}.obj'
    vertices, uvcoords, faces, tfaces = read_textured_obj_file(meshfile)
    mesh = HETriMesh()
    mesh.init_mesh(vertices, faces)

    ## get vertices near the boundary (3-rings)
    vertex_list = mesh.boundary_vertices()
    for i in range(3):
        ## get 1-ring neighbors of vertices in vertex_set
        neighbor_set = set()
        for v in vertex_list:
            neighbor_set.update(mesh.vertex_vertex_neighbors(v))
        vertex_list = list(neighbor_set)
    face_set = set()
    for v in vertex_list:
        face_set.update(mesh.vertex_face_neighbors(v))
    face_list = list(face_set)
    faces = mesh.faces[face_list]
    vertex_indices = np.unique(faces)

    ## replace vertices in faces with new indices
    new_faces = np.zeros_like(faces) - 1
    for i, v in enumerate(vertex_indices):
        new_faces[faces == v] = i
    print("new_faces", new_faces.min(), new_faces.max())

    ## make a boundary mesh for smoothing
    new_vertices = mesh.vs[vertex_indices]
    boundary_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False, maintain_order=True)
    smoother = LlyodRelax(boundary_mesh)
    smoother.run()
    updated_vertices = smoother.get_vertices()

    ## update the original mesh with the smoothed boundary vertices
    texture_img = Image.open(f'./assets/uv_color.png')
    all_vertices = mesh.vs
    all_vertices[vertex_indices] = updated_vertices
    outmesh = trimesh.Trimesh(vertices=all_vertices, faces=mesh.faces, process=False, maintain_order=True)
    outmesh.visual = trimesh.visual.TextureVisuals(uv=uvcoords, material=None, image=texture_img)
    outmesh.export(f'{folder}/mesh_uv_{pid}.obj')