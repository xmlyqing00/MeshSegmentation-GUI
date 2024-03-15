import json
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import time, datetime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def read_json(fname):
    with open(fname) as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, 'w') as handle:
        json.dump(content, handle, indent=4, sort_keys=False, cls=NpEncoder)


def read_obj_file(filename):
    ## read obj file
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    vertices = []
    faces = []
    for l in lines:
        ## vertices
        if len(l) > 0 and l[0] == 'v':
            vertices.append([float(l[1]), float(l[2]), float(l[3])])
        ## faces
        elif len(l) > 0 and l[0] == 'f':
            faces.append([int(l[1]), int(l[2]), int(l[3])])
    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    return vertices, faces

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

def write_xyz_file(filename, V, N=None):
    with open(filename, 'w') as f:
        if N is not None:
            for Vi, Ni in zip(V, N):
                f.write(f"{Vi[0]} {Vi[1]} {Vi[2]} {Ni[0]} {Ni[1]} {Ni[2]}\n")
        else:
            for Vi in V:
                f.write(f"{Vi[0]} {Vi[1]} {Vi[2]}\n")


def write_obj_file_poly(filename, V, F=None, C=None, N=None, vid_start=1):
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
                  
        for Li in F:
            line = "f "
            for i in Li:
                line = f"{line}{i+vid_start} "
            f.write(line+'\n')


def read_m_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
        vertices = []
    faces = []
    for l in lines:
        ## vertices
        if len(l) > 0 and l[0] == 'Vertex':
            vertices.append([float(l[2]), float(l[3]), float(l[4])])
        ## faces
        elif len(l) > 0 and l[0] == 'Face':
            faces.append([int(l[2]), int(l[3]), int(l[4])])
    vertices = np.array(vertices)
    faces = np.array(faces) - 1
    return vertices, faces


def read_line_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.split() for l in lines]
    vertices = []
    edges = []
    max_length_edge = 0
    for l in lines:
        ## vertices
        if len(l) > 0 and l[0] == 'v':
            vertices.append([float(l[1]), float(l[2]), float(l[3])])
        ## faces
        elif len(l) > 0 and l[0] == 'l':
            if len(l) - 1 > max_length_edge:
                max_length_edge = len(l)-1
            edge_pts = [int(id) for id in l[1:]]
            edges.append(edge_pts)
    vertices = np.array(vertices)
    # np_edges = np.zeros((len(edges), max_length_edge))
    # for i in range(len(edges)):
    #     np_edges[i, :len(edges[i])] = edges[i]
    return vertices, edges


def write_line_file(filename, V, L, vid_start=1, closed=True):
    with open(filename, 'w') as f:
        for Vi in V:
            f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        for Li in L:
            if closed:
                for i in range(len(Li)):
                    f.write(f"l {Li[i]+vid_start} {Li[(i+1)%len(Li)]+vid_start}\n")
            else:
                for i in range(len(Li)-1):
                    f.write(f"l {Li[i]+vid_start} {Li[(i+1)]+vid_start}\n")

## can draw line segments
def write_line_file2(save_to, V, L, vid_start=1):
    with open(save_to, 'w') as f:
        for Vi in V:
            f.write(f"v {Vi[0]} {Vi[1]} {Vi[2]}\n")
        # f.write('s off\n')
        for Li in L:
            line = "l "
            for i in Li:
                line = f"{line}{i+vid_start} "
            f.write(line+'\n')

def draw_colored_points_to_obj(
    filename, vertices, scalars_for_color, colormap="jet", faces=None, in_vmax=None, in_vmin=None):
    # print(f"draw colored points to obj file :{filename}\n")
    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3
    if colormap == "set":
        if in_vmax is not None:
            vmax = in_vmax
        else:
            vmax = 20
        if in_vmin is not None:
            vmin = in_vmin
        else:
            vmin = 0
        norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)
    else:
        if in_vmax is not None:
            vmax = in_vmax
        else:
            vmax = scalars_for_color.mean()*3
        if in_vmin is not None:
            vmin = in_vmin
        else:
            vmin = 0.0
        norm = matplotlib.colors.Normalize(vmin, vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = [(r, g, b) for r, g, b, a in mapper.to_rgba(scalars_for_color)]
    write_obj_file(filename, vertices.reshape(-1, 3), F=faces, C=colors)

def find_nan(xtensor, name):
    if torch.isnan(xtensor.detach()).any():
        print(f"{name} is nan")
        print(xtensor)
        assert False

def find_nan_np(xarray, name):
    if np.isnan(xarray).any():
        print(f"{name} is nan")
        print(xarray)
        assert False

def to_numpy(a_tensor):
    if a_tensor.device != "cpu":
        return a_tensor.detach().cpu().numpy()
    else:
        return a_tensor.numpy()


def get_timestamp():
    # ct stores current time
    ct = datetime.datetime.now()
    print("current time:-", ct)
    
    # ts store timestamp of current time
    ts = ct.timestamp()
    return ts