
"""
We need to know if a boundary connects to a non-disk patch (i.e., annulus or other)
Then, the cuts which intersect with this boundary should be booked
Then, the booked cuts should be connected to form a path

"""

import numpy as np
from sklearn.neighbors import KDTree
from loguru import logger


class Path():
    def __init__(self):
        self.points = None
        self.id = None
        self.dead = False
    
    def set_points(self, points):
        self.points = points

    def build_kdtree(self):
        assert self.points is not None
        self.kdtree = KDTree(self.points, metric='euclidean')

    def compute_arc_length(self):
        assert self.points is not None
        arc_length = [np.linalg.norm(self.points[i+1] - self.points[i]) for i in range(len(self.points)-1)]
        arc_length = np.array(arc_length).sum()
        return arc_length
    
    def get_endpoints(self):
        return [self.points[0], self.points[-1]]
    
    def set_dead(self, dead:bool=True):
        self.dead = dead


class Cut(Path): 
    def __init__(self, points, mask_id):
        super().__init__()
        self.set_points(points)
        self.mask_id = mask_id
        
        # connected_boundary_indices and get_endpoints are one-to-one mapping.
        self.connected_boundary_indices = []

    def set_connected_boundary_indices(self, connected_boundary_indices:list):
        self.connected_boundary_indices = connected_boundary_indices

    
class Boundary(Path):
    def __init__(self, points):
        super().__init__()
        self.set_points(points)
        self.build_kdtree()
        self.connected_cut_indices = []
        self.mask_ids = set()
        self.fixed_indices = set()

    def add_mask_id(self, mask_id:int):
        assert type(mask_id) == int
        logger.debug(f'Add mask id {mask_id} to boundary {self.id}')
        self.mask_ids.add(mask_id)

    def add_connected_cut_indices(self, cut_id:int):
        assert type(cut_id) == int
        logger.debug(f'Add cut id {cut_id} to boundary {self.id}')
        self.connected_cut_indices.append(cut_id)

    def set_boundary_vertex_indices_to_mesh(self, boundary_vertex_indices):
        self.boundary_vertex_indices = boundary_vertex_indices


class PatchTopo():
    
    TYPES = ["disk", "annulus", "other"]

    def __init__(self, mask_id:int):
        self.mask_id = mask_id
        self.type = None
        self.boundary_ids = []
        self.cut_ids = []

    def set_type(self, type_str):
        assert type_str in self.TYPES
        self.type = type_str

    def extend_boundary_ids(self, boundary_ids:list):
        self.boundary_ids.extend(boundary_ids)

    def extend_cut_ids(self, cut_ids:list):
        self.cut_ids.extend(cut_ids)
