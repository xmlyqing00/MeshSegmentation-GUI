from typing import Any
import trimesh
import numpy as np
from potpourri3d import EdgeFlipGeodesicSolver
from tsp_solver.greedy import solve_tsp
from view_psd_data import *
import shutil
import argparse

from loguru import logger
from mesh_data_structure.halfedge_mesh import HETriMesh
from mesh_data_structure.utils import trace_boundary_edges, close_holes
import networkx as nx
from sklearn.neighbors import KDTree

from src.mesh_tools import split_mesh, split_mesh_by_path, floodfill_label_mesh

def convert_list_to_string(l:list):
    return ",".join([str(i) for i in l])


def compute_distance_matrix(mesh, b_close_holes=True):
    ## close the holes and compute the centroid of each hole
    if b_close_holes:
        mesh_holefilled, cids = mesh_close_holes(mesh)
        path_solver = GeoPathSolverWrapper(mesh_holefilled)
        paths, distance_matrix = path_solver.solve_all2all(cids, return_distance_matrix=True)
    ## compute the shortest distance between each pair of holes
    else:
        path_solver = GeoPathSolverWrapper(mesh)
        boundary_loops = get_open_boundary(mesh)
        distance_matrix = np.zeros((len(boundary_loops), len(boundary_loops)))
        paths = {}
        for i in range(len(boundary_loops)-1):
            for j in range(i+1, len(boundary_loops)):
                b0 = np.array(boundary_loops[i], dtype=np.int32)[:,0]
                b0 = np.hstack([b0, boundary_loops[i][-1][1]])
                b1 = np.array(boundary_loops[j], dtype=np.int32)[:,0]
                b1 = np.hstack([b1, boundary_loops[j][-1][1]])
                path_pts, distance = path_solver.solve_path2path(b0, b1)
                paths[f"{i},{j}"] = path_pts
                ## 
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    return paths, distance_matrix 

class GeoPathSolverWrapper():
    def __init__(self, mesh : trimesh.Trimesh) -> None:
        v = mesh.vertices
        f = np.array(mesh.faces, dtype=np.int32)
        self.solver = EdgeFlipGeodesicSolver(v, f) # shares precomputation for repeated solves

    def solve_all2all(self, vids, return_distance_matrix=False):
        paths = {}
        distance_matrix = np.zeros((len(vids), len(vids)))
        for i in range(len(vids)):
            for j in range(i+1, len(vids)):
                v_start = vids[i]
                v_end = vids[j]
                path_pts = self.solve(v_start, v_end)
                paths[f"{i},{j}"] = path_pts
                if return_distance_matrix:
                    distance = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
                    distance = np.array(distance).sum()
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        if return_distance_matrix:
            return paths, distance_matrix
        else:
            return paths

    def solve(self, v_start, v_end):
        path_pts = self.solver.find_geodesic_path(v_start, v_end)
        return path_pts
    
    def solve_vlist(self, vlist):
        return self.solver.find_geodesic_path_poly(vlist)
    
    def solve_path2path(self, vlist0, vlist1):
        min_distance = 1e10
        min_pair = None
        for i in range(len(vlist0)):
            for j in range(len(vlist1)):
                path_pts = self.solve(vlist0[i], vlist1[j])                
                distance = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
                distance = np.array(distance).sum()
                if distance < min_distance:
                    min_distance = distance
                    min_pair = [vlist0[i], vlist1[j]]
        path_pts = self.solve(min_pair[0], min_pair[1])
        return path_pts, min_distance


def get_open_boundary(mesh : trimesh.Trimesh):
    he_mesh = HETriMesh()
    he_mesh.init_mesh(mesh.vertices, mesh.faces)
    boundary_loops = trace_boundary_edges(he_mesh)
    return boundary_loops


def mesh_close_holes(mesh : trimesh.Trimesh):
    he_mesh = HETriMesh()
    he_mesh.init_mesh(mesh.vertices, mesh.faces)
    boundary_loops = trace_boundary_edges(he_mesh)
    ## close the hole
    he_mesh, centroid_ids = close_holes(he_mesh, boundary_loops)
    mesh = trimesh.Trimesh(he_mesh.vs, he_mesh.faces, process=False, maintain_order=True)
    return mesh, centroid_ids


def tsp_segment(mesh : trimesh.Trimesh):
    
    patch_mesh_holefilled, centroid_ids = mesh_close_holes(mesh)

    ## GeoPathSolverWrapper
    path_solver = GeoPathSolverWrapper(patch_mesh_holefilled)
    paths, distance_matrix = path_solver.solve_all2all(centroid_ids, return_distance_matrix=True)    
    ## solve TSP to find the order of throughhole
    tsp_path = solve_tsp(distance_matrix)
    ## find throughhole path
    throughhole_path = []
    for k in range(len(tsp_path)):
        i = tsp_path[k]
        j = tsp_path[(k+1)%len(tsp_path)]
        keys = [f"{i},{j}", f"{j},{i}"]
        for key in keys:
            if key in paths:
                throughhole_path.append(paths[key][1:-1])
                break
    return throughhole_path


def simple_floodfill_label_mesh(
    mesh: trimesh.Trimesh, 
    mask: list
):  
    mask = np.array(mask)
    mask_connected = []
    patch_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[mask,:], maintain_order=True, process=False)
    out = trimesh.graph.connected_component_labels(patch_mesh.face_adjacency)
    for i in range(out.max()+1):
        mask_connected.append((mask[out==i]).tolist())
    return mask_connected


"""
We need to know if a boundary connects to a non-disk patch (i.e., annulus or other)
Then, the cuts which intersect with this boundary should be booked
Then, the booked cuts should be connected to form a path

"""

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

    def add_mask_id(self, mask_id:int):
        assert type(mask_id) == int
        self.mask_ids.add(mask_id)

    def add_connected_cut_indices(self, cut_id:int):
        assert type(cut_id) == int
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

class MeshSegmentator():

    def __init__(
            self, 
            mesh: trimesh.Trimesh,
            mask: list,
            save_dir:str = None
            ) -> None:
        
        self.save_dir = save_dir
        
        self.mesh = mesh

        ## proximity
        self.pq_mesh = trimesh.proximity.ProximityQuery(self.mesh)

        self.mask = mask
        self.b_close_holes = False

        self.patch_topo_list = [] ## list of PatchTopo class objects
        self.boundary_list = [] ## list of Boundary class objects
        self.cut_list = [] ## list of Cut class objects

        logger.success(f"mesh info: num vertices: {self.mesh.vertices.shape}; num faces: {self.mesh.faces.shape}; num masks: len(mask)")

    def save_paths(self):
        for cut in self.cut_list:
            if cut.dead:
                continue
            trimesh.PointCloud(cut.points).export(os.path.join(self.save_dir,f"cut_{cut.id}.obj"))
        for boundary in self.boundary_list:
            if boundary.dead:
                continue
            trimesh.PointCloud(boundary.points).export(os.path.join(self.save_dir,f"boundary_{boundary.id}.obj"))
    
    def save_mesh(self):
        self.mesh.export(os.path.join(self.save_dir,"mesh_new.obj"))


    def query_closeest_boundary(self, queries):
        queries = np.array(queries)
        min_distances = np.ones((queries.shape[0],)) * 1e10
        min_boundary_indices = np.ones((queries.shape[0],)) * -1
        for i, b in enumerate(self.boundary_list):
            distances, _ = b.kdtree.query(queries, k=1, return_distance=True)
            distances = distances[:,0]
            # print(distances, "   ", i)
            # print(distances.shape, min_distances.shape)
            min_boundary_indices[distances < min_distances] = i
            min_distances[distances < min_distances] = distances[distances < min_distances]
            # print(min_distances)            
            # print(min_boundary_indices)
            # input()
        return np.array(min_boundary_indices, dtype=np.int32).tolist()


    def build_mask_structure(self, mask_id):
        m = np.array(self.mask[mask_id])
        he_mesh = HETriMesh()
        he_mesh.init_mesh(self.mesh.vertices, self.mesh.faces[m,:])
        tmp_boundaries = trace_boundary_edges(he_mesh)

        patch_topo = PatchTopo(mask_id)

        ## classify the patch type and cut
        if len(tmp_boundaries) == 1:
            logger.success(f"disk topology --- {mask_id}")
            patch_topo.set_type('disk')
        elif len(tmp_boundaries) == 2:
            logger.success(f"annulus topology --- {mask_id}")
            patch_topo.set_type('annulus')
        elif len(tmp_boundaries) > 2:
            logger.success(f"non-disk non-annulus topology --- {mask_id}")
            patch_topo.set_type('other')
        else:
            logger.error("Error: no boundary found")

        
        for i in range(len(tmp_boundaries)):
            boundary = np.array(tmp_boundaries[i], dtype=np.int32)[:,0]
            boundary = np.hstack([boundary, tmp_boundaries[i][-1][1]])
            ## check if this boundary has been found before
            b_found = -1
            for j in range(len(self.boundary_list)):
                if set(boundary) == set(self.boundary_list[j].boundary_vertex_indices):
                    b_found = self.boundary_list[j].id
                    self.boundary_list[j].add_mask_id(mask_id)
                    break
            if b_found == -1:
                boundary_obj = Boundary(self.mesh.vertices[boundary])
                boundary_obj.set_boundary_vertex_indices_to_mesh(boundary)
                boundary_obj.add_mask_id(mask_id)
                boundary_obj.id = len(self.boundary_list)
                self.boundary_list.append(boundary_obj)
                patch_topo.extend_boundary_ids([boundary_obj.id])
            else:
                patch_topo.extend_boundary_ids([b_found])
        
        self.patch_topo_list.append(patch_topo)
        return patch_topo.type


    def cut_through_holes(self, mask_id):
        m = np.array(self.mask[mask_id], dtype=np.int32)
        patch_mesh = trimesh.Trimesh(self.mesh.vertices, self.mesh.faces[m,:])
        paths, distance_matrix = compute_distance_matrix(patch_mesh, b_close_holes=False)
        print(distance_matrix)
        ## solve TSP
        tsp_path = solve_tsp(distance_matrix)
        print(tsp_path)

        throughhole_paths = []
        for k in range(len(tsp_path)):
            i = tsp_path[k]
            j = tsp_path[(k+1)%len(tsp_path)]
            keys = [f"{i},{j}", f"{j},{i}"]
            for key in keys:
                if key in paths:
                    throughhole_paths.append(paths[key])
                    break
        return throughhole_paths
    

    def cut_mask(self, mask_id):
        logger.success("cut mask")
        cuts = self.cut_through_holes(mask_id)

        cut_obj_list = []
        for cut_pts in cuts:
            cut_obj = Cut(cut_pts, mask_id)
            cut_obj_list.append(cut_obj)

            cut_obj.id = len(self.cut_list)
            self.cut_list.append(cut_obj)

            ## track cuts
            self.patch_topo_list[mask_id].extend_cut_ids([cut_obj.id])

        ## find the boundary that connects to the cuts
        for cut_obj in cut_obj_list:
            queries = cut_obj.get_endpoints()
            connected_boundary_indices = self.query_closeest_boundary(queries)
            cut_obj.set_connected_boundary_indices(connected_boundary_indices)
            for boundary_id in connected_boundary_indices:
                self.boundary_list[boundary_id].add_connected_cut_indices(cut_obj.id)
        

    def align_cuts(self, boundary_obj:Boundary):
        cut_ids = boundary_obj.connected_cut_indices
        logger.success(f"need align; cut ids: {cut_ids}")

        cut_list = [self.cut_list[i] for i in cut_ids]
    
        extended_cut_dict = {}
        for i in range(len(cut_list)-1):
            for j in range(i+1, len(cut_list)):
                cut0 = cut_list[i]
                cut1 = cut_list[j]
                if cut0.mask_id == cut1.mask_id:
                   continue
                pair_cuts = [cut0.id, cut1.id] if cut0.id < cut1.id else [cut1.id, cut0.id]
                keystr = convert_list_to_string(pair_cuts)
                if keystr not in extended_cut_dict:
                    extended_cut_dict[keystr] = {}
                    extended_cut_dict[keystr]['cutpair'] = pair_cuts
                    extended_cut_dict[keystr]['distance'] = 1e10
                    extended_cut_dict[keystr]['points'] = []
                    extended_cut_dict[keystr]['boundary'] = set(cut0.connected_boundary_indices).union(set(cut1.connected_boundary_indices))
                    extended_cut_dict[keystr]['boundary'].difference_update(set([boundary_obj.id]))



        ## find the shortest path between the two cuts
        path_solver = GeoPathSolverWrapper(self.mesh)
        for k, v in extended_cut_dict.items():
            print()
            print("cutpair", k)
            points = []
            cut0 = self.cut_list[v['cutpair'][0]]
            cut1 = self.cut_list[v['cutpair'][1]]

            queries = [cut0.points[0], cut0.points[-1]]
            d, _ = boundary_obj.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] < d[1]:
                ## flip
                points.append(cut0.points[-1]) ## far
                points.append(cut0.points[0]) ## near
            else:
                ## not flip
                points.append(cut0.points[0])
                points.append(cut0.points[-1])

            queries = [cut1.points[0], cut1.points[-1]]
            d, _ = boundary_obj.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] > d[1]:
                ## flip
                points.append(cut1.points[-1]) ## near
                points.append(cut1.points[0]) ## far
            else:
                ## not flip
                points.append(cut1.points[0])
                points.append(cut1.points[-1])
            
            d, vertex_ids = self.pq_mesh.vertex(np.array(points))

            path_pts = path_solver.solve_vlist(vertex_ids)
            dist = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
            dist = np.array(dist).sum()
            v['distance'] = dist
            v['points'] = path_pts
            # print(vertex_ids, dist)
            # trimesh.PointCloud(path_pts).export(os.path.join(self.save_dir,f"align_cuts_{k}.obj"))

        print(extended_cut_dict)
        ## sort extended_cut_dict by distance key
        extended_cut_dict = sorted(extended_cut_dict.items(), key=lambda x: x[1]['distance'])
        print(extended_cut_dict)

        ## keep the shortest half of cuts
        cnt = 0
        max_keep = len(extended_cut_dict)//2
        for k, v in extended_cut_dict:
            cut = Cut(v['points'], -1)
            cut.set_connected_boundary_indices(list(v['boundary']))
            cut.id = len(self.cut_list)
            self.cut_list.append(cut)

            for cut_id in v['cutpair']:
                self.cut_list[cut_id].set_dead()

            cnt += 1
            if cnt >= max_keep:
                break
        

    def cut_annulus_aligned(self, patch_topo:PatchTopo):
        logger.success("cut annulus aligned")

        ## find the two boundaries
        boundary_ids = patch_topo.boundary_ids
        assert len(boundary_ids) == 2, "len of boundary_ids should be 2"
        boundary0 = self.boundary_list[boundary_ids[0]]
        boundary1 = self.boundary_list[boundary_ids[1]]
        print("boundary0", boundary0.id)
        print("boundary1", boundary1.id)

        ## find the two cuts on each boundary
        cuts_on_boundary0 = []
        for cut_id in boundary0.connected_cut_indices:
            cuts_on_boundary0.append(self.cut_list[cut_id])
            print("cut_id on b0", cut_id)
        cuts_on_boundary1 = []
        for cut_id in boundary1.connected_cut_indices:
            cuts_on_boundary1.append(self.cut_list[cut_id])
            print("cut_id on b1", cut_id)

        ## find the extended cuts and smooth them
        extended_cut_dict = {}
        for cut0 in cuts_on_boundary0:
            for cut1 in cuts_on_boundary1:
                pair_cuts = [cut0.id, cut1.id] if cut0.id < cut1.id else [cut1.id, cut0.id]
                keystr = convert_list_to_string(pair_cuts)
                if keystr not in extended_cut_dict:
                    extended_cut_dict[keystr] = {}
                    extended_cut_dict[keystr]['cutpair'] = pair_cuts
                    extended_cut_dict[keystr]['distance'] = 1e10
                    extended_cut_dict[keystr]['points'] = []
        
        ## find the shortest path between the two cuts
        path_solver = GeoPathSolverWrapper(self.mesh)
        for k, v in extended_cut_dict.items():
            print()
            print("cutpair", k)
            points = []
            cut0 = self.cut_list[v['cutpair'][0]]
            cut1 = self.cut_list[v['cutpair'][1]]

            queries = [cut0.points[0], cut0.points[-1]]
            d, _ = boundary0.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] < d[1]:
                ## flip
                points.append(cut0.points[-1]) ## far
                points.append(cut0.points[0]) ## near
            else:
                ## not flip
                points.append(cut0.points[0])
                points.append(cut0.points[-1])

            queries = [cut1.points[0], cut1.points[-1]]
            d, _ = boundary1.kdtree.query(queries, k=1, return_distance=True)
            d = d[:,0]
            if d[0] > d[1]:
                ## flip
                points.append(cut1.points[-1]) ## near
                points.append(cut1.points[0]) ## far
            else:
                ## not flip
                points.append(cut1.points[0])
                points.append(cut1.points[-1])
            
            d, vertex_ids = self.pq_mesh.vertex(np.array(points))

            path_pts = path_solver.solve_vlist(vertex_ids)
            dist = [np.linalg.norm(path_pts[i+1] - path_pts[i]) for i in range(len(path_pts)-1)]
            dist = np.array(dist).sum()
            v['distance'] = dist
            v['points'] = path_pts
            # print(vertex_ids, dist)
            # trimesh.PointCloud(path_pts).export(os.path.join(self.save_dir,f"align_cuts_{k}.obj"))

        ## sort extended_cut_dict by distance key
        extended_cut_dict = sorted(extended_cut_dict.items(), key=lambda x: x[1]['distance'])

        ## keep the shortest half of cuts
        cnt = 0
        max_keep = len(extended_cut_dict)//2
        for k, v in extended_cut_dict:
            cut = Cut(v['points'], -1)
            cut.id = len(self.cut_list)
            self.cut_list.append(cut)

            for cut_id in v['cutpair']:
                self.cut_list[cut_id].set_dead()

            cnt += 1
            if cnt >= max_keep:
                break


    def split_mesh_with_cuts(self):
        for cut in self.cut_list:
            if cut.dead:
                continue
            new_mesh, new_mask = split_mesh(self.mesh, cut.points, self.mesh.faces)
            self.mesh = new_mesh
            self.mask = new_mask
            print("add a cut")



    def __call__(self, b_close_holes:bool = None):
        if b_close_holes is not None:
            self.b_close_holes = b_close_holes
        
        ## floodfild the mask
        mask_connected = []
        for i in range(len(self.mask)):
            m = np.array(self.mask[i])
            mask_connected.extend(simple_floodfill_label_mesh(self.mesh, m))
        self.mask = mask_connected

        for i in range(len(self.mask)):
            mask_type = self.build_mask_structure(i)
            if mask_type == 'other':
                self.cut_mask(i)

        logger.success("align cuts between two non-disk, non-annulus patches")
        for boundary_obj in self.boundary_list:
            # print()
            # print("boundary id: ", boundary_obj.id)
            # print("boundary mask ids: ")
            mask_ids = list(boundary_obj.mask_ids)
            if self.patch_topo_list[mask_ids[0]].type == 'other' and self.patch_topo_list[mask_ids[1]].type == 'other':
                self.align_cuts(boundary_obj)

        logger.success("cut annulus patches")
        for patch in self.patch_topo_list:
            if patch.type == "annulus":
                self.cut_annulus_aligned(patch)


        ## 
        self.split_mesh_with_cuts()

        ## 
        pq_mesh = trimesh.proximity.ProximityQuery(self.mesh)
        picked_pt_ids = []
        for cut in self.cut_list:
            if cut.dead:
                continue
            _, vids = pq_mesh.vertex(cut.points)
            picked_pt_ids.append(vids.tolist())
        for boundary in self.boundary_list:
            _, vids = pq_mesh.vertex(boundary.points)
            picked_pt_ids.append(vids.tolist())
        ## convert cut edges to mesh edges
        self.mask = floodfill_label_mesh(self.mesh, set(), picked_pt_ids)



if __name__ == "__main__":

    parser = argparse.ArgumentParser('Segmentation GUI')
    parser.add_argument('--input', type=str, default='167', help='Input mesh path.')
    args = parser.parse_args()

    save_dir = "output_throughhole"
    if os.path.exists(save_dir):
        print("remove", save_dir)
        shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    ## load mesh
    shape_id = args.input
    fpath = f"./data/segmentation_data/*/{shape_id}.off"
    mesh, mask = visualize_psd_shape(fpath, fpath.replace(".off", "_labels.txt"))
    mesh.export(os.path.join(save_dir,"mesh.obj"))


    segmentor = MeshSegmentator(mesh, mask, save_dir)
    segmentor(b_close_holes=False)
    segmentor.save_paths()
    segmentor.save_mesh()

    


    # list_boundaries, non_disk_mask, annulus_mask = process_data(
    #     mesh, mask, merge_annulus=False)
    # # annulus_cuts = cut_annulus(mesh, annulus_mask)
    # throughhole_paths = cut_through_holes(mesh, non_disk_mask)

    # cnt = 0
    # for i, boundaries in enumerate(list_boundaries):
    #     # print(i, len(boundaries))
    #     for boundary in boundaries:
    #         boundary = np.vstack(boundary)
    #         # print("boundary", boundary[:,0])
    #         bverts = mesh.vertices[boundary[:,0]]
    #         trimesh.PointCloud(bverts).export(os.path.join(save_dir,f"list_boundaries_{cnt}.obj"))
    #         cnt += 1
    
    # for i in range(len(throughhole_paths)):
    #     path = np.vstack(throughhole_paths[i])
    #     trimesh.PointCloud(path).export(os.path.join(save_dir,f"throughhole_path_{i}.obj"))
    
    # # for i in range(len(annulus_cuts)):
    # #     path = np.vstack(annulus_cuts[i])
    # #     trimesh.PointCloud(path).export(os.path.join(save_dir,f"annulus_cuts_{i}.obj"))