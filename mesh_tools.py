import numpy as np
import trimesh
from vedo import Mesh
import queue
import networkx as nx


"""
vert_edges: [N, 2, 3]
queries: [M, 3]
"""
def find_intersection_pairs(queries, edge_verts):
    v = queries[:, None, :] - edge_verts[None, :, 0, :] ## [M, N, 3]
    v_norm = np.linalg.norm(v, axis=-1) 
    ev = v / v_norm[...,None] ## [M, N, 3]
    # x = np.nonzero(v_norm < 1e-5)
    # print('len != 0', x)
    # print(np.nonzero(v_norm==0))

    b = edge_verts[:, 1, :] - edge_verts[:, 0, :] ## [N, 3]
    b_norm = np.linalg.norm(b, axis=-1) 
    eb = b/b_norm[...,None] ## [N, 3]
    
    cos_angle = (ev*eb[None,]).sum(axis=-1, keepdims=False)
    qids, eids = np.nonzero(cos_angle>0.9999)

    intersections = []
    unique_qids, counts = np.unique(qids, return_counts=True)

    eid_set = set()
    queries_for_insert = []
    for qid, cnt in zip(unique_qids, counts):
        
        eid = eids[qids == qid]
        ratio = v_norm[qid, eid] / b_norm[eid]

        min_eid = ratio.argmin()
        intersection_eid = eid[min_eid]
        if ratio[min_eid] < 1:

            if intersection_eid in eid_set:
                continue

            eid_set.add(intersection_eid)
            intersections.append((qid, intersection_eid))
            queries_for_insert.append(queries[qid])

    return queries_for_insert, intersections


"""
inserted_points: [M, 3]
edges: a list of edges (vid0, vid1) in mesh
intersection_pairs: a list of pairs, pair:(insert pt id, edge id)
"""
def split_mesh_by_path(mesh, inserted_points, edges, intersection_pairs):

    num_verts = len(mesh.vertices)

    reverse_edges = edges[:, [1,0]]
    edges = edges.tolist()
    reverse_edges = reverse_edges.tolist()

    # masked_face_ids = set()
    # for i, e in enumerate(mesh.edges):
    #     e = e.tolist()
    #     if e in edges:
    #         masked_face_ids.add(mesh.edges_face[i].tolist())
    #     elif e in reverse_edges:
    #         masked_face_ids.add(mesh.edges_face[i].tolist())

    masked_face_ids = set()
    for i, pair in enumerate(intersection_pairs):

        e = edges[pair[1]]
        
        for fid in mesh.vertex_faces[e[0]]:
            if fid == -1:
                break
            else:
                masked_face_ids.add(fid)
        
        for fid in mesh.vertex_faces[e[1]]:
            if fid == -1:
                break
            else:
                masked_face_ids.add(fid)
        
    masked_face_ids = list(masked_face_ids)        
    kept_face_ids = np.ones(len(mesh.faces))
    kept_face_ids[masked_face_ids] = 0

    # print('masked_face_ids', masked_face_ids)

    faces_new = mesh.faces[masked_face_ids].tolist()
    ## remove faces from inserted faces that will be inserted with nodes
    for i, pair in enumerate(intersection_pairs):
        e = edges[pair[1]]
        ## find faces that will get inserted nodes
        removed_faces = []
        inserted_faces = []
        inserted = num_verts + i

        for fi in faces_new:
            if e[0] in fi and e[1] in fi:
                removed_faces.append(fi)
                a = e[0]
                b = e[1]
                ia = fi.index(e[0])
                if fi[(ia+1)%3] == b:
                    c = fi[(ia-1)%3]
                    inserted_faces.append([a, inserted, c])
                    inserted_faces.append([inserted, b, c])
                else: ## reverse
                    c = fi[(ia+1)%3]
                    inserted_faces.append([a, c, inserted])
                    inserted_faces.append([inserted, c, b])
        if len(inserted_faces) == 0:
            print('new face no exist', i, pair, e)
            print('faces', faces_new)
            print('vertex-face e0', mesh.vertex_faces[e[0]])
            print('vertex-face e1', mesh.vertex_faces[e[1]])
        for rf in removed_faces:
            faces_new.remove(rf)
        for fi in inserted_faces:
            faces_new.append(fi)


    # print(faces_new)

    vertices = np.concatenate([mesh.vertices, inserted_points], axis=0)
    faces = np.concatenate([mesh.faces[kept_face_ids>0], faces_new], axis=0)

    # print('check', vertices.shape, faces.max())

    # for f in faces:
    #     for i in range(3):
    #         v0 = f[i]
    #         v1 = f[(i + 1) % 3 ]
    #         if v0 == v1:
    #             print('self_intersect', f)
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    # out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return out_mesh



def split_mesh(mesh, path_pts):
    ## add some noises to the path_pts, so we can quickly find the faces that the path intersects
    # path_pths_without_ends = path_pts[1:-1, :] ## remove the start and end
    path_pths_without_ends = path_pts[:, :] 
    pq_mesh = trimesh.proximity.ProximityQuery(mesh)
    queries = np.array(path_pths_without_ends)

    _, _, fids = pq_mesh.on_surface(queries)
    fids = np.unique(fids)
    vids = np.unique((mesh.faces[fids]).reshape(-1))
    # face_verts = mesh.vertices[vids]

    ##
    edges = [
        mesh.faces[fids][:,0], 
        mesh.faces[fids][:,1], 
        mesh.faces[fids][:,1], 
        mesh.faces[fids][:,2], 
        mesh.faces[fids][:,2], 
        mesh.faces[fids][:,0]
    ]
    edges = np.stack(edges, axis=-1)
    edges = edges.reshape(-1, 2)
    edges_sorted = np.sort(edges, axis=1)
    unique, _ = trimesh.grouping.unique_rows(edges_sorted)
    edges_unique = edges_sorted[unique] 
    # write_obj_file(os.path.join(save_dir, f"selected_faces.obj"), face_verts)
    edge_verts = mesh.vertices[edges_unique]
    queries_for_insert, intersection_pairs = find_intersection_pairs(path_pths_without_ends, edge_verts)

    """
    intersection_pairs: a list of pairs
    pair: (qid, eid)
    """
    mesh = split_mesh_by_path(mesh, queries_for_insert, edges_unique, intersection_pairs)
    return mesh


def floodfill_label_mesh(
    mesh: Mesh, 
    all_picked_pts: list, 
    face_adjacency: list,
    face_adjacency_edges: list
):

    if len(all_picked_pts) == 0:
        f = mesh.faces()
        return [list(range(len(f)))]
        
    seg_edges = set()
    face_adjacency_edges = np.sort(face_adjacency_edges, axis=1)
    for seg_path in all_picked_pts:
        for i in range(1, len(seg_path)):
            geodesic_mesh = mesh.geodesic(seg_path[i - 1], seg_path[i])
            geodesic_vertices = geodesic_mesh.vertices()
            for i in range(1, len(geodesic_vertices)):
                v0 = mesh.closest_point(geodesic_vertices[i - 1], return_point_id=True)
                v1 = mesh.closest_point(geodesic_vertices[i], return_point_id=True)
                seg_edges.add((min(v0, v1), max(v0, v1)))

    assert len(face_adjacency) == len(face_adjacency_edges)

    face_adjacency_valid = []
    for i in range(len(face_adjacency_edges)):
        if tuple(face_adjacency_edges[i].tolist()) not in seg_edges:
            face_adjacency_valid.append(face_adjacency[i])

    graph = nx.from_edgelist(face_adjacency_valid)
    groups = [x for x in nx.connected_components(graph)]

    return groups
