import numpy as np
import trimesh
from vedo import Mesh
import queue


"""
vert_edges: [N, 2, 3]
queries: [M, 3]
"""
def find_intersection_pairs(queries, edge_verts):
    v = queries[:, None, :] - edge_verts[None, :, 0, :] ## [M, N, 3]
    v_norm = np.linalg.norm(v, axis=-1) 
    ev = v / v_norm[...,None] ## [M, N, 3]
    x = np.nonzero(v_norm < 1e-5)
    print('len != 0', x)

    # print(np.nonzero(v_norm==0))

    b = edge_verts[:, 1, :] - edge_verts[:, 0, :] ## [N, 3]
    b_norm = np.linalg.norm(b, axis=-1) 
    eb = b/b_norm[...,None] ## [N, 3]
    
    cos_angle = (ev*eb[None,]).sum(axis=-1, keepdims=False)
    qids, eids = np.nonzero(cos_angle>0.9999)

    intersections = []
    unique_qids, counts = np.unique(qids, return_counts=True)

    queries_for_insert = []
    for qid, cnt in zip(unique_qids, counts):
        # print(qid, cnt)
        eid = eids[qids == qid]
        ratio = v_norm[qid, eid] / b_norm[eid]
        intersected_eid = eid[ratio <= 1]
        # print(ratio, eid, intersected_eid)
        # assert len(intersected_eid) == 1
        intersections.append((qid, intersected_eid[0]))
        queries_for_insert.append(queries[qid])
    # print(intersections, len(intersections))
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

    masked_face_ids = set()
    for i, e in enumerate(mesh.edges):
        e = e.tolist()
        if e in edges:
            masked_face_ids.add(mesh.edges_face[i].tolist())
        elif e in reverse_edges:
            masked_face_ids.add(mesh.edges_face[i].tolist())
    
    print('check edges == masked_face_ids')
    print(len(masked_face_ids), len(edges))

    masked_face_ids = list(masked_face_ids)
    # print(masked_face_ids)
    faces_new = mesh.faces[masked_face_ids].tolist()
    # print(faces_new)
    # input()

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
            print('new face no exist', i, pair)
        for rf in removed_faces:
            faces_new.remove(rf)
        for fi in inserted_faces:
            faces_new.append(fi)

    kept_face_ids = np.ones(len(mesh.faces))
    kept_face_ids[masked_face_ids] = 0

    print(faces_new)

    vertices = np.concatenate([mesh.vertices, inserted_points], axis=0)
    faces = np.concatenate([mesh.faces[kept_face_ids>0], faces_new], axis=0)

    print('check')
    print(vertices.shape, faces.max())

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
    face_verts = mesh.vertices[vids]

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


def floodfill_label_mesh(mesh: Mesh, all_geodesic_paths: list, vertex_faces: list):

    segment_vertices = np.concatenate(all_geodesic_paths)
    pids_border = set()
    for i in range(segment_vertices.shape[0]):
        pid = mesh.closest_point(segment_vertices[i], return_point_id=True)
        pids_border.add(pid)

    f = mesh.faces()
    label_num = 0
    f_labels = np.zeros(len(f), np.int32)
    for fid in range(len(f)):
        if f_labels[fid] > 0:
            continue
        
        label_num += 1
        f_labels[fid] = label_num
        que = queue.Queue()
        que.put(fid)
        while not que.empty():

            fid_cur = que.get()
            f_pts = f[fid_cur]
            flag_border_face = False
            for vid0 in range(3):
                vid1 = (i + 3) % 3
                if f_pts[vid0] in pids_border:
                    flag_border_face = True
                    break
            
            if flag_border_face:
                # f_labels[fid_cur] = 1000
                print(fid_cur)
                continue

            for vid0 in range(3):
                f_next_list = vertex_faces[f_pts[vid0]]
                # print('print vertex-faces:', f_pts[vid0])
                for f_next_id in f_next_list:
                    if f_next_id == -1:
                        break
                    if f_labels[f_next_id] > 0:
                        continue

                    f_labels[f_next_id] = label_num
                    que.put(f_next_id)
    
    print(label_num)

    return f_labels
