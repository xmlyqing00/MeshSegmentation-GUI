import numpy as np
import trimesh
from vedo import Mesh
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
def split_mesh_by_path(mesh, face_patches, inserted_points, edges, intersection_pairs):

    num_verts = len(mesh.vertices)

    reverse_edges = edges[:, [1,0]]
    edges = edges.tolist()
    reverse_edges = reverse_edges.tolist()

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

    faces_new = mesh.faces[masked_face_ids].tolist()
    face_patches_new = face_patches[masked_face_ids].tolist()

    ## remove faces from inserted faces that will be inserted with nodes
    for i, pair in enumerate(intersection_pairs):
        e = edges[pair[1]]
        ## find faces that will get inserted nodes
        removed_faces = []
        inserted_faces = []
        inserted_faces = []
        inserted = num_verts + i
        face_patches_old = []

        for fidx, fi in enumerate(faces_new):
            if e[0] in fi and e[1] in fi:
                removed_faces.append(fi)
                face_patches_old.append(face_patches_new[fidx])
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
            rf_idx = faces_new.index(rf)
            faces_new.pop(rf_idx)
            face_patches_new.pop(rf_idx)
            # faces_new.remove(rf)
        for fidx, fi in enumerate(inserted_faces):
            faces_new.append(fi)
            face_patches_new.append(face_patches_old[fidx // 2])

        # print(len(removed_faces), len(inserted_faces))

    vertices = np.concatenate([mesh.vertices, inserted_points], axis=0)
    faces = np.concatenate([mesh.faces[kept_face_ids>0], faces_new], axis=0)
    face_patches_out = np.concatenate([face_patches[kept_face_ids>0], face_patches_new], axis=0)
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    # out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return out_mesh, face_patches_out


def split_mesh(mesh, path_pts, face_patches):
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
    mesh, face_patches_out = split_mesh_by_path(mesh, face_patches, queries_for_insert, edges_unique, intersection_pairs)
    group_id = 0
    mask = []
    while True:
        idmask = face_patches_out == group_id
        face_ids = idmask.nonzero()[0]
        if len(face_ids) == 0:
            break
        mask.append(face_ids)
        # print('seg output', group_id, mask)
        group_id += 1

    return mesh, mask


def floodfill_label_mesh(
    tri_mesh: trimesh.Trimesh, 
    boundary_edges: set,
    all_picked_pt_pid: list, 
):
    
    face_adjacency = tri_mesh.face_adjacency
    face_adjacency_edges = tri_mesh.face_adjacency_edges

    assert len(face_adjacency) == len(face_adjacency_edges)

    edge_attributes = [
        (tri_mesh.edges_unique[i][0], tri_mesh.edges_unique[i][1], {'weight': tri_mesh.edges_unique_length[i]})
        for i in range(len(tri_mesh.edges_unique_length))
    ]
    graph = nx.from_edgelist(edge_attributes)
    
    for picked_pt_pid in all_picked_pt_pid:
        for i in range(1, len(picked_pt_pid)):
            path_pts = nx.shortest_path(
                graph, 
                picked_pt_pid[i - 1], 
                picked_pt_pid[i], 
                weight='weight'
            )
            for i in range(1, len(path_pts)):
                v0 = path_pts[i - 1]
                v1 = path_pts[i]
                boundary_edges.add((min(v0, v1), max(v0, v1)))   

    face_adjacency_edges = np.sort(face_adjacency_edges, axis=1)
    face_adjacency_valid = []
    for i in range(len(face_adjacency_edges)):
        if tuple(face_adjacency_edges[i].tolist()) not in boundary_edges:
            face_adjacency_valid.append(face_adjacency[i])

    graph = nx.from_edgelist(face_adjacency_valid)
    groups = [list(x) for x in nx.connected_components(graph)]

    return groups

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