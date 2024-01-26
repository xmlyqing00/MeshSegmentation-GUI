import numpy as np
import trimesh
import networkx as nx
from loguru import logger


def find_intersection_pairs(path_pts, edge_verts, intersection_merged_threshold):

    v = path_pts[:, None, :] - edge_verts[None, :, 0, :] ## [M, N, 3]
    v_norm = np.linalg.norm(v, axis=-1) 
    ev = v / v_norm[...,None] ## [M, N, 3]

    b = edge_verts[:, 1, :] - edge_verts[:, 0, :] ## [N, 3]
    b_norm = np.linalg.norm(b, axis=-1) 
    eb = b/b_norm[...,None] ## [N, 3]
    
    cos_angle = (ev*eb[None,]).sum(axis=-1, keepdims=False)

    eid_set = set()
    queries_for_insert = []
    intersection_ignore_cnt = 0
    path_pts_new = []
    intersections = []

    for qid in range(len(path_pts)):
    # qids, eids = np.nonzero(cos_angle > 0.99)

    # unique_qids, _ = np.unique(qids, return_counts=True)
    # assert len(unique_qids) == len(path_pts)

    # for qid in unique_qids:
        eps = 0.99
        
        eid = np.nonzero(cos_angle[qid] > eps)[0]
        if len(eid) == 0:
            logger.warning(f'No intersection found for query {qid}')
            eid = np.array([cos_angle[qid].argmax()])

        ratio = v_norm[qid, eid] / b_norm[eid]

        min_eid = ratio.argmin()
        intersection_eid = eid[min_eid]
        ratio_min = ratio[min_eid]

        if ratio_min < 1:  # path is on vertex

            # Don't insert if the intersection is close to the end of the edge
            if ratio_min < intersection_merged_threshold or ratio_min > 1 - intersection_merged_threshold:
                intersection_ignore_cnt += 1
                if ratio_min < intersection_merged_threshold:
                    path_pts_new.append(edge_verts[intersection_eid, 0, :])
                else:
                    path_pts_new.append(edge_verts[intersection_eid, 1, :])
            else:
                
                path_pts_new.append(path_pts[qid])

                if intersection_eid not in eid_set:
                    eid_set.add(intersection_eid)
                    intersections.append((qid, intersection_eid))
                    queries_for_insert.append(path_pts[qid])
        else:
            path_pts_new.append(path_pts[qid])
        


    logger.debug(f'Intersection {len(eid_set)}, ignore count: {intersection_ignore_cnt}')

    return queries_for_insert, intersections, path_pts_new


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
    return out_mesh, face_patches_out


def split_mesh(mesh, path_pts, face_patches, intersection_merged_threshold=0.15):
    
    pq_mesh = trimesh.proximity.ProximityQuery(mesh)
    path_pts = np.array(path_pts)

    _, _, fids = pq_mesh.on_surface(path_pts)
    # vts = mesh.vertices[mesh.faces[fids]]

    fids = np.unique(fids)

    # edges = [
    #     mesh.faces[fids][:,0], 
    #     mesh.faces[fids][:,1], 
    #     mesh.faces[fids][:,1], 
    #     mesh.faces[fids][:,2], 
    #     mesh.faces[fids][:,2], 
    #     mesh.faces[fids][:,0]
    # ]
    # edges = np.stack(edges, axis=-1)
    # edges = edges.reshape(-1, 2)
    # edges_sorted = np.sort(edges, axis=1)
    # unique, _ = trimesh.grouping.unique_rows(edges_sorted)
    # edges_unique = edges_sorted[unique] 

    edges = mesh.edges_unique[mesh.faces_unique_edges[fids]].reshape(-1, 2)
    unique, _ = trimesh.grouping.unique_rows(edges)
    edges_unique = edges[unique]

    edge_verts = mesh.vertices[edges_unique]

    queries_for_insert, intersection_pairs, path_pts_new = find_intersection_pairs(path_pts, edge_verts, intersection_merged_threshold)

    if len(intersection_pairs) == 0:
        face_patches_out = face_patches
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    else:
        mesh, face_patches_out = split_mesh_by_path(mesh, face_patches, queries_for_insert, edges_unique, intersection_pairs)

    group_id = 0
    mask = []
    while True:
        idmask = face_patches_out == group_id
        face_ids = idmask.nonzero()[0]
        if len(face_ids) == 0:
            break
        mask.append(face_ids)
        group_id += 1

    return mesh, mask, path_pts_new


def floodfill_label_mesh(
    tri_mesh: trimesh.Trimesh, 
    boundary_edges: set,
):
    
    face_adjacency = tri_mesh.face_adjacency
    face_adjacency_edges = tri_mesh.face_adjacency_edges

    assert len(face_adjacency) == len(face_adjacency_edges)

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