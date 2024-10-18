import numpy as np
from trimesh import primitives, util, transformations


def build_spheres(pt: np.array, radius, color):
    
    if len(pt.shape) > 1:
        meshes = []
        for p in pt:
            meshes.append(build_spheres(p, radius, color))
        return util.concatenate(meshes)

    sphere = primitives.Sphere(radius=radius, center=pt)
    sphere.visual.vertex_colors = color

    return sphere


def build_cylinders(pt0: np.array, pt1: np.array, radius, color):
    
    if len(pt0.shape) > 1:
        meshes = []
        for p0, p1 in zip(pt0, pt1):
            meshes.append(build_cylinders(p0, p1, radius, color))
        return util.concatenate(meshes)

    h = np.linalg.norm(pt0 - pt1)
    stick = primitives.Cylinder(radius=radius, height=h, sections=6)
    stick.visual.vertex_colors = color

    normal = pt0 - pt1
    normal = normal / np.linalg.norm(normal)
    rot_axis = np.cross(stick.direction, normal)
    rot_angle = np.arccos(np.dot(stick.direction, normal))
    rot_mat = transformations.rotation_matrix(rot_angle, rot_axis, (0, 0, 0))
    trans_mat1 = transformations.translation_matrix((0, 0, h / 2))
    trans_mat2 = transformations.translation_matrix(pt1)
    transform_mat = trans_mat2 @ rot_mat @ trans_mat1
    stick.apply_transform(transform_mat)
    
    return stick