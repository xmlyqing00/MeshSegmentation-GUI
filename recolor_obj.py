import trimesh
from glob import glob


obj_paths = glob('data/color/*.obj')
for obj_path in obj_paths:

    mesh = trimesh.load(obj_path)
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        mesh.visual = mesh.visual.to_color()
    
    mesh.export(obj_path.replace('.obj', '_vertex_color.obj'))