import trimesh
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check the manifold of the mesh')
    parser.add_argument('--mesh', required=True, type=str, help='path to mesh')
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, process=False, maintain_order=True)
    print('Is the mesh manifold?', mesh.is_watertight)