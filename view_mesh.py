import argparse
from vedo import Plotter, Sphere, Text2D, Mesh, write, Line, utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser('View Mesh')
    parser.add_argument('--mesh', type=str, help='Path to the mesh file')
    parser.add_argument('--fid', type=int, help='Face id')
    parser.add_argument('--vid', type=int, help='Vertex id')
    parser.add_argument('--highlight', action='store_true', help='Highlight the face')

    args = parser.parse_args()
    print(args)

    plotter = Plotter(bg='white')

    mesh = Mesh(args.mesh)

    if args.fid is not None:
        mesh.cellcolors[args.fid] = [255, 0, 0, 255]
        if args.highlight:
            vs = mesh.vertices[mesh.cells[args.fid]]
            for v in vs:
                s = Sphere(v, r=0.001, c='r')
                plotter.add(s)
        # face = mesh.faces(args.fid)
    
    if args.vid is not None:
        s = Sphere(mesh.vertices[args.vid], r=0.001, c='g')
        plotter.add(s)

    plotter.add(mesh)
    plotter.show()
    plotter.close()
    