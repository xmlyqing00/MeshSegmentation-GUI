from vedo import load, Plotter, Sphere, Arrow, Text2D, Mesh, write, Line


if __name__ == '__main__':

    mesh_path = 'data/color/catcher_vertex_color.obj'

    mesh = load(mesh_path)

    plt = Plotter(axes=0, bg='white', size=(800, 800))

    plt.add(mesh)

    plt.show()

    plt.close()