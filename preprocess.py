import argparse
import trimesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    mesh = trimesh.load(args.input)

    if args.output is None:
        args.output = args.input.replace('.obj', '_preprocessed.obj')

    mesh.export(args.output)
    print(f'Exported to {args.output}')

