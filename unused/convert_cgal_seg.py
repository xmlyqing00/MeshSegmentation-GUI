import argparse
import pathlib
import numpy as np
import json


root_dir = pathlib.Path('data/cgal_segs')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-file', type=str, default='seg_1.txt')
    args = parser.parse_args()

    seg_file_path = root_dir / args.seg_file
    with open(seg_file_path, 'r') as f:
        lines = f.readlines()

    lines = [int(line.strip()) for line in lines]
    
    mask = {}

    for i, group_id in enumerate(lines):
        # print(i, group_id)
        if mask.get(group_id):
            mask[group_id].append(i)
        else:
            mask[group_id] = [i]

        # print(mask)

    
    mask = dict(sorted(mask.items()))
    mask_list = []
    for k, v in mask.items():
        print(k, len(v))
        mask_list.append(v)

    mask_path = str(seg_file_path).replace('.txt', '.json')
    with open(mask_path, 'w') as f:
        json.dump(mask_list, f, ensure_ascii=False, indent=4)
    # print(mask)

