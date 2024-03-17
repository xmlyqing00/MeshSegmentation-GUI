import os
from glob import glob

glob_path = './data/mydata/*'
folders = glob(glob_path)
for folder in folders:
    print(folder)

    try:
        os.system(f'python build_data_mydata.py --model_name {folder.split("/")[-1]}')
    except:
        print(f'Error in {folder}')
        continue
