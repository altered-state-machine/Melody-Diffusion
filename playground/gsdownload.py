import csv
from google.cloud import storage
from pathlib import Path

import os, sys
print(os.path.abspath(__file__))
from multiprocessing.pool import ThreadPool

# root = '/home/Melody-Diffusion/data/splits/split-0/autotagging-train.tsv'
root = '/home/Melody-Diffusion/data/splits/split-0/autotagging-validation.tsv'
# root = 'data/splits/split-0/autotagging-train.tsv'
prefix = 'gs://asm-ai-data/Michael/Database/MTG_Audio/'
prefix_local = '/home/MTG_Audio/'


def get_files_path(root, prefix):
    l = []
    with open(root) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            l.append(row[3])
    return l

def download_(file): 
    print(file)
    if not (Path(prefix_local)/file.split('/')[0]).exists():
        (Path(prefix_local)/file.split('/')[0]).mkdir(parents=True, exist_ok=True)
    if (Path(prefix_local)/file).exists():
        print('{} Already Exists'.format(file))
    else:
        with open(Path(prefix_local)/file, 'wb') as file_obj:
            client.download_blob_to_file('gs://asm-ai-data/Michael/Database/MTG_Audio/'+file, file_obj)

if __name__ == "__main__":
    from tqdm import tqdm
    files = get_files_path(root, prefix)
    client = storage.Client(project="ai-innovation-370705")
    Path(prefix_local).mkdir(parents=True, exist_ok=True)
    pool = ThreadPool(256)
    _ = pool.starmap(download_, zip(files))
    pool.close()