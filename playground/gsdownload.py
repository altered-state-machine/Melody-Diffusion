import csv
from google.cloud import storage
from pathlib import Path

root = '/Melody-Diffusion/data/splits/split-0/autotagging-train.tsv'
prefix = 'gs://asm-ai-data/Michael/Database/MTG_Audio/'


def get_files_path(root, prefix):
    l = []
    with open(root) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            l.append(''.join([prefix, row[3]]))
    return l


if __name__ == "__main__":
    files = get_files_path(root, prefix)
    client = storage.Client(project="ai-innovation-370705")
    Path('/MTG_Audio').mkdir(parents=True, exist_ok=True)
    for file in files:
        with open(file.split('Database')[-1]) as file_obj:
            client.download_blob_to_file(file, file_obj)