import librosa
import argparse
from pathlib import Path
from tqdm import tqdm
 
def get_duration_mp3_and_wav(file_path):
     """
     :param file_path:
     :return: int duration in seconds
     """
     duration = librosa.get_duration(filename=file_path)
     return duration
 

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/EDM/')
    args = parser.parse_args()
    
    d = 0
    for i in tqdm(list(Path(args.file_path).glob('*.mp3'))):
        duration = get_duration_mp3_and_wav(i.as_posix()) # as_posix is used to convert the path to string
        d += duration
 
    print(f'duration = {d}')
