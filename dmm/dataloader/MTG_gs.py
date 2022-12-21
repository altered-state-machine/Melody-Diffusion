import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import random
import csv
import torch
from collections import defaultdict
from torch.utils import data
import torchaudio
from torchaudio import transforms
from util import waveform_from_spectrogram, spectrogram_from_waveform, image_from_spectrogram, wav_bytes_from_spectrogram_image, waveform_from_tensor


CATEGORIES = ['genre', 'instrument', 'mood/theme']
TAG_HYPHEN = '---'
METADATA_DESCRIPTION = 'TSV file with such columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS'
DESCRIPTION = ['track', 'song', 'music', 'clip', 'melody']
PROMPT_SINGER = ['The singer is ', 'The artist is ', 'The performer is ', 'The musician is ', 'The song is created by ', 'This song is sung by ']
PROMPT_TITLE = ['Creator', 'Artist', 'Performer', 'Musician']

def get_id(value):
    return int(value.split('_')[1])

def plural(l):
    l = list(l)
    if len(l) == 1:
        return 'is '+l[0]
    elif len(l) == 2:
        return 'are '+' and '.join(l)
    else:
        return 'are '+', '.join(l[:-1]) + ' and ' + l[-1]

def tag2prompt(list):
    prompts = ''
    tags = {}
    for tag_str in list:
        cat, tag = tag_str.split(TAG_HYPHEN)
        if cat not in tags:
            tags[cat] = set()
        tags[cat].update(set(tag.split(",")))
    for cat in tags:
        if cat == 'instrument':
            prompt = 'The {} used in the {} {}. '.format(cat, random.choice(DESCRIPTION), plural(tags[cat]))
        if cat == 'mood/theme':
            prompt = 'The {} of the {} {}. '.format(random.choice(['mood','theme']), random.choice(DESCRIPTION), plural(tags[cat]))
        else:
            prompt = 'The {} of the {} {}. '.format(cat, random.choice(DESCRIPTION), plural(tags[cat]))
        prompts += prompt
    # print(prompts)
    return prompts


def tsv2dict(fn):
    tags = defaultdict(dict)
    tracks = {}
    with open(fn, 'r') as pf:
        reader = csv.reader(pf, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = get_id(row[0])
            tracks[track_id] = {
                'artist_id': get_id(row[1]),
                'album_id': get_id(row[2]),
                'path': row[3],
                'duration': float(row[4]),
                'tags': [i.split(TAG_HYPHEN) for i in row[5:]],  # raw tags, not sure if will be used
                'prompt': tag2prompt(row[5:])+'{}{}{}. '.format(random.choice(PROMPT_SINGER), random.choice(PROMPT_TITLE),str(get_id(row[1])))
            }
            # tracks[track_id].update({category: set() for category in CATEGORIES})

            for tag_str in row[5:]:
                category, tag = tag_str.split(TAG_HYPHEN)

                if tag not in tags[category]:
                    tags[category][tag] = set()

                tags[category][tag].add(track_id)

                if category not in tracks[track_id]:
                    tracks[track_id][category] = set()

                tracks[track_id][category].update(set(tag.split(",")))
    return tracks, tags


def stereo2mono(x):
    return torch.mean(x, dim=0, keepdim=True)

def gray2rgb(x):
    return torch.cat([x, x, x], dim=0)

class AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', type='audio', split=0):
        self.trval = tr_val
        self.root = root
        file_path = os.path.abspath(__file__).split('dmm')[0]
        fn = ''.join([file_path,'data/splits/split-%d/%s-%s.tsv' % (split, subset, tr_val)])
        self.type = type
        if type == "audio":
            # 512*512 mel_spectrogram
            # self.transform = transforms.MelSpectrogram(sample_rate=44100, n_fft=16384, hop_length=512, n_mels=512, f_min=0, f_max=10000, power=2)
            # self.power2db = transforms.AmplitudeToDB(top_db=80)
            self.loader = torchaudio.load
            self.suffix = 'mp3'
        elif type == 'mel':
            self.transform = None
            self.loader = np.load
            self.suffix = 'npy'
        elif type == 'image':
            self.transform = None
            self.suffix = 'jpg'
        else:
            raise ValueError("type not allowed")
        self.get_dictionary(fn)

    def get_norm_mel(self, waveform, _, window_size):
        begin_index = random.randint(0, len(waveform[0])-window_size)
        waveform_slice = waveform[:,begin_index:begin_index+window_size]
        mel = torch.pow(spectrogram_from_waveform(waveform_slice, _, n_fft=16384, hop_length=1024, win_length=4096, n_mels=512) , 0.25)
        mel = (mel/mel.max()).flip(0)
        mel = mel*2.0-1.0
        if torch.isnan(mel).any():
            mel, waveform_slice = self.get_norm_mel(waveform, _, window_size)
        return mel, waveform_slice

    def __getitem__(self, index):
        index_list = list(self.dictionary.keys())
        index = index_list[index]
        file_path = os.path.join(self.root, self.dictionary[index]['path'][:-3]+self.suffix)
        raw_data = self.loader(file_path)
        if self.type == 'audio':
            waveform, _ = raw_data[0], raw_data[1]
            if _ != 44100:
                waveform = torchaudio.transforms.Resample(_ , 44100)(waveform)
            window_size = 1024*512 # waveform slice size = 4096*512 n_fft*resolution

            mel, waveform_slice = self.get_norm_mel(waveform, _, window_size)
            # begin_index = random.randint(0, len(waveform[0])-window_size)
            # waveform_slice = waveform[:,begin_index:begin_index+window_size]
            # mel = torch.pow(spectrogram_from_waveform(waveform_slice,_, n_fft=16384, hop_length=1024, win_length=4096, n_mels=512) , 0.25)
            # mel = (mel/mel.max()).flip(0)

            # log_mel = self.power2db(mel)
            # log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())
            # log_mel = 1 - log_mel
            # mel = torch.pow(mel, 0.25)
            # mel = (log_mel/log_mel.max()).flip(0)
        elif self.type == 'mel':
            mel = raw_data[:,:90]
            mel = (mel-mel.min()) / (mel.max()-mel.min())
            mel = mel*2-1
        # tags = self.dictionary[index]['tags']
        assert torch.isnan(gray2rgb(mel[...,:512].unsqueeze(0))).any() == False, "Nan in Mel"
        return_dict = {'jpg': gray2rgb(mel[...,:512].unsqueeze(0)), 
        'caption': self.dictionary[index]['prompt'],
        'audio': waveform_slice}
        return return_dict

    def get_dictionary(self, fn):
        tracks, _ = tsv2dict(fn)
        self.dictionary = tracks

    def __len__(self):
        return len(self.dictionary)


def get_audio_loader(root, subset, batch_size, tr_val='train', type='audio',split=0, num_workers=0):
    data_loader = data.DataLoader(dataset=AudioFolder(root, subset, tr_val, type,split),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    # a,b = tsv2dict('/home/hu/audio-diffusion/data/splits/split-0/autotagging-train.tsv')
    a = get_audio_loader('/home/hu/database/MTG_audio', 'autotagging', 1, 'train','audio', 0, 0)

    # from tqdm import tqdm
    # for i in tqdm(a, total=len(a)):
    #     i

    sp = next(iter(a))
    spec_tensor = sp['jpg']
    sample_tensor = waveform_from_tensor((spec_tensor[0]+1.)/2.)
    torchaudio.save('rec_test_large_tensor.wav', torch.tensor(sample_tensor).unsqueeze(0), 44100)
    from torchvision.utils import save_image
    # torchaudio.save('test_large.wav', torch.mean(sp[1],1), 44100)
    save_image(sp['jpg'],'test_large.png') 
    # image_from_spectrogram(sp[0][...,:512].cpu().numpy()).save('test_large.png')
    # recover = torch.pow(sp[0][...,:512]*5,4)
    # rec_wav = waveform_from_spectrogram(recover,n_fft=16384, hop_length=512, win_length=4096,sample_rate=44100,num_samples=0)
    # torchaudio.save('rec_test_large.wav', torch.tensor(rec_wav), 44100)
    from PIL import Image
    img = Image.open('test_large.png')
    sample, durations = wav_bytes_from_spectrogram_image(img)
    sample = (sample - sample.min()) / (sample.max() - sample.min())
    torchaudio.save('rec_test_large.wav', torch.tensor(sample).unsqueeze(0), 44100)
    torchaudio.save('test_large.wav', torch.mean(sp['audio'],1), 44100)
    print('ok')