
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmm.util import waveform_from_tensor, wav_bytes_from_spectrogram_image
import PIL

import torch

import torchaudio

from torchvision.transforms import ToTensor


# image0 = PIL.Image.open('/home/hu/audio-diffusion/logs/2022-12-21T20-51-02_cond_1.5/images/train/reconstruction_gs-000010_e-000000_b-000009.png')
image = PIL.Image.open('test_large.png')
# wavb,_ = wav_bytes_from_spectrogram_image(image)
# wavb = (wavb - wavb.min()) / (wavb.max() - wavb.min())

# image0 = ToTensor()(image0)
# image0 = torch.clamp(image0 *2 -1,0,1)
image = ToTensor()(image)
wav = waveform_from_tensor(image)
torchaudio.save('rec_test_dele.wav', torch.tensor(wav).unsqueeze(0), 44100)


print('ok')