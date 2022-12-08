from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale


def read_image(path):
    return Image.open(path).convert('RGB')

def get_transform():
    return Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize(mean=[0.5],
                  std=[0.5]),
    ])

