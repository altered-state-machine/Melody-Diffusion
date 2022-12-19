from . import get_transform
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Mel_UnconDataset(Dataset):
    def __init__(self, root, train=True,transform=None):
        self.root = Path(root)
        self.transform = transform if transform is not None else get_transform()
        self.loader = default_loader
        files = list(self.root.glob('*.png'))
        if train:
            self.files = files[:int(len(files)*0.8)]
        else:
            self.files = files[int(len(files)*0.8):]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.transform(self.loader(self.files[idx]))
        example = {'image': img}
        return example
