from . import get_transform
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Mel_UnconDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform if transform is not None else get_transform()
        self.loader = default_loader
        self.files = list(self.root.glob('*.png'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.transform(self.loader(self.files[idx]))
        example = {'image': img}
        return example
