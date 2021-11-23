import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from maad import sound, util

# Image transforms for augmentation during training
train_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomAdjustSharpness(sharpness_factor=8),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# During validation use only tensor and normalization
eval_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SoundDetection(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    This function was implemented to load only precomputed audio spectrograms.
    """
    def __init__(self, data_dir, transform):
        """
        Loads data from csv file. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        super().__init__()
        df = pd.read_csv(data_dir)
        self.filenames = df.fname_path.to_list()
        self.labels = df.label.to_numpy()
        self.transform = transform


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = image.convert('RGB') # treat grayscale images as RGB
        image = self.transform(image)
        return image, self.labels[idx]

class SoundDetectionAudio(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    Accepts raw audio data. Computes the audio spectrogram when loading the data.
    """
    def __init__(self, data_dir, transform):
        """
        Loads data from csv file. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        super().__init__()
        df = pd.read_csv(data_dir)
        self.filenames = df.fname_path.to_list()
        self.labels = df.label.to_numpy()
        self.transform = transform


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        target_fs = 24000  # target fs of project
        spectro_nperseg = 512
        spectro_noverlap = 256
        db_range = 80

        # preprocess sound
        s, fs = sound.load(self.filenames[idx])
        s = sound.resample(s, fs, target_fs)
        #s = sound.normalize(s, max_db=0)
        Sxx, tn, fn, ext = sound.spectrogram(
            s, fs=target_fs, nperseg=spectro_nperseg, noverlap=spectro_noverlap)
        Sxx = util.power2dB(Sxx, db_range, db_gain=20)

        # treat as image
        image = 255 * (Sxx + db_range) / db_range
        image = Image.fromarray(np.uint8(image))
        image = image.convert('RGB') # treat grayscale images as RGB
        image = self.transform(image)
        return image, self.labels[idx]


class SoundDetectionInfer(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Loads data from csv file. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        super().__init__()
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
        self.labels = [os.path.basename(filename) for filename in self.filenames]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = image.convert('RGB') # treat grayscale images as RGB
        image = self.transform(image)
        return image, self.labels[idx]



def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset. names must be 
        <train_customname> and <val_customname.csv>, or <test_customname.csv>
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    if types==['infer']:
        dl = DataLoader(SoundDetectionInfer(data_dir, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                            num_workers=params.num_workers,
                                            pin_memory=params.cuda)
        dataloaders['infer'] = dl

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_boapra.csv".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(SoundDetectionAudio(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(SoundDetectionAudio(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders

