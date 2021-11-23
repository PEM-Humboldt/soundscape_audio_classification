import os
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

class MultilabelDataset(Dataset):
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

        self.transform = transform

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.insect_label = []
        self.bird_label = []

        # read the annotations from the CSV file
        with open(data_dir) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['file_path'])
                self.insect_label.append(int(row['insects']))
                self.bird_label.append(int(row['birds']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)
        img = img.convert('RGB') # treat grayscale images as RGB

        # apply the image transformation
        img = self.transform(img)
        labels = {
            'insect_label': self.insect_label[idx],
            'bird_label': self.bird_label[idx]}
        
        return img, labels


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset. names must be 
        <train_customname.csv> and <val_customname.csv>, or <test_customname.csv>
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_birdinsect.csv".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(MultilabelDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(MultilabelDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
