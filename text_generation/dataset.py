import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from configs import configs
from tokenizer import create_tokenizer
from utils import get_max_report_len

tokenizer = create_tokenizer()

class ChestXRayCaptionDataset(Dataset):
    def __init__(self, data_split: str, size_256=True, augment_func=None, transform=None):
        """
        params
        
        data_split: train, val, or test
        augment_func: augments the image, leave None to do nothing
        """
        if data_split == 'train':
            data_path = configs['train_csv']
        elif data_split == 'val':
            data_path = configs['val_csv']
        elif data_split == 'test':
            data_path = configs['test_csv']
        elif data_split == 'all':
            data_path = configs['mimic_csv_file_path']
        else:
            raise AttributeError(f"Invalid data_split: {data_split}")

        # Load dataframe
        df = pd.read_csv(data_path)

        # Stores captions and image paths
        if size_256:
            # Cached resize
            self.image_paths = df['img_path_256'].values
        else:
            # Original size (512)
            self.image_paths = df['img_path'].values

        self.captions = df['report'].values
        self.caplens = df['report'].str.split().apply(lambda x: len(x)).values

        self.data_split = data_split
        self.input_shape = configs['input_shape']
        self.augment_func = augment_func
        self.dataset_size = self.captions.shape[0]
        self.tokenizer = tokenizer
        self.pad_length = get_max_report_len()
        self.transform = transform

    def __getitem__(self, index):
        """
        Parameters:
            index: index of item in dataset

        Returns:
            image: torch.FloatTensor of size configs['input_shape']
            caption: unpadded caption
            caplen: [length of caption]
        """
        # Load image
        image = self.load_image(self.image_paths[index]) # (256, 256, 3)
        if self.transform is not None:
            image = self.transform(image)

        # Load caption
        caption = torch.LongTensor(self.tokenizer.lookup_indices(self.captions[index].split()))
        caption = self.pad_to_length(caption, self.pad_length, self.tokenizer.stoi['<pad>'])
        caplen = torch.LongTensor([self.caplens[index]])

        return image, caption, caplen

    def __len__(self):
        """
        Returns:
            number of elements in this split dataset
        """
        return self.dataset_size

    def load_image(self, image_path):
        """
        Parameters:
            image_path

        Returns:
            image: image of size configs['input_shape']
        """
        image = Image.open(image_path).convert('RGB')
        if self.augment_func is not None:
            image = self.augment_func(image)
        return image

    def pad_to_length(self, seq: torch.Tensor, length: int, pad_value: int):
        """
        Parameters:
            seq: 1D torch.Tensor of indices
            length: size to pad to
        
        Returns:
            1D torch.Tensor of size (length,)
        """
        to_right = length - seq.shape[0]
        return F.pad(seq, pad=(0, to_right), mode='constant', value=pad_value)

class MultiLabelDataset(Dataset):
    def __init__(self, data_split: str, size_256=True, augment_func=None, transform=None, image_shape=None, device=None):
        """
        params
        
        data_split: train, val, or test
        augment_func: augments the image, leave None to do nothing
        """
        if data_split == 'train':
            image_data_path = configs['train_csv']
            label_data_path = configs['train_label_csv']
        elif data_split == 'val':
            image_data_path = configs['val_csv']
            label_data_path = configs['val_label_csv']
        elif data_split == 'test':
            image_data_path = configs['test_csv']
            label_data_path = configs['test_label_csv']
        else:
            raise AttributeError(f"Invalid data_split: {data_split}")

        # Load dataframes
        image_df = pd.read_csv(image_data_path)
        label_df = pd.read_csv(label_data_path)

        # Stores labels and image paths
        if size_256:
            # Cached resize
            self.image_paths = image_df['img_path_256'].values
        else:
            # Original size (512)
            self.image_paths = image_df['img_path'].values
        self.labels = label_df.values

        # Other stuff
        self.data_split = data_split
        self.input_shape = configs['input_shape'] if image_shape == None else image_shape
        self.dataset_size = len(self.image_paths)
        self.transform = transform
        self.augment_func = augment_func
        self.device = device
    
    def __getitem__(self, index):
        """
        Parameters:
            index: index of item in dataset

        Returns:
            image: torch.FloatTensor of size configs['input_shape']
            caption: unpadded caption
            caplen: [length of caption]
        """
        # Load image
        image = self.load_image(self.image_paths[index]) # (256, 256, 3)
        if self.transform is not None:
            image = self.transform(image)

        # Load labels
        label = self.labels[index]
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        """
        Returns:
            number of elements in this split dataset
        """
        return self.dataset_size

    def load_image(self, image_path):
        """
        Parameters:
            image_path

        Returns:
            image: image of size configs['input_shape']
        """
        image = Image.open(image_path).convert('RGB')
        if self.augment_func is not None:
            image = self.augment_func(image)
        return image


def test_report_dataset():
    val = ChestXRayCaptionDataset('val')
    image, caption, caplen, allcaps = val[1]
    print(image.shape)
    print(caption.shape)
    print([tokenizer.itos[i] for i in caption])
    print(caplen)
    print(allcaps)
    plt.imshow(image.numpy())
    plt.show()
    print(f"Image size: {image.numpy().shape}")

def test_label_dataset():
    import torchvision.transforms as transforms

    train = MultiLabelDataset(
        'train',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
        ])
    )
    img, label = next(iter(train))
    print(type(img))
    print(img.shape)
    print(label.shape)

if __name__ == "__main__":
    test_label_dataset()
