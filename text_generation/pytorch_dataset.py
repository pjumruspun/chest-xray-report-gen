from torch.utils.data import Dataset
from configs import configs
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import torch
from pytorch_tokenizer import create_tokenizer
import matplotlib.pyplot as plt
from utils import get_max_report_len
import torch.nn.functional as F

tokenizer = create_tokenizer()

class ChestXRayDataset(Dataset):
    def __init__(self, data_split: str, augment_func=None, transform=None):
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
        else:
            raise AttributeError(f"Invalid data_split: {data_split}")

        # Load dataframe
        df = pd.read_csv(data_path)

        # Stores captions and image paths
        self.captions = df['report'].values
        self.caplens = df['report'].str.split().apply(lambda x: len(x)).values
        self.image_paths = df['img_path'].values
        self.data_split = data_split
        self.input_shape = configs['input_shape']
        self.augment_func = augment_func
        self.dataset_size = self.captions.shape[0]
        self.tokenizer = tokenizer
        self.pad_length = get_max_report_len()
        self.transform = transform
        self.cpi = 1

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
        image = torch.FloatTensor(image)
        if self.transform is not None:
            image = image.permute(2, 0, 1) # (3, 256, 256)
            image = self.transform(image)
            image = image.permute(1, 2, 0) # (256, 256, 3)

        # Load caption
        caption = torch.LongTensor(self.tokenizer.lookup_indices(self.captions[index].split()))
        caption = self.pad_to_length(caption, self.pad_length, self.tokenizer.stoi['<pad>'])
        caplen = torch.LongTensor([self.caplens[index]])

        if self.data_split == 'train':
            return image, caption, caplen 
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            # cpi = 1 so allcaps = caption
            # This is redundant, should fix it soon
            return image, caption, caplen , caption

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
        image = imread(image_path)
        image = resize(image, self.input_shape)
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
        

if __name__ == "__main__":
    val = ChestXRayDataset('val')
    image, caption, caplen, allcaps = val[1]
    print(image.shape)
    print(caption.shape)
    print([tokenizer.itos[i] for i in caption])
    print(caplen)
    print(allcaps)
    plt.imshow(image.numpy())
    plt.show()
    print(f"Image size: {image.numpy().shape}")