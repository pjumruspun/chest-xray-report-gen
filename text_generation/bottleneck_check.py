import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from torch import nn
import torchvision
from torchvision import transforms
from dataset import MultiLabelDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device("cuda")
CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices']

normalize = transforms.Normalize(
    mean=[0.4684, 0.4684, 0.4684], 
    std=[0.3021, 0.3021, 0.3021]
)

generic_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    normalize
])

class Chexnet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, input_size=(256, 256), pretrained=True):
        super(Chexnet, self).__init__()

        encoded_image_size = input_size[0] // 32
        self.linear_input_size = 1024 * encoded_image_size ** 2
        
        # Load densenet pretrained weight into this
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        
        # Original chexnet classifier
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )

        # Pretrained weights?
        if pretrained:
            print(f"loading pretrained weights..")
            chexnet_checkpoint_path = 'weights/chexnet.pth.tar'
            checkpoint = torch.load(chexnet_checkpoint_path)
            self.load_state_dict(checkpoint['state_dict'], strict=False)

        # Create the true densenet we're going to use
        modules = list(self.densenet121.children())[:-1]
        self.true_densenet = nn.Sequential(*modules)

        # New classifier head
        self.fc = nn.Linear(self.linear_input_size, 13)

        self = self.to(device)

    def forward(self, x):
        encoded_images = self.true_densenet(x)
        flattened = encoded_images.reshape(-1, self.linear_input_size)
        probs = torch.sigmoid(self.fc(flattened))
        return encoded_images, probs

def save_checkpoint(epoch, encoder, optimizer, scheduler, train_loss, val_loss):
    cur_time = str(datetime.now())
    state = {'epoch': epoch,
             'encoder': encoder,
             'optimizer': optimizer,
             'scheduler': scheduler,
             'time_saved': cur_time,
             'train_loss': train_loss,
             'val_loss': val_loss,
    }

    cur_time = cur_time.replace(' ', '_').replace(':', '-')
    filename = 'weights/pretrained_encoder/pretrained_enc_' + cur_time + '.pth.tar'
    torch.save(state, filename)

def pretrain_encoder(encoder, train_loader, loss_fn, optimizer, epoch):
    encoder.train()
    losses = []

    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)

        _, probs = encoder(imgs)
        loss = loss_fn(probs, labels)
        losses.append(loss)
        loss.backward()
        if ((i + 1) % 4) == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

    return torch.stack(losses).mean()

def validate_pretrained_encoder(encoder, val_loader, loss_fn):
    encoder.eval()
    losses = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(device)
            labels = labels.to(device)

            _, probs = encoder(imgs)
            loss = loss_fn(probs, labels)
            losses.append(loss)
            

    return torch.stack(losses).mean()

# checkpoint_path = 'weights/pretrained_encoder/pretrained_enc_2022-02-25_11-02-35.926222.pth.tar'
checkpoint_path = None
epochs = 1
if checkpoint_path != None:
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    encoder = checkpoint['encoder']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
else:
    start_epoch = 0
    encoder = Chexnet()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=7e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor=0.2, 
        patience=2, 
        min_lr=1e-6
    )

loss_fn = nn.BCELoss().to(device)

train_loader = DataLoader(
    MultiLabelDataset('train', transform=generic_transform),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    MultiLabelDataset('val', transform=generic_transform),
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

lrs = []
train_plot = []
val_plot = []

def train():
    print(f"start training from epoch {start_epoch}")
    for epoch in range(start_epoch, start_epoch + epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Training epoch {epoch+1} with lr = {current_lr}...")
        # train_loss = pretrain_encoder(encoder, val_loader, loss_fn, optimizer, epoch)
        val_loss = validate_pretrained_encoder(encoder, val_loader, loss_fn)

train()
# https://pastebin.com/a50FBU0F