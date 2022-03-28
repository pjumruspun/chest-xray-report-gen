import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MultiLabelDataset
from model import Chexnet
from utils import evaluate_transform, train_transform

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
    filename = f'weights/pretrained_encoder/pretrained_enc_epoch_{epoch+1}_' + cur_time + '.pth.tar'
    torch.save(state, filename)

def pretrain_encoder(encoder, train_loader, loss_fn, optimizer, gradient_accum_iter, epoch):
    encoder.train()
    losses = []

    for i, (imgs, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        imgs = imgs.cuda()
        labels = labels.cuda()

        _, probs = encoder(imgs)
        loss = loss_fn(probs, labels)
        losses.append(loss.detach())
        loss.backward()
        if ((i + 1) % gradient_accum_iter) == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

    return torch.stack(losses).mean()

def validate_pretrained_encoder(encoder, val_loader, loss_fn):
    encoder.eval()
    losses = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.cuda()
            labels = labels.cuda()

            _, probs = encoder(imgs)
            loss = loss_fn(probs, labels)
            losses.append(loss.detach())
            

    return torch.stack(losses).mean()

def plot_history(train_plot, val_plot, lrs):
    plt.title('train loss by epoch')
    plt.plot(train_plot)
    plt.show()
    plt.title('val loss by epoch')
    plt.plot(val_plot)
    plt.show()
    plt.title('train and val')
    plt.plot(train_plot, label='train_loss')
    plt.plot(val_plot, label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(lrs)
    plt.show()

def main():
    # checkpoint_path = 'weights/pretrained_encoder/pretrained_enc_epoch_10_2022-03-09_05-30-31.063956.pth.tar'
    checkpoint_path = None
    epochs = 5
    batch_size = 16
    gradient_accum_iter = 8
    learning_rate = 3e-5
    if checkpoint_path != None:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        encoder = checkpoint['encoder']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        print(f"start from epoch: {start_epoch} val_loss: {checkpoint['val_loss']}")
    else:
        print(f"epochs: {epochs}, batch_size: {batch_size}, accum_iter: {gradient_accum_iter}, lr={learning_rate}")
        start_epoch = 0
        encoder = Chexnet(pretrained=False).cuda()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min', 
            factor=0.2, 
            patience=1, 
            min_lr=1e-6
        )

    loss_fn = nn.BCELoss().cuda()

    train_loader = DataLoader(
        MultiLabelDataset('train', transform=train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        MultiLabelDataset('val', transform=evaluate_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    lrs = []
    train_plot = []
    val_plot = []

    print(f"start training from epoch {start_epoch}")
    for epoch in range(start_epoch, start_epoch + epochs):
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"Training epoch {epoch+1} with lr = {current_lr}...")
        train_loss = pretrain_encoder(encoder, train_loader, loss_fn, optimizer, gradient_accum_iter, epoch)
        print(f"Validating...")
        val_loss = validate_pretrained_encoder(encoder, val_loader, loss_fn)
        print(f"train_loss: {train_loss}, val_loss: {val_loss}")
        train_plot.append(train_loss)
        val_plot.append(val_loss)
        scheduler.step(val_loss)
        save_checkpoint(epoch, encoder, optimizer, scheduler, train_loss, val_loss)

    plot_history(train_plot, val_plot, lrs)

if __name__ == '__main__':
    main()