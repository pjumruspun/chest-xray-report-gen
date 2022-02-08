from sklearn.model_selection import train_test_split
from torch.nn.modules import loss
from pytorch_model import Encoder, Decoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from pytorch_dataset import ChestXRayDataset
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from configs import configs
import time
from datetime import datetime
from pytorch_tokenizer import create_tokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

alpha_c = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_freq = 5

# Train parameters
start_epoch = 2
epochs = 20
train_batch_size = 32
val_batch_size = 32
fine_tune_encoder = True

# Model parameters
emb_dim = 256  # dimension of word embeddings
attention_dim = 256  # dimension of attention linear layers
decoder_dim = 256  # dimension of decoder RNN
dropout = 0.5

# Do we save?
checkpoint_path = 'weights/pytorch_attention/' + 'checkpoint_2022-02-07_16-36-00.519912.pth.tar'
# checkpoint_path = None

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def topk_accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def main():
    tokenizer = create_tokenizer()
    loss_function = nn.CrossEntropyLoss().to(device)

    if checkpoint_path is None:
        start_epoch = 0
        encoder = Encoder()
        decoder = Decoder(attention_dim=attention_dim,
                                        embed_dim=emb_dim,
                                        decoder_dim=decoder_dim,
                                        vocab_size=len(tokenizer),
                                        dropout=dropout)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=configs['encoder_lr'])
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=configs['decoder_lr'])

    else:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        # epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        print(f"Time this checkpoint was saved: {checkpoint['time_saved']}")
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=configs['decoder_lr'])
        

    encoder.to(device)
    decoder.to(device)

    # Transform for DataLoader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = DataLoader(
        ChestXRayDataset('train', transform=transforms.Compose([normalize])), 
        batch_size=train_batch_size, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True)

    val_loader = DataLoader(
        ChestXRayDataset('val', transform=transforms.Compose([normalize])), 
        batch_size=val_batch_size, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        train(encoder, decoder, loss_function, train_loader, encoder_optimizer, decoder_optimizer, epoch)
        
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                loss_function=loss_function,
                                tokenizer=tokenizer)

        print(f"Bleu4: {recent_bleu4}")
        
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4)

def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4):
    cur_time = str(datetime.now())
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer,
             'bleu4': bleu4,
             'time_saved': cur_time}

    cur_time = cur_time.replace(' ', '_').replace(':', '-')
    filename = 'weights/pytorch_attention/checkpoint_' + cur_time + '.pth.tar'
    torch.save(state, filename)

def train(encoder, decoder, loss_function, train_loader, encoder_optimizer, decoder_optimizer, epoch):
    print(f"Training epoch {epoch+1}...")

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    # train mode (dropout and batchnorm is used)
    decoder.train()
    encoder.train()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        preds, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        preds = pack_padded_sequence(preds, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = loss_function(preds, targets)

        # Add doubly stochastic attention regularization
        # Read more at: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning#loss-function
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients, prevents gradient explosion
        # if grad_clip is not None:
        #     clip_gradient(decoder_optimizer, grad_clip)
        #     if encoder_optimizer is not None:
        #         clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = topk_accuracy(preds, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('\nBatch Time {batch_time.val:.3f} (avg={batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} (avg={data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} (avg={loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
    
    avg_train_loss = losses.avg

def validate(val_loader, encoder, decoder, loss_function, tokenizer):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    print("Validating...")
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm(val_loader)):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = loss_function(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = topk_accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('\nValidation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = [allcaps[j].tolist()]
                
                startseq = tokenizer.stoi['<startseq>']
                pad = tokenizer.stoi['<pad>']
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {startseq, pad}], img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            # for ref, hyp in zip(references, hypotheses):
            #     print([tokenizer.itos[idx] for idx in ref[0]])
            #     print([tokenizer.itos[idx] for idx in hyp])
            #     print('')

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    avg_val_loss = losses.avg

    return bleu4

if __name__ == "__main__":
    main()
