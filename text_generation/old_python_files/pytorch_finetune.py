import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pytorch_chest_xray_env import ChestXRayEnv
from pytorch_dataset import ChestXRayDataset
from pytorch_label import temperature_sampling
from pytorch_tokenizer import create_tokenizer
from pytorch_train import save_checkpoint, validate
from utils import decode_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 100
temperature = 1.0
eps = np.finfo(np.float32).eps.item()
print_freq = 20
accumulation_iter = 4
tb = SummaryWriter()

def finetune(encoder, decoder, train_loader, tokenizer, envs, epoch, lr):
    encoder.train()
    decoder.train()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    reward_history = []
    losses = []
    policy_losses = []
    value_losses = []

    for i, (imgs, caps, _) in enumerate(tqdm(train_loader)):
        # Actual batch size might not be equal to num_stances at the end of the dataset
        batch_size = imgs.shape[0]

        # Reset the environment and set ground truth
        for j in range(batch_size):
            gt = caps[j].numpy()
            envs[j].reset()
            envs[j].set_ground_truth(gt)
        
        # Variables before running an opisode
        dones = [False] * batch_size
        res = [[] for _ in range(batch_size)]
        ep_rewards = [[] for _ in range(batch_size)]
        
        seqs, _ = temperature_sampling(
            encoder, decoder, tokenizer, images=imgs, 
            temperature=temperature, max_len=max_len, rl=True)

        for j, (env, seq) in enumerate(zip(envs, seqs)):
            for action in seq:
                if not dones[j]:
                    s, r, done, info = env.step(action)
                    dones[j] = done
                    last_nonzero_idx = np.max(s.nonzero())
                    idx = int(s[last_nonzero_idx])
                    res[j].append(tokenizer.itos[idx])
                    ep_rewards[j].append(r)

                if all(dones):
                    break
        
        # Append results
        final_rewards = [sum(ep_reward) for ep_reward in ep_rewards]
        loss, policy_loss, value_loss = finish_episode(i, len(train_loader), decoder, ep_rewards, optimizer)
        reward_history.extend(final_rewards)

        if loss is not None:
            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            ith = i + len(train_loader) * epoch
            tb.add_scalar("Episode_reward", np.mean(final_rewards), ith)
            tb.add_scalar("Loss", loss, ith)
            tb.add_scalar("Policy_loss", policy_loss, ith)
            tb.add_scalar("Value_loss", value_loss, ith)
            tb.add_text("Generated_sentence", decode_sequences(tokenizer, seqs)[0], ith)


def ragged_list_to_tensor(nested_ls, default_value):
    # only works on 2D nested list
    lengths = [len(e) for e in nested_ls]
    max_list_len = max(lengths)
    for ls in nested_ls:
        to_fill = max_list_len - len(ls)
        ls.extend([default_value for _ in range(to_fill)])

    lengths = torch.tensor(lengths, device=device)
    res = torch.tensor(nested_ls, device=device)

    return res, lengths


def finish_episode(i, loader_length, decoder, ep_rewards, optimizer):
    gamma = 0.99
    total_policy_losses = torch.tensor([0.0], device=device, requires_grad=True)
    total_value_losses = torch.tensor([0.0], device=device, requires_grad=True)
    # Truncate saved_actions to a proper length

    # Convert to padded tensors
    model_rewards, r_lengths = ragged_list_to_tensor(ep_rewards, default_value=float('nan'))
    saved_actions, s_lengths = ragged_list_to_tensor(decoder.saved_actions, default_value=torch.tensor([float('nan'), float('nan')], device=device, requires_grad=True))

    model_rewards.to(device)
    saved_actions.to(device)

    # Sort by size, descending
    r_lengths, r_sort_ind = r_lengths.sort(dim=0, descending=True)
    s_lengths, s_sort_ind = s_lengths.sort(dim=0, descending=True)
    model_rewards = model_rewards[r_sort_ind] # (batch_size, max_seq_len)
    saved_actions = saved_actions[s_sort_ind] # (batch_size, max_seq_len)

    # Transpose so we can loop through each token
    model_rewards = torch.t(model_rewards) # (max_seq_len, batch_size)
    saved_actions = torch.transpose(saved_actions, 0, 1) # (max_seq_len, batch_size)

    # Accumulated reward batched
    accu_rewards = torch.zeros(len(ep_rewards), device=device)
    total_batch_size = 0

    for i, (log_prob_values, rewards)  in enumerate(zip(saved_actions[:-1], model_rewards[:-1])):

        batch_size = sum([l > i+1 for l in r_lengths]).item() # i+1 because we use values_next
        accu_rewards = accu_rewards + gamma * rewards

        # Get rid of padded values
        log_probs, values = torch.t(log_prob_values)[:,:batch_size]
        _, values_next = torch.t(saved_actions[i+1])[:,:batch_size]
        rewards = rewards[:batch_size]

        # Calculate advantage
        adv = rewards + gamma * values_next - values

        # Calculate policy losses
        policy_loss = -log_probs * adv

        # Calculate value losses
        # Huber loss, less fluctuative than squared loss
        value_loss = F.smooth_l1_loss(values, accu_rewards[:batch_size])

        # L = avg(log_prob * adv)
        total_policy_losses = total_policy_losses + torch.sum(policy_loss)
        total_value_losses = total_value_losses + torch.sum(value_loss)
        
        # Total batch size as divisor
        total_batch_size += batch_size
        
    total_policy_losses = total_policy_losses
    total_value_losses = total_value_losses
    loss = total_policy_losses + total_value_losses
    loss.backward()

    # Gradient accumulation
    if (i + 1) % accumulation_iter == 0 or (i + 1) == loader_length:
        optimizer.step()
        optimizer.zero_grad()

    # print(f"policy_loss={[round(pl.item(), 2) for pl in total_policy_losses]}\nvalue_loss={[round(vl.item(), 2) for vl in total_value_losses]}\n{round(loss.item(), 2)=}")
    
    return loss.item(), total_policy_losses.item(), total_value_losses.item()


def get_expected_returns(rewards, gamma, normalize=True):
    returns = []
    g = 0 # Terminal step return = 0
    for reward in rewards[::-1]:
        g = reward + gamma * g
        returns.insert(0, g)
    
    returns = torch.tensor(returns)
    if normalize and len(returns) > 1: # Prevent division by zero
        returns = (returns - returns.mean()) / (returns.std() + eps)

    return returns


def main():
    parser = argparse.ArgumentParser(description='Finetune existing model with actor critic')
    parser.add_argument('--learning-rate', '-lr', type=float, help='Learning rate')
    parser.add_argument('--epoch', '-e', type=int, help='Learning rate')
    parser.add_argument('--metrics', '-m', type=str, help='Metrics for reward module', choices=['f1', 'recall', 'precision'])
    parser.add_argument('--batch_size', '-bs', type=int, help='Batch size for both training and validating')
    parser.add_argument('--checkpoint', '-c', type=str, help='Relative checkpoint path, should be pth.tar file')
    args = parser.parse_args()

    train_batch_size = args.batch_size
    val_batch_size = args.batch_size
    epochs = args.epoch
    tokenizer = create_tokenizer()
    sparse_reward = False
    num_instances = train_batch_size

    # Models
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    decoder_optimizer = checkpoint['decoder_optimizer']
    epoch = checkpoint['epoch'] + 1
    print(f"Using epoch {epoch} model...")
    encoder.to(device)
    decoder.to(device)

    # DataLoader
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
        shuffle=False, 
        num_workers=1, 
        pin_memory=True)
    
    loss_function = nn.CrossEntropyLoss().to(device)

    # Create env (Parallel)
    envs = []
    for _ in range(num_instances):
        env = ChestXRayEnv(tokenizer, max_len, sparse_reward=sparse_reward, metrics=args.metrics)
        env.reset()
        envs.append(env)

    for epoch in range(epochs):
        print(f"Finetuning epoch {epoch+1}")
        finetune(encoder, decoder, train_loader, tokenizer, envs, epoch, args.learning_rate)
        # recent_bleu4 = validate(val_loader, encoder, decoder, loss_function, tokenizer)
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, -1)
        

if __name__ == "__main__":
    main()
