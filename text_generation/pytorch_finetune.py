import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_dataset import ChestXRayDataset
from pytorch_chest_xray_env import ChestXRayEnv
from pytorch_tokenizer import create_tokenizer
from pytorch_label import temperature_sampling
from pytorch_train import validate, save_checkpoint
from utils import decode_sequences
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 100
temperature = 1.25
eps = np.finfo(np.float32).eps.item()
print_freq = 20
tb = SummaryWriter()
lr = 1e-3

def finetune(encoder, decoder, train_loader, tokenizer, envs, epoch):
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
        
        # print([tokenizer.itos[e] for e in seqs[0]])
        # print(len(seqs[0]))
        # print(decoder.saved_actions)

        for j, (env, seq) in enumerate(zip(envs, seqs)):
            for action in seq:
                s, r, done, info = env.step(action)
                if done: 
                    break
                    
                # print(f"{tokenizer.itos[action]}, r: {r}")
                ep_rewards[j].append(r)

        # Append results
        final_rewards = [sum(ep_reward) for ep_reward in ep_rewards]
        loss, policy_loss, value_loss = finish_episode(decoder, ep_rewards, optimizer)
        reward_history.extend(final_rewards)

        if loss is not None:
            losses.append(loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            tb.add_scalar("Episode_reward", final_rewards[0], i)
            tb.add_scalar("Loss", loss, i)
            tb.add_scalar("Policy_loss", policy_loss, i)
            tb.add_scalar("Value_loss", value_loss, i)
            tb.add_text("Generated_sentence", decode_sequences(tokenizer, seqs)[0], i)

        if ((i + 1) % print_freq) == 0:
            print(f"\nloss avg: {np.mean(losses):.4f}\tlast {print_freq} loss avg: {np.mean(losses[-print_freq:])}")


# def finish_episode(decoder, ep_rewards, optimizer):
#     gamma = 0.99
#     policy_loss = []
#     value_loss = []

#     # Batch size 1 for now
#     model_rewards = ep_rewards[0]
#     print(f"Model rewards: {model_rewards}")
#     # return(t) = return(t-1) + gamma * 
#     returns = get_expected_returns(model_rewards, gamma, normalize=True)
#     print(f"Returns: {[round(g.item(), 2) for g in returns]}")

#     for (log_prob, value), g in zip(decoder.saved_actions, returns):
#         # Minimize -log(pi) * (return - v)
#         # -log_prob always positive, but delta sometimes negative and make total policy loss negative
#         # How to fix this????
#         # delta = torch.abs(g - value.item())
#         delta = g - value.item()
#         policy_loss.append(-torch.squeeze(log_prob) * delta)
#         print(f"-log_prob:{round(-torch.squeeze(log_prob).item(), 2)}\tg: {g}\tvalue: {value.item()}\tdelta: {round(delta.item(), 2)}")

#         # Try minimize -log(pi) * v?
#         # policy_loss.append(-torch.squeeze(log_prob) * value)
#         # print(f"-log_prob:{round(-torch.squeeze(log_prob).item(), 2)}\tg: {g}\tvalue: {value.item()}")

#         # Huber loss, less fluctuative than squared loss
#         v_loss = F.smooth_l1_loss(value, torch.tensor([g]).to(device))
#         value_loss.append(v_loss)

#     optimizer.zero_grad()
#     # print(f"{policy_loss=}, {value_loss=}")
#     if len(policy_loss) != 0 and len(value_loss) != 0:
#         total_policy_loss = torch.stack(policy_loss).mean()
#         total_value_loss = torch.stack(value_loss).mean()
#         loss = total_policy_loss + total_value_loss
#         loss.backward()
#         optimizer.step()
#         # print(f"policy_loss={[round(pl.item(), 2) for pl in policy_loss]}\nvalue_loss={[round(vl.item(), 2) for vl in value_loss]}\n{round(loss.item(), 2)=}")

#         decoder.saved_actions = []
#         return loss.item(), total_policy_loss.item(), total_value_loss.item()
#     else:
#         return None

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

def finish_episode(decoder, ep_rewards, optimizer):
    gamma = 0.99
    total_policy_losses = torch.tensor([0.0], device=device, requires_grad=True)
    total_value_losses = torch.tensor([0.0], device=device, requires_grad=True)
    # Truncate saved_actions to a proper length

    # Convert to padded tensors
    model_rewards, r_lengths = ragged_list_to_tensor(ep_rewards, default_value=float('nan'))
    saved_actions, s_lengths = ragged_list_to_tensor(decoder.saved_actions, default_value=torch.tensor([float('nan'), float('nan')], device=device, requires_grad=True))
    # print(f"Model rewards: {model_rewards}")
    # print(f"Lengths: {lengths}")

    model_rewards.to(device)
    saved_actions.to(device)

    # Sort by size, descending
    r_lengths, r_sort_ind = r_lengths.sort(dim=0, descending=True)
    s_lengths, s_sort_ind = s_lengths.sort(dim=0, descending=True)
    model_rewards = model_rewards[r_sort_ind] # (batch_size, max_seq_len)
    saved_actions = saved_actions[s_sort_ind] # (batch_size, max_seq_len)

    # print(f"Lengths after sorting: {lengths}")
    # print(f"{r_lengths}\t{s_lengths}")
    # print(f"{r_sort_ind=}\t{s_sort_ind}")
    # print(f"Model rewards after sorting: {model_rewards}")

    # Transpose so we can loop through each token
    # print(model_rewards.shape)
    # print(saved_actions.shape)
    model_rewards = torch.t(model_rewards) # (max_seq_len, batch_size)
    saved_actions = torch.transpose(saved_actions, 0, 1) # (max_seq_len, batch_size)
    # print(model_rewards.shape)
    # print(saved_actions.shape)

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
        policy_loss = log_probs * adv

        # Calculate value losses
        # Huber loss, less fluctuative than squared loss
        value_loss = F.smooth_l1_loss(values, accu_rewards[:batch_size])
        

        # print(f"{i=}\t{batch_size=}\n{log_probs=}\n{values=}\n{values_next=}\n{rewards=}\n{adv=}")
        # print(f"{value_loss=}")

        # L = avg(log_prob * adv)
        total_policy_losses = total_policy_losses + torch.sum(policy_loss)
        total_value_losses = total_value_losses + torch.sum(value_loss)
        # print(policy_loss)
        # print(total_policy_losses)
        # print(value_loss)
        # print(total_value_losses)

        total_batch_size += batch_size
        
        
    
    optimizer.zero_grad()
    # policy_losses = torch.Tensor(policy_losses)
    # value_losses = torch.Tensor(value_losses)

    # print(f"{policy_loss=}, {value_loss=}")
    total_policy_losses = total_policy_losses / total_batch_size
    total_value_losses = total_value_losses / total_batch_size
    loss = total_policy_losses + total_value_losses
    loss.backward()
    optimizer.step()
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
    train_batch_size = 12
    epochs = 1
    tokenizer = create_tokenizer()
    sparse_reward = False
    num_instances = train_batch_size

    # Models
    checkpoint_path = "weights\pytorch_attention\checkpoint_2021-12-21_03-32-00.004925.pth.tar"
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
        batch_size=32, 
        shuffle=True, 
        num_workers=1, 
        pin_memory=True)
    
    loss_function = nn.CrossEntropyLoss().to(device)

    # Create env (Parallel)
    envs = []
    for _ in range(num_instances):
        env = ChestXRayEnv(tokenizer, max_len, sparse_reward=sparse_reward)
        env.reset()
        envs.append(env)

    for epoch in range(epochs):
        finetune(encoder, decoder, train_loader, tokenizer, envs, epoch)
        recent_bleu4 = validate(val_loader, encoder, decoder, loss_function, tokenizer)
        print(f"Bleu4: {recent_bleu4}")
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4)
        

if __name__ == "__main__":
    main()