import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from pytorch_chexpert import calculate_reward
from dataset import get_train_materials
from label import prep_models

class ChestXRayEnv(gym.Env):
    """
    Observation:
        [/] Report sequences
    
    Actions:
        [/] Word to append ([0, vocab_size-1])

    Reward:
        [/] 0: If the action is not EOS
        [/] Chexpert(observation): If the action is EOS

        TODO: [x] Reduce the sparseness of the rewards
        Maybe gen reward after each full stop?
        But how to handle model to not repeatedly spam full stops?

    Starting State:
        [/] Sequence that contains nothing but BOS token

    Episode Termination:
        [/] EOS token is generated
        [/] Max sequence length is reached
    """

    def __init__(self, tokenizer, max_len, sparse_reward=True):
        self.bos = tokenizer.stoi['<startseq>']
        self.eos = tokenizer.stoi['<endseq>']
        self.fullstop = tokenizer.stoi['.']
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.given_reward = False
        self.ground_truth = None
        self.idx = 0
        self.last_metric = 0.0
        self.sparse_reward = sparse_reward

        # Action is choosing vocab
        self.action_space = spaces.Discrete(len(tokenizer.stoi))

        # Huge observation space of length=max_len
        # each space has vocab_size possibilities
        self.observation_space = spaces.MultiDiscrete([len(tokenizer.stoi)] * max_len)

        self.seed()
        self.state = None
    
    def reset(self):
        self.given_reward = False
        self.state = np.zeros((self.max_len))
        self.state[0] = self.tokenizer.stoi[self.bos]
        self.idx = 0
        self.last_metric = 0.0
        return self.state

    def set_ground_truth(self, padded_ground_truth):
        """
        Set ground truth in format of padded sequence
        """
        self.ground_truth = padded_ground_truth


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Append action to current state, if the action is EOS, generate reward
        """
        if type(self.ground_truth) == type(None):
            # Can't calculate possible reward if
            # ground truth hasn't been set
            raise ValueError(
                "Please set ground truth for sake of reward calculation "
                "before stepping in this episode.\n"
                "Example: env.set_ground_truth(padded_ground_truth)"
                f"Additional details:\n"
                f"\t{self.idx=}\n"
                f"\t{self.state=}\n"
                f"\t{self.ground_truth=}\n"
                )

        
        # Increase index
        self.idx += 1

        # Append the action
        self.state[self.idx] = action

        # Check if done
        # print(f"{self.idx=}")
        done = bool(
            self.idx >= self.max_len - 1 # Reached max len
            or action == self.tokenizer.stoi[self.eos]
        )

        if self.sparse_reward:
            if not done:
                # No reward
                reward = 0.0
            else:
                if not self.given_reward:
                    # Give reward
                    # print(self.ground_truth.shape)
                    # print(self.state.shape)
                    precision, recall, f1 = calculate_reward(self.ground_truth, self.state, self.tokenizer)
                    reward = f1

                    # Reset ground truth
                    self.ground_truth = None
                else:
                    # Already given reward and done before this
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
        else:
            # print(f"word: {self.tokenizer.index_word[action]}")
            # Less sparse reward
            # Check if the generated word was a full stop or is done
            if action == self.fullstop or done:
                precision, recall, f1 = calculate_reward(self.ground_truth, self.state, self.tokenizer)
                # print(f"f1: {f1}, self.last_metric: {self.last_metric}")
                reward = f1 - self.last_metric
                # print(f"reward: {reward}")
                self.last_metric = f1
                
                # Additionally, if done, reset ground truth
                if done:
                    self.ground_truth = None
            else:
                # No reward
                reward = 0.0

        return self.state, reward, done, {}

def run_env_with_test():
    from configs import configs
    from utils import get_max_report_len
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from pytorch_dataset import ChestXRayDataset
    from pytorch_tokenizer import create_tokenizer
    from pytorch_label import temperature_sampling
    import torchvision.transforms as transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sparse_reward = False
    num_instances = 16
    temperature = 1.5

    max_len = get_max_report_len()

    # For real data and stuff need to create env
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = DataLoader(
        ChestXRayDataset('test', transform=transforms.Compose([normalize])),
        batch_size=num_instances,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Tokenizer
    tokenizer = create_tokenizer()

    # Models
    checkpoint_path = "weights\pytorch_attention\checkpoint_2021-12-14_01-07-09.016551.pth.tar"
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    epoch = checkpoint['epoch'] + 1
    print(f"Using epoch {epoch} model...")
    encoder.to(device)
    decoder.to(device)
    
    # Create env (Parallel)
    envs = []
    for _ in range(num_instances):
        env = ChestXRayEnv(tokenizer, max_len, sparse_reward=sparse_reward)
        env.reset()
        envs.append(env)

    reward_history = []

    for i, (imgs, caps, caplens, _) in enumerate(tqdm(test_loader)):
        # Actual batch size might not be equal to num_stances at the end of the dataset
        batch_size = imgs.shape[0]

        # Reset the environment and set ground truth
        for i in range(batch_size):
            gt = caps[i].numpy()
            envs[i].reset()
            envs[i].set_ground_truth(gt)
        
        # Variables before running an opisode
        dones = [False] * batch_size
        res = [[] for _ in range(batch_size)] 
        ep_rewards = [[] for _ in range(batch_size)]
        
        seqs, _ = temperature_sampling(encoder, decoder, tokenizer, images=imgs, temperature=temperature, max_len=max_len)

        for j, (env, seq) in enumerate(zip(envs, seqs)):
            for action in seq:
                s, r, done, info = env.step(action)
                if done: 
                    break
                # print(f"{tokenizer.itos[action]}, r: {r}")
                ep_rewards[j].append(r)

        # Append results
        final_rewards = [sum(ep_reward) for ep_reward in ep_rewards]
        print(f"{final_rewards=}")
        reward_history.extend(final_rewards)

    
    print(f"average_reward = {np.mean(reward_history)}")
    plt.hist(reward_history, bins=10)
    plot_name = 'reward_history_parallel'
    if sparse_reward:
        plot_name += f'_sparse_{str(num_instances)}.png'
    else:
        plot_name += f'_dense_{str(num_instances)}.png'

    plt.savefig(plot_name)
    plt.show()

def run_random():
    from configs import configs
    from utils import get_max_report_len
    import tensorflow as tf
    import random

    max_len = get_max_report_len()

    # For real data and stuff need to create env
    _, _, tokenizer, _, vocab_size = get_train_materials()

    # Create env
    env = ChestXRayEnv(configs['START_TOK'], configs['STOP_TOK'], tokenizer, vocab_size, max_len)
    env.reset()


    # Need to set ground truth after reset
    ground_truth = tf.ones((max_len))
    env.set_ground_truth(ground_truth)
    
    # Variables before running an opisode
    done = False
    res = []

    i = 0
    done = False
    while not done:

        # Random action
        pred_id = random.randint(1, vocab_size-1)
        if i > 10:
            pred_id = 9 # eos

        word = tokenizer.index_word[pred_id]
        res.append(word)
        a = pred_id
        s, r, done, info = env.step(a)
        i += 1
        
    print("final s, r, done:")
    print(f"{s}, {r}, {done}")
    print(f"res: {res}")

if __name__ == '__main__':
    run_env_with_test()