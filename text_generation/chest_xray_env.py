import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from chexpert import calculate_reward

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

    def __init__(self, bos_token, eos_token, tokenizer, vocab_size, max_len):
        self.bos = bos_token
        self.eos = eos_token
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.given_reward = False
        self.ground_truth = None
        self.idx = 0

        # Action is choosing vocab
        self.action_space = spaces.Discrete(vocab_size)

        # Huge observation space of length=max_len
        # each space has vocab_size possibilities
        self.observation_space = spaces.MultiDiscrete([vocab_size] * max_len)

        self.seed()
        self.state = None
    
    def reset(self):
        self.given_reward = False
        self.state = np.zeros((self.max_len))
        self.state[0] = self.tokenizer.word_index[self.bos]
        self.idx = 0
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
                "Please set ground truth for sake of reward calculation"
                "before stepping in this episode.\n"
                "Example: env.set_ground_truth(padded_ground_truth)"
                )

        # Increase index
        self.idx += 1

        # Append the action
        self.state[self.idx] = action

        # Check if done
        done = bool(
            self.idx >= self.max_len # Reached max len
            or action == self.tokenizer.word_index[self.eos]
        )

        if not done:
            reward = 0.0
        else:
            if not self.given_reward:
                # Give reward
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
            
        return self.state, reward, done, {}

if __name__ == '__main__':
    from configs import configs
    from utils import get_max_report_len
    from tokenizer import cnn_rnn_tokenizer
    import random

    max_len = get_max_report_len()
    tokenizer = cnn_rnn_tokenizer()
    vocab_size = len(tokenizer.word_index)
    env = ChestXRayEnv(configs['START_TOK'], configs['STOP_TOK'], tokenizer, vocab_size, max_len)
    env.reset()

    ground_truth = tokenizer.texts_to_sequences(
        ['lungs are overall hyperexpanded with flattening of the diaphragms . no focal consolidation . no pleural effusions or pneumothoraces . heart and mediastinum of normal size and contour . degenerative changes in the thoracic spine . hyperexpanded but clear lungs .']
    )[0]

    ground_truth = np.array(ground_truth)
    # print(ground_truth)
    ground_truth = np.pad(ground_truth, (0, max_len - ground_truth.shape[0]), 'constant')

    # print(ground_truth)
    print(max_len)
    print(ground_truth.shape[0])

    env.set_ground_truth(ground_truth)

    done = False

    i = 0
    while not done:
        if i < 20:
            a = random.randint(0, vocab_size)
        else:
            a = tokenizer.word_index[configs['STOP_TOK']]
        s, r, done, info = env.step(a)
        

        i += 1
    print(f"{s}, {r}, {done}")