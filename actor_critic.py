import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense


class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, features_dim, batch_size=1, fc1_dims=1024, fc2_dims=512,
                 name='actor_critic', ckpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.features_dim = features_dim
        self.batch_size = batch_size
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = ckpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        # shares same hidden layers
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')

        # value function estimator, no activation
        self.critic = Dense(1, activation=None)

        # discrete action space
        self.actor = Dense(self.n_actions, activation='softmax')

    def call(self, state):
        # correct the dimension
        if state.ndim == 1:
            state = tf.convert_to_tensor([state], dtype=tf.float32)

        if state.shape != (self.batch_size, self.features_dim):
            raise ValueError(
                f"expected state shape to be ({self.batch_size}, {self.features_dim}), got: {state.shape}")

        x = self.fc1(state)
        x = self.fc2(x)

        value = self.critic(x)
        probs = self.actor(x)

        return value, probs
