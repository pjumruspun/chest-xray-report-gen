import chula_rl as rl
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from actor_critic import ActorCriticNetwork
from tensorflow.keras.optimizers import Adam


class Agent(rl.policy.BasePolicy):
    def __init__(self, features_dim, batch_size=1, lr=3e-4, discount_factor=0.99, n_actions=2, fc1_dims=1024, fc2_dims=512):
        self.lr = lr
        self.features_dim = features_dim
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.n_actions = n_actions
        self.last_action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(
            n_actions=self.n_actions,
            features_dim=self.features_dim,
            batch_size=self.batch_size,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
        )

        self.actor_critic.compile(optimizer=Adam(learning_rate=self.lr))

    def step(self, state):
        state = tf.convert_to_tensor(state)

        # choose action, doesn't care about value
        _, probs = self.actor_critic(state)

        # sample action from probs generated from actor model
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        action = tf.squeeze(action)
        self.last_action = action

        return action

    def save_models(self):
        # print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def optimize_step(self, data):
        states, rewards, next_states, dones = data['s'], data['r'], data['ss'], data['done']

        # states and next_states should have shape of [batch_size, features]
        # rewards should have shape of [batch_size, 1]
        # done should have shape of [batch_size, 1]

        # to make sure these are tensors
        if states.ndim == 1:
            states = tf.convert_to_tensor([states], dtype=tf.float32)
            rewards = tf.convert_to_tensor([rewards], dtype=tf.float32)
            next_states = tf.convert_to_tensor([next_states], dtype=tf.float32)
        elif states.ndim == 2:
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # values and probs of current states
            values, probs = self.actor_critic(states)

            # critic outputs values of the next states
            next_values, _ = self.actor_critic(next_states)

            # squeeze for loss calculation purpose
            values = tf.squeeze(values)  # [[v1, v2, v3]] -> [v1, v2, v3]
            next_values = tf.squeeze(next_values)

            # log probs for loss calculation
            action_probs = tfp.distributions.Categorical(probs=probs)
            # log prob of the prob[self.last_action]
            log_probs = action_probs.log_prob(tf.squeeze(self.last_action))

            # if done, v(s') should equal to 0
            dones = np.logical_not(dones).astype(np.float32)
            dones = tf.squeeze(dones)

            # delta = r + gamma * v(s') * not done? - v(s)
            delta = rewards + self.discount_factor * next_values * dones - values

            # calculate loss
            actor_loss = -log_probs * delta
            critic_loss = delta ** 2

            total_loss = actor_loss + critic_loss

        # apply gradients
        gradient = tape.gradient(
            total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables
        ))
