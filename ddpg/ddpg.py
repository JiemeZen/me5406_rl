import gym
import numpy as np
import tensorflow as tf
from .actor import ActorNetwork
from .critic import CriticNetwork
from .replay_buffer import ReplayBuffer
from .ounoise import OUNoise

class DDPG():
    def __init__(self, env, batch_size=128, gamma=0.99):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
                
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.obs_dim, self.act_dim)
        self.critic_network = CriticNetwork(self.sess, self.obs_dim, self.act_dim)

        self.replay_buffer = ReplayBuffer(1000000)

        self.exploration_noise = OUNoise(self.act_dim)

    def train(self):
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        state_batch = np.resize(state_batch, [self.batch_size, self.obs_dim])
        action_batch = np.resize(action_batch, [self.batch_size, self.act_dim])

        # actor takes in state, here i calculate the predicted action by target network, (label) for my main network to chase
        next_action_batch = self.actor_network.predict_target(state_batch)
        # critic takes in state and action, i calculate q value, (label) for my critic main network to chase
        q_value_batch = self.critic_network.predict_target(state_batch, next_action_batch)

        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])
        
        y_batch = np.resize(y_batch, [self.batch_size])

        # training, train critic and actor
        self.critic_network.train(y_batch, state_batch, action_batch)
        new_actions = self.actor_network.predict(state_batch)
        action_grad = self.critic_network.action_gradient(state_batch, new_actions)
        self.actor_network.train(state_batch, action_grad)
        self.actor_network.update_target()
        self.critic_network.update_target()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.replay_buffer.size() > self.batch_size:
            self.train()


    def learn(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(100):
                action = self.actor_network.predict(np.reshape(state, (-1, self.obs_dim))) + self.exploration_noise.noise()
                new_state, reward, done, info = self.env.step(action)
                self.store_experience(state, action, reward, new_state, done)
                state = new_state
                episode_reward += reward
                if done:
                    self.exploration_noise.reset()
                    print("Episode {}: {} reward".format(episode, episode_reward))
                    break
        
        

