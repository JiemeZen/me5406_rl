import gym
import numpy as np
import tensorflow as tf
from .actor import ActorNetwork
from .critic import CriticNetwork
from .replay_buffer import ReplayBuffer
from .ounoise import OUNoise

class DDPG():
    """
    Deep deterministic policy gradient algorithm
    """
    def __init__(self, env, batch_size=64, gamma=0.99, tensorboard_log=None):
        """Intialize an instance of the algo.
        
        An OpenAI environment is expected. Batch size, gamma and tensorboard logs are optional parameters.
        """
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma

        self.sess = tf.InteractiveSession()  # Create the tf session.
        
        if tensorboard_log is not None:
            self.summary_ops, self.summary_vars = self.build_summaries() # Initialize tensorboard summaries.
            self.writer = tf.summary.FileWriter(tensorboard_log, self.sess.graph)  # Intialize the writer.

        self.actor_network = ActorNetwork(self.sess, self.obs_dim, self.act_dim, self.batch_size)  # Create both the main and target actor network.
        self.critic_network = CriticNetwork(self.sess, self.obs_dim, self.act_dim, self.batch_size, self.writer)  # Create  both the main and target critic network.

        self.replay_buffer = ReplayBuffer(1000000)  # Intialize the replay buffer with size of 1m.
        self.exploration_noise = OUNoise(self.act_dim)  # Intialize OUNoise.

    def __train(self):
        """
        Train the RL agent.
        """
        minibatch = self.replay_buffer.get_batch(self.batch_size)  # Randomly sample a batch of experiences.
        state_batch = np.array([data[0] for data in minibatch])
        action_batch = np.array([data[1] for data in minibatch])
        reward_batch = np.array([data[2] for data in minibatch])
        next_state_batch = np.array([data[3] for data in minibatch])
        done_batch = np.array([data[4] for data in minibatch])

        # Forward pass to actor network with next_state to get next_action.
        next_action_batch = self.actor_network.predict_target(next_state_batch)

        # Generate target Q with critic network.
        q_value_batch = self.critic_network.predict_target(next_state_batch, next_action_batch)

        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])
        
        y_batch = np.reshape(y_batch, (self.batch_size, 1))

        # Train main actor and critic network.
        self.critic_network.train(y_batch, state_batch, action_batch)
        new_actions = self.actor_network.predict(state_batch)
        action_grad = self.critic_network.action_gradient(state_batch, new_actions)
        self.actor_network.train(state_batch, action_grad)

        # Soft update the actor and critic network.
        self.actor_network.update_target()
        self.critic_network.update_target()


    def learn(self, episodes):
        """
        Learn for num episodes.
        """
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(5000):
                action = self.actor_network.predict(np.reshape(state, (-1, self.obs_dim))) + self.exploration_noise.noise()  # Predict the action to take, perturbed by OUNoise.
                new_state, reward, done, info = self.env.step(action)  # Advance the simulation
                self.replay_buffer.add(                     # Add a step of experiennce to the replay buffer
                    np.reshape(state, (self.obs_dim,)),  
                    np.reshape(action, (self.act_dim,)),
                    reward,
                    np.reshape(new_state, (self.obs_dim,)),
                    done)

                if self.replay_buffer.size() > self.batch_size:
                    self.__train()

                state = new_state
                episode_reward += reward
                if done:
                    self.exploration_noise.reset()
                    summary = self.sess.run(self.summary_ops, feed_dict={
                        self.summary_vars[0]: episode_reward
                    })
                    self.writer.add_summary(summary, episode)  # Log episode reward in TensorBoard.
                    self.writer.flush()
                    # print("Episode {}: {} reward".format(episode, episode_reward))
                    break

    def save(self, path):
        """
        Save both the actor and critic network.
        """
        self.actor_network.save_network(path)
        self.critic_network.save_network(path)

    def build_summaries(self):
        """
        Initialize tensorboard.
        """
        episode_reward = tf.Variable(0)
        tf.summary.scalar("Episode reward", episode_reward)
        summary_vars = [episode_reward]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    def load_network(self, path=None):
        """
        Load actor and critic network.
        """
        self.actor_network.load_network(path=path)
        self.critic_network.load_network(path=path)
        
    
