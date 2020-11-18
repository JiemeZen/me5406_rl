import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import numpy as np

class ActorNetwork():
    """ Actor network for ddpg algorithm.

    A tf session needs to be created, follow by defining the dimensions for observation
    and action space, as well as batch size. Two optional parameters are the learning rate
    and tau.
    """
    def __init__(self, sess, state_dim, act_dim, batch_size, lr=0.0001, tau=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.act_dim = act_dim  
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.time_step = 0

        self.saver = tf.train.Saver()  # tf Saver to save the network session.

        self.model = self.create_network("actor")  # Create the main network.
        self.target_model = self.create_network("target_actor")  # Create the target network.

        self.action_gradients = tf.placeholder(tf.float32, [None, self.act_dim])  # Create a tf placeholder for gradient optimization.
        self.parameter_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradients)
        gradients = zip(self.parameter_gradients, self.model.trainable_weights)
        self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(gradients)

        self.sess.run(tf.initialize_all_variables())  # Initialize all placeholders, variables.

    def train(self, state, action_gradient):
        """
        Train the actor network
        """
        self.time_step += 1  # Log the global timestep.
        self.sess.run(self.optimizer, feed_dict={
            self.model.input: state,
            self.action_gradients: action_gradient
        })

    def update_target(self):
        """
        Update target network with the formula:
        target_w = tau * main_w + (1 - tau) * target_w
        """
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state):
        """
        Predict action to take given the state.
        """
        return self.model.predict(state)

    def predict_target(self, state_batch):
        """
        Predict action batch given state batch.
        """
        return self.target_model.predict(state_batch)

    def create_network(self, name):
        """
        Create the neural network. (Functional)
        
        Input: Observation space (28 dim)
        Layers: 2 Hidden FC with RELU
        Output: Action Space (8 dim)
        """
        inp = Input(shape=[self.state_dim])
        x = Dense(64, activation='relu')(inp)  # Attach Dense layer above inp
        x = Dense(64, activation='relu')(x)  # Attach Dense layer on top of the previous later
        out = Dense(self.act_dim,activation='tanh', kernel_initializer=RandomUniform())(x)
        return Model(inputs=inp, outputs=out, name=name)

    def create_optimizer(self):
        """
        Create custom optimizer.
        """
        self.action_gradients = tf.placeholder(tf.float32, [None, self.act_dim])
        self.parameter_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradients)
        gradients = zip(self.parameter_gradients, self.model.trainable_weights)
        return tf.train.AdamOptimizer(self.lr).apply_gradients(gradients)

    def load_network(self, path=None):
        """
        Load saved network from ./actor folder
        """
        checkpoint = tf.train.get_checkpoint_state(path + "/actor")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("[INFO] Successfully loaded: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("[ERROR] Unable to load network!")

    def save_network(self, path):
        """
        Save network to ./actor folder.
        """
        print('[INFO] Saving ActorNetwork to {}'.format(path + "/actor/"))
        self.saver.save(self.sess, path + "/actor/actor-network", global_step=self.time_step)

    