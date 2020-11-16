import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import numpy as np

class CriticNetwork():
    def __init__(self, sess, state_dim, act_dim, batch_size, writer, lr=0.001, tau=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.time_step = 0

        self.saver = tf.train.Saver()
        writer.add_graph(self.sess.graph)
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log)

        self.model = self.create_network('critic')
        self.target_model = self.create_network('critic_target')
        self.model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(lr), metrics=['accuracy'])
        self.target_model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(lr), metrics=['accuracy'])
       
        self.action_gradients = tf.gradients(self.model.output, self.model.input[1])
        self.sess.run(tf.initialize_all_variables())

    def train(self, labels, state, action):
        self.time_step += 1
        self.model.train_on_batch([state, action], labels)
        # self.model.fit(x=[state, action], y=labels, batch_size=self.batch_size, callbacks=[self.tensorboard_callback])

    def action_gradient(self, states, actions):
        return self.sess.run(self.action_gradients, feed_dict={
            self.model.input[0]: states,
            self.model.input[1]: actions
        })[0]

    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict_target(self, state_batch, action_batch):
        return self.target_model.predict([state_batch, action_batch])

    def create_network(self, name):
        state = Input(shape=[self.state_dim])
        action = Input(shape=[self.act_dim])
        x = Dense(64, activation='relu', name="Critic_Hidden_1")(state)
        x = concatenate([Flatten()(x), action])
        x = Dense(64, activation='relu', name="Critic_Hidden_2")(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform(), name="Critic_Output")(x)
        return Model(inputs=[state, action], outputs=out, name=name)

    def load_network(self, path=None):
        checkpoint = tf.train.get_checkpoint_state(path + "/critic")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("[INFO] Successfully loaded: {}".format(checkpoint.model_checkpoint_path))
        else:
            print("[ERROR] Unable to load network!")

    def save_network(self, path):
        print('[INFO] Saving CriticNetwork to {}'.format(path + "/critic/"))
        self.saver.save(self.sess, path + "/critic/critic-network", global_step=self.time_step)
