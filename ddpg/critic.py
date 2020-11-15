import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import numpy as np

class CriticNetwork():
    def __init__(self, sess, state_dim, act_dim, lr=0.001, tau=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.lr = lr
        self.tau = tau

        self.model = self.create_network('critic')
        self.target_model = self.create_network('critic_target')
        self.model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(lr))
        self.target_model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(lr), metrics=['accuracy'])
       
        self.action_gradients = tf.gradients(self.model.output, self.model.input[1])
        self.sess.run(tf.initialize_all_variables())

    def train(self, labels, state, action):
        self.model.train_on_batch([state, action], labels)

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
        x = Dense(64, activation='relu')(state)
        x = concatenate([Flatten()(x), action])
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='linear', kernel_initializer=RandomUniform())(x)
        return Model(inputs=[state, action], outputs=out, name=name)
