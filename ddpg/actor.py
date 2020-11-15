import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
import numpy as np


class ActorNetwork():
    def __init__(self, sess, state_dim, act_dim, lr=0.0001, batch_size=64, tau=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.act_dim = act_dim  
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau

        self.model = self.create_network("actor")
        print(self.model.summary())
        self.target_model = self.create_network("target_actor")

        self.action_gradients = tf.placeholder(tf.float32, [None, self.act_dim])
        self.parameter_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradients)
        gradients = zip(self.parameter_gradients, self.model.trainable_weights)
        self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(gradients)

        self.sess.run(tf.initialize_all_variables())

    def train(self, state, action_gradient):
        self.sess.run(self.optimizer, feed_dict={
            self.model.input: state,
            self.action_gradients: action_gradient
        })

    def update_target(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def predict(self, state):
        return self.model.predict(state)

    def predict_target(self, state_batch):
        return self.target_model.predict(state_batch)

    def create_network(self, name):
        inp = Input(shape=[self.state_dim])
        x = Dense(64, activation='relu')(inp)
        x = Dense(64, activation='relu')(x)
        out = Dense(self.act_dim,activation='tanh', kernel_initializer=RandomUniform())(x)
        return Model(inputs=inp, outputs=out, name=name)

    def create_optimizer(self):
        self.action_gradients = tf.placeholder(tf.float32, [None, self.act_dim])
        self.parameter_gradients = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradients)
        gradients = zip(self.parameter_gradients, self.model.trainable_weights)
        return tf.train.AdamOptimizer(self.lr).apply_gradients(gradients)

    