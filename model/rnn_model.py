import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

learning_rate = 0.0001
batch_size = 10
training_iters = 1000

class RNNModel(object):
    def __init__(self, n_input, n_steps, n_hidden, n_classes, keep_prob=1.0, is_training=True, config=None):
        self.n_input, self.n_steps, self.n_hidden, self.n_classes = n_input, n_steps, n_hidden, n_classes
        self.keep_prob = keep_prob
        self.is_training = is_training

        if config is None:
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
        else:
            self.config = config

    def RNN(self, x, weights, biases):
        x = tf.unstack(x, self.n_steps, 1)
        x_dropout = [tf.nn.dropout(x_i, self.keep_prob) for x_i in x]
        cell = rnn.GRUCell(self.n_hidden)
        outputs, states = rnn.static_rnn(cell, x_dropout, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    def build_model(self, input_tensor):
        # self.x = tf.stack(input_tensors, axis=1)
        self.x = tf.reshape(input_tensor, [-1, self.n_steps, self.n_input])
        # self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.p = tf.placeholder("float", shape=())

        W = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        b = tf.Variable(tf.random_normal([self.n_classes]))

        self.pred = self.RNN(self.x, W, b)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.output_label = tf.argmax(self.pred, 1)
        self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def train(self):
        init = tf.global_variables_initializer()
        with tf.Session(config=self.config) as sess:
            sess.run(init)
            step = 1
            while step * batch_size <= training_iters:
                pass

