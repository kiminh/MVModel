import tensorflow as tf
import numpy as np

import vgg19_trainable as vgg19

class CNNModel(object):
    def __init__(self, is_training=True):
        self.is_training = is_training

    def build_model(self, input_images, vgg_path):
        self.train_mode = tf.placeholder(tf.bool)
        self.vgg = vgg19.Vgg19(vgg_path, dropout=1.0)
        self.vgg.build(input_images, self.train_mode)
        self.output = self.vgg.fc7



        # self.x1 = tf.placeholder(tf.float32, [None, 784])
        # self.x2 = tf.placeholder(tf.float32, [None, 784])
        # # self.x2 = tf.Variable([None, 784])
        # self.W = tf.Variable(tf.zeros([784, 4096]))
        # self.b = tf.Variable(tf.zeros([4096]))
        # self.output1 = tf.matmul(self.x1, self.W) + self.b
        # self.output2 = tf.matmul(self.x2, self.W) + self.b
        # self.output = self.output1 + self.output2

    def all_outputs(self):
        # return [self.output1, self.output2]
        return self.output