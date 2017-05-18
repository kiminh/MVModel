import tensorflow as tf
import numpy as np
from utils import imgutils
import functools

from data import modelnet
# from tensorflow.examples.tutorials.mnist import input_data
from cnn_model import CNNModel
from rnn_model import RNNModel

#tf.flags.DEFINE_string('model_path', '~/PycharmProjects/TF/trained_model', 'path for saving model')
#tf.flags.DEFINE_string('vgg_path', '/home/shangmingyang/PycharmProjects/TF/trained_model/vgg19.npy', 'trained model for vgg')

tf.flags.DEFINE_string('model_path', '/home1/shangmingyang/data/3dmodel/trained_model', 'path for saving model')
tf.flags.DEFINE_string('vgg_path', '/home1/shangmingyang/models/vgg/vgg19.npy', 'trained model for vgg')
tf.flags.DEFINE_string('modelnet_path', '/home3/lxh/modelnet/modelnet40v1_2', 'modelnet dir')

FLAGS = tf.flags.FLAGS

# total_epoches = 10
# save_epoches = 1
# batch_size = 10
# display_step = 10
# save_step = 100
# need_save = True
# training_iters = 1000
# keep_prob = 1.0 #TODO don't use dropout

class MVModel(object):
    # def __init__(self, n_input, n_steps, n_hidden, n_classes, keep_prob=1.0, is_training=True, config=None):
    #     self.cnn_model = CNNModel()
    #     self.rnn_model = RNNModel(n_input, n_steps, n_hidden, n_classes, keep_prob, is_training, config)
    #
    #     if config:
    #         self.config = config
    #     else:
    #         self.config = tf.ConfigProto()
    #         self.config.gpu_options.allow_growth = True
    #
    #     # self.mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    #     self.data = modelnet.read_data(FLAGS.modelnet_path)

    def __init__(self, train_config, model_config, is_training=True):
        self.train_config, self.model_config = train_config, model_config
        self.cnn_model = CNNModel(self.cnn_model.keep_prob)
        self.rnn_model = RNNModel(model_config.n_fcs, model_config.n_views, model_config.n_hidden, model_config.n_classes, train_config.rnn_keep_prob if is_training else 1.0)
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.data = modelnet.read_data(FLAGS.modelnet_path)


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.cnn_model.build_model(self.images, FLAGS.vgg_path)
        self.rnn_model.build_model(self.cnn_model.all_outputs())

        self.optimizer = self.rnn_model.optimizer

    def co_train(self):
        with tf.Session(config=self.gpu_config) as sess:
            self.build_model()
            print('build model finished')
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            # TODO restore trained model before
            sess.run(init)
            print('init model parameter finished')
            epoch = 1
            # training_epoches = 100
            # display_epoch = 1
            # save_epoch = 10
            print('start training')
            while epoch <= self.train_config.training_epoches:
                batch = 1
                while batch * self.train_config.batch_size <= self.data.train.size():
                    batch_imgpaths, batch_labels = self.data.train.next_batch(self.train_config.batch_size)
                    batch_img = self.build_input(batch_imgpaths)
                    sess.run(self.optimizer, feed_dict={self.images: batch_img, self.rnn_model.y: batch_labels, self.cnn_model.train_mode: True})
                    print("batch " + str(epoch * batch))
                    batch += 1

                if epoch % self.train_config.display_epoches == 0:
                    acc, loss = sess.run([self.rnn_model.accuracy, self.rnn_model.cost], feed_dict={self.images:batch_img, self.rnn_model.y:batch_labels, self.cnn_model.train_mode:False})
                    print("epoch " + str(epoch) + ", Minibatch loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                if epoch % self.train_config.save_epoches == 0:
                    test_imgpaths, test_labels = self.data.test.views(), self.data.test.labels()
                    test_imgs = self.build_input(test_imgpaths)
                    acc = sess.run([self.rnn_model.accuracy], feed_dict={self.images:test_imgs, self.rnn_model.y:test_labels, self.cnn_model.train_mode:False})
                    print("epoch" + str(epoch) + ", Testing accuracy=" + "{:.6f}".format(acc))

                epoch += 1


    def build_input(self, imgpaths):
        """
        build input data for model input tensor
        :param imgpaths: images path  array
        :return: 3 channel image array
        """
        images = np.array([imgutils.load_image(img_path) for img_path in imgpaths])
        return [img.reshape(224, 224, 3) for img in images]


    # def train(self, goon=False):
    #     with tf.Session(config=self.config) as sess:
    #         self.build_model()
    #         init = tf.global_variables_initializer()
    #         saver = tf.train.Saver()
    #         if goon:
    #             pass
    #             saver.restore(sess, FLAGS.model_path)
    #         else:
    #             sess.run(init)
    #         step = 1
    #         while step * batch_size <= training_iters:
    #             batch_x, batch_y = self.mnist.train.next_batch(batch_size)
    #             batch_x = batch_x.reshape((batch_size, 784))
    #             batch_x = np.repeat(batch_x, 2, axis=0)
    #             # batch_y = np.repeat(batch_y, 2, axis=0)
    #             sess.run(self.optimizer, feed_dict={self.cnn_model.x1:batch_x, self.cnn_model.x2:batch_x, self.rnn_model.y:batch_y, self.rnn_model.p:keep_prob})
    #
    #             if step % display_step == 0:
    #                 acc, loss = sess.run([self.rnn_model.accuracy, self.rnn_model.cost], feed_dict={self.cnn_model.x1:batch_x, self.cnn_model.x2:batch_x, self.rnn_model.y:batch_y, self.rnn_model.p:1.0})
    #                 print("Iter " + str(step*batch_size) + ", Minibatch loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    #             if step % save_step == 0 and need_save:
    #                 test_data = self.mnist.test.images.reshape((-1, 784))
    #                 test_data = np.repeat(test_data, 2, axis=0)
    #                 test_label = self.mnist.test.labels
    #                 # test_label = np.repeat(test_label, 2, axis=0)
    #                 print("Testing Accuracy:", sess.run(self.rnn_model.accuracy, feed_dict={self.cnn_model.x1:test_data, self.cnn_model.x2:test_data, self.rnn_model.y:test_label, self.rnn_model.p:1.0}))
    #             step += 1


