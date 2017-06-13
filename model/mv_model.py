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

tf.flags.DEFINE_string('model_path', '/home1/shangmingyang/data/3dmodel/trained_mvmodel/1.0_0.5_4/mvmodel.ckpt', 'path for saving model')
tf.flags.DEFINE_string('vgg_path', '/home1/shangmingyang/models/vgg/vgg19.npy', 'trained model for vgg')
tf.flags.DEFINE_string('modelnet_path', '/home3/lxh/modelnet/modelnet40v1_2', 'modelnet dir')

FLAGS = tf.flags.FLAGS

class MVModel(object):
    def __init__(self, train_config, model_config, is_training=True):
        self.train_config, self.model_config = train_config, model_config
        self.is_training = is_training
        self.cnn_model = CNNModel(self.train_config.cnn_keep_prob, is_training=is_training)
        self.rnn_model = RNNModel(train_config.learning_rate, model_config.n_fcs, model_config.n_views,
                                  model_config.n_hidden, model_config.n_classes, train_config.rnn_keep_prob if is_training else 1.0, is_training=self.is_training)
        self.gpu_config = tf.ConfigProto()
        self.gpu_config.gpu_options.allow_growth = True
        self.data = modelnet.read_data(FLAGS.modelnet_path)


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.variable_scope('mv-cnn') as scope:
            self.cnn_model.build_model(self.images, FLAGS.vgg_path)
        with tf.variable_scope('mv-rnn') as scope:
            self.rnn_model.build_model(self.cnn_model.all_outputs())

        self.optimizer = self.rnn_model.optimizer

    def co_train(self):
        with tf.Session() as sess:
            self.build_model()
            print('build model finished')
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=10)
            # TODO restore trained model before or run init op
            # saver.restore(sess, FLAGS.model_path)
            sess.run(init)
            print('init model parameter finished')
            epoch = 1
            print('start training')
            with open('variables.txt', 'w') as f:
                f.writelines('\n'.join([v.name for v in tf.global_variables()]))
            #print([v.name for v in tf.global_variables()])
            #saver.save(sess, FLAGS.model_path, global_step=0)

            fc6_weights = [v for v in tf.global_variables() if v.name=='mv-cnn/fc6/fc6_weights:0'][0]
            #cell_biases = [v for v in tf.global_variables() if v.name=='rnn/gru_cell/gates/biases/Adam_1:0'][0]

            while epoch <= self.train_config.training_epoches:
                batch = 1
                while batch * self.train_config.batch_size <= self.data.train.size():
                    batch_imgpaths, batch_labels = self.data.train.next_batch(self.train_config.batch_size)
                    batch_img = self.build_input(batch_imgpaths)
                    sess.run(self.optimizer, feed_dict={self.images: batch_img, self.rnn_model.y: batch_labels, self.cnn_model.train_mode: True})
                    acc, loss = sess.run([self.rnn_model.accuracy, self.rnn_model.cost], feed_dict={self.images:batch_img, self.rnn_model.y:batch_labels, self.cnn_model.train_mode:False})
                    print("epoch " + str(epoch) + ",batch " + str(epoch*batch) + ", Minibatch loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                    #print("fc6 weights:", sess.run(fc6_weights))
                    #print("cell biases:", sess.run(cell_biases))
                    batch += 1

                if epoch % self.train_config.display_epoches == 0:
                    acc, loss = sess.run([self.rnn_model.accuracy, self.rnn_model.cost], feed_dict={self.images:batch_img, self.rnn_model.y:batch_labels, self.cnn_model.train_mode:False})
                    print("epoch " + str(epoch) + ", Minibatch loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                if epoch % self.train_config.save_epoches == 0:
                    #test_imgpaths, test_labels = self.data.test.views(), self.data.test.labels
                    #test_imgs = self.build_input(test_imgpaths)
                    #acc = sess.run([self.rnn_model.accuracy], feed_dict={self.images:test_imgs, self.rnn_model.y:test_labels, self.cnn_model.train_mode:False})
                    #print("epoch" + str(epoch) + ", Testing accuracy=" + "{:.6f}".format(acc))
                    try:
                        saver.save(sess, FLAGS.model_path, global_step=epoch)
                    except Exception as e:
                        print("save model exception:", e)

                epoch += 1


    def test(self, test_model_path=""):
        with tf.Session() as sess:
            self.build_model()
            print("build model finished")
            saver = tf.train.Saver()
            saver.restore(sess, test_model_path if len(test_model_path)>0 else FLAGS.model_path)
            print("restore model parameter finished")

            total_acc = 0.0

            test_imgpaths, test_labels = self.data.test.views(), self.data.test.labels
            with open('wrong_model.txt', 'w') as f:
                for i in xrange(len(test_labels)):
                    test_imgs = self.build_input(test_imgpaths[i*self.model_config.n_views : (i+1)*self.model_config.n_views])
                    acc = sess.run([self.rnn_model.accuracy],
                                  feed_dict={self.images:test_imgs, self.rnn_model.y:test_labels[i:i+1], self.cnn_model.train_mode:False})
                    acc = acc[0]
                    if acc < 0.5:
                        f.write(test_imgpaths[self.model_config.n_views*i]+'\n')
                    total_acc += acc
            print("accuracy in %d models: %f using model %s" %(len(test_labels), total_acc/len(test_labels), test_model_path))

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


