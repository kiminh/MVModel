import tensorflow as tf
import numpy as np
from mv_model import MVModel
from config import TrainingConfig

tf.flags.DEFINE_string('train_dir', '/home/shangmingyang/PycharmProjects/TF/mvmodel', 'Directory for saving and loading model checkpoints.')
tf.flags.DEFINE_boolean('train_inception', False, 'whether to train inception submodel varibales.')
tf.flags.DEFINE_integer('number_of_steps', 1000000, 'Number of training steps')
tf.flags.DEFINE_integer('log_every_n_steps', 1, 'Frequency at which loss and global step are logged')

FLAGS = tf.flags.FLAGS


def main(unused_argv):
    assert FLAGS.train_dir, '--train_dir is required'

    model_config = TrainingConfig()

    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    mv_model = MVModel(4096, 12, 128, 40)
    mv_model.co_train()

if __name__ == '__main__':
    tf.app.run()
