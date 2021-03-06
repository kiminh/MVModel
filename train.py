import tensorflow as tf
import numpy as np
from model.mv_model import MVModel
from model.config import TrainingConfig, ModelConfig

tf.flags.DEFINE_string('train_dir', '/home/shangmingyang/PycharmProjects/TF/mvmodel', 'Directory for saving and loading model checkpoints.')
tf.flags.DEFINE_boolean('train_inception', False, 'whether to train inception submodel varibales.')
tf.flags.DEFINE_integer('number_of_steps', 1000000, 'Number of training steps')
tf.flags.DEFINE_integer('log_every_n_steps', 1, 'Frequency at which loss and global step are logged')

FLAGS = tf.flags.FLAGS


def main(unused_argv):
    assert FLAGS.train_dir, '--train_dir is required'

    train_config, model_config = TrainingConfig(), ModelConfig()

    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    mv_model = MVModel(train_config, model_config, is_training=True)
    mv_model.co_train()

if __name__ == '__main__':
    tf.app.run()
