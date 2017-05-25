import tensorflow as tf
import numpy as np
from model.mv_model import MVModel
from model.config import TrainingConfig, ModelConfig

tf.flags.DEFINE_string('test_model_path', '/home1/shangmingyang/data/3dmodel/trained_mvmodel/0.9_0.9_4/mvmodel.ckpt-5', 'trained model path')
# tf.flags.DEFINE_boolean('train_inception', False, 'whether to train inception submodel varibales.')
# tf.flags.DEFINE_integer('number_of_steps', 1000000, 'Number of training steps')
# tf.flags.DEFINE_integer('log_every_n_steps', 1, 'Frequency at which loss and global step are logged')

FLAGS = tf.flags.FLAGS

def main(unused_argv):
    # assert FLAGS.model_path, '--model_path is required'

    train_config, model_config = TrainingConfig(), ModelConfig()

    mv_model = MVModel(train_config, model_config, is_training=False)
    mv_model.test(FLAGS.test_model_path)

if __name__ == '__main__':
    tf.app.run()
