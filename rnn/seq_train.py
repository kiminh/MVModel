import tensorflow as tf
import numpy as np

from seq_rnn_model import SequenceRNNModel
import data_utils
import model_data

tf.flags.DEFINE_string('data_path', '/home3/lhl/tensorflow-vgg-master/feature', 'file dir for saving features and labels')

FLAGS = tf.flags.FLAGS

learning_rate = 0.0001
training_iters = 3183 * 100
batch_size = 10
display_step = 100
save_step =  3183 * 1
need_save = True
training_epoches = 100
display_epoch = 1
save_epoch = 10

# Network Parameters
n_steps = 12 # timesteps
n_input = 4096 # model_data data input (img shape: 28*28)
#n_hidden = 128 # hidden layer num of features
n_classes = 40 # model_data total classes (0-9 digits)

def main(unused_argv):
    data =  model_data.read_data(FLAGS.data_path)
    seq_rnn_model = SequenceRNNModel(4096, 12, 128, 1, 40, 128, 40, None)
    with tf.Session() as sess:
        seq_rnn_model.build_model()
        init = tf.global_variables_initializer()
        sess.run(init)

        epoch = 1
        while epoch <= training_epoches:
            batch = 1
            while batch * batch_size <= data.train.size():
                batch_encoder_inputs, batch_decoder_inputs = data.train.next_batch(batch_size)
                batch_encoder_inputs = batch_encoder_inputs.reshape((batch_size, n_steps, n_input))
                batch_weights = [np.ones(batch_size) for _ in xrange(n_classes)]
                sess.run(seq_rnn_model.cost, feed_dict={seq_rnn_model.encoder_inputs:batch_encoder_inputs, seq_rnn_model.decoder_inputs:batch_decoder_inputs, seq_rnn_model.target_weights: batch_weights})
                batch += 1
            if epoch % display_epoch == 0:
                print("epoch %d: " %(epoch))
            if epoch % save_epoch == 0:
                print("epoch %d:" %(epoch))
            epoch += 1





if __name__ == '__main__':
    tf.app.run()