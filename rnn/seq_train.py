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
    seq_rnn_model = SequenceRNNModel(4096, 12, 128, 1, 41, 128, batch_size=batch_size)
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
                batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = seq_rnn_model.get_batch(batch_encoder_inputs, batch_decoder_inputs)
                _, loss, outputs = seq_rnn_model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_target_weights)
                print("epoch %d batch %d: loss=%f" %(epoch, batch, loss))
                batch += 1
            if epoch % display_epoch == 0:
                print("epoch %d:display" %(epoch))
            if epoch % save_epoch == 0:
                print("epoch %d:save" %(epoch))
            epoch += 1

if __name__ == '__main__':
    tf.app.run()