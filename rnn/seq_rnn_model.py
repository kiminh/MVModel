import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder, sequence_loss
import numpy as np

class SequenceRNNModel(object):
    def __init__(self, encoder_n_input, encoder_n_steps, encoder_n_hidden,
                 decoder_n_input, decoder_n_steps, decoder_n_hidden, n_classes, output_projection, decoder_embedding_size=256,
                 learning_rate = 0.00001, keep_prob=1.0, is_training=True, use_lstm=True):
        self.encoder_n_input, self.encoder_n_steps, self.encoder_n_hidden = encoder_n_input, encoder_n_steps, encoder_n_hidden
        self.decoder_n_input, self.decoder_n_steps, self.decoder_n_hidden = decoder_n_input, decoder_n_steps, decoder_n_hidden
        self.n_classes, self.decoder_symbols_size = n_classes, n_classes * 2 + 1
        self.learning_rate, self.keep_prob, self.is_training = learning_rate, keep_prob if is_training else 1.0, is_training
        self.use_lstm, self.decoder_embedding_size, self.output_projection = use_lstm, decoder_embedding_size, output_projection
        if is_training:
            self.feed_previous = False
        else:
            self.feed_previous = True

    def single_cell(self, hidden_size):
        if self.use_lstm:
            return rnn.BasicLSTMCell(hidden_size)
        else:
            return rnn.GRUCell(hidden_size)

    def build_model(self):
        # encoder
        self.encoder_inputs = tf.placeholder(tf.float32, [None, self.encoder_n_steps, self.encoder_n_input])
        _, encoder_hidden_state = self.encoder_RNN(self.encoder_inputs)
        # decoder
        self.decoder_inputs, self.target_weights = [], []
        for i in xrange(self.decoder_n_steps + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
        self.targets = [self.decoder_inputs[i+1] for i in xrange(self.decoder_n_steps)]
        decoder_cell = self.single_cell(self.decoder_n_hidden)
        if self.output_projection is None:
            decoder_cell = rnn.core_rnn_cell.OutputProjectionWrapper(decoder_cell, self.decoder_symbols_size)
        self.outputs, self.decoder_hidden_state = embedding_rnn_decoder(self.decoder_inputs, encoder_hidden_state,
                decoder_cell, self.decoder_symbols_size, self.decoder_embedding_size, output_projection=self.output_projection, feed_previous=self.feed_previous)
        # cost function
        if self.is_training:
            self.cost = sequence_loss(self.outputs, self.targets, self.target_weights)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # self.accuracy =

        self.preds = self.RNN(self.x, self.fc_w, self.fc_b)

        self.output_label = tf.argmax(self.pred, 1)
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def encoder_RNN(self, encoder_inputs):
        """
        Encoder: Encode images, generate outputs and last hidden states
        :param encoder_inputs: inputs for all steps,shape=[batch_size, step, feature_size]
        :return: outputs of all steps and last hidden state
        """
        encoder_input_list = tf.unstack(encoder_inputs, self.encoder_n_steps, 1)
        encoder_input_dropout = [tf.nn.dropout(input_i, self.keep_prob) for input_i in encoder_input_list]
        cell = self.single_cell(self.encoder_n_hidden)
        outputs, states = rnn.static_rnn(cell, encoder_input_dropout, dtype=tf.float32)
        return outputs, states

    def decoder_RNN(self, decoder_inputs, fc_weights, fc_biases):
        """
        Decoder RNN: decode input to each class probalities
        :param decoder_inputs: view features vectors
        :param fc_weights: fc weights used in each cell's output
        :param fc_biases: fc biases used in each cell's output
        :return: each step's output after fc
        """
        decoder_input_list = tf.unstack(decoder_inputs, self.decoder_n_steps, 1)
        decoder_x_dropout = [tf.nn.dropout(x_i, self.keep_prob) for x_i in decoder_input_list]
        cell = rnn.GRUCell(self.decoder_n_hidden)
        outputs, states = rnn.static_rnn(cell, decoder_x_dropout, dtype=tf.float32)

        return [tf.matmul(output, fc_weights) + fc_biases for output in outputs]


    def get_fc_vars(self, reuse=False):
        with tf.variable_scope("fc") as scope:
            if reuse:
                scope.reuse_variables()