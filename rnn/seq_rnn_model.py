import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder, sequence_loss, rnn_decoder
import numpy as np

class SequenceRNNModel(object):
    def __init__(self, encoder_n_input, encoder_n_steps, encoder_n_hidden,
                 decoder_n_input, decoder_n_steps, decoder_n_hidden, batch_size=10,
                 learning_rate = 0.00001, keep_prob=1.0, is_training=True, use_lstm=True):
        self.encoder_n_input, self.encoder_n_steps, self.encoder_n_hidden = encoder_n_input, encoder_n_steps, encoder_n_hidden
        self.decoder_n_input, self.decoder_n_steps, self.decoder_n_hidden = decoder_n_input, decoder_n_steps, decoder_n_hidden
        n_classes = decoder_n_steps - 1
        self.n_classes, self.decoder_symbols_size = n_classes, n_classes * 2 + 1
        self.batch_size = batch_size
        self.learning_rate, self.keep_prob, self.is_training = learning_rate, keep_prob if is_training else 1.0, is_training
        self.use_lstm, self.decoder_embedding_size = use_lstm, decoder_n_hidden
        if is_training:
            self.feed_previous = False
        else:
            self.feed_previous = True
        self.is_training = is_training

    def single_cell(self, hidden_size):
        if self.use_lstm:
            return rnn.BasicLSTMCell(hidden_size)
        else:
            return rnn.GRUCell(hidden_size)

    def build_model(self):
        # encoder
        self.encoder_inputs = tf.placeholder(tf.float32, [None, self.encoder_n_steps, self.encoder_n_input], name="encoder")
        _, encoder_hidden_state = self.encoder_RNN(self.encoder_inputs)
        # decoder
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)) for i in xrange(self.decoder_n_steps + 1)]
        self.target_weights = [tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)) for i in xrange(self.decoder_n_steps)]
        self.targets = [self.decoder_inputs[i+1] for i in xrange(self.decoder_n_steps)]
        decoder_cell = self.single_cell(self.decoder_n_hidden)
        decoder_proj_w = tf.get_variable("proj_w", [self.decoder_n_hidden, self.decoder_symbols_size])
        decoder_proj_b = tf.get_variable("proj_b", [self.decoder_symbols_size])
        self.decoder_output_projection = (decoder_proj_w, decoder_proj_b)
        if self.decoder_output_projection is None:
            decoder_cell = rnn.core_rnn_cell.OutputProjectionWrapper(decoder_cell, self.decoder_symbols_size)
        constant_embedding = np.ones([self.decoder_symbols_size, 1], dtype=np.float32)
        for i in xrange(self.decoder_symbols_size):
            constant_embedding[i] = np.array([i], dtype=np.float32)
        self.fake_embedding =tf.constant(constant_embedding)
        self.outputs, self.decoder_hidden_state = self.noembedding_rnn_decoder(self.decoder_inputs[:self.decoder_n_steps], encoder_hidden_state, decoder_cell)
        # self.outputs, self.decoder_hidden_state = embedding_rnn_decoder(self.decoder_inputs[:self.decoder_n_steps], encoder_hidden_state,
        #         decoder_cell, self.decoder_symbols_size, self.decoder_embedding_size, output_projection=self.decoder_output_projection, feed_previous=self.feed_previous)
        # do wx+b for output, to generate decoder_symbols_size length
        for i in xrange(self.decoder_n_steps-1): #ignore last output, we only care 40 classes
            self.outputs[i] = tf.matmul(self.outputs[i], self.decoder_output_projection[0]) + self.decoder_output_projection[1]
        if self.feed_previous:
            # do softmax
            self.logits = tf.nn.softmax(self.outputs[:-1], dim=-1, name="output_softmax")

        # cost function
        if self.is_training:
            self.cost = sequence_loss(self.outputs, self.targets, self.target_weights)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # output label and accuracy
        # output_labels, output_labels_probs = [], []
        # for i in xrange(self.decoder_n_steps-1):
        #     output_logits = self.outputs[i]
        #     output_labels.append(tf.argmax(output_logits, 1))
        #     output_labels_probs.append(tf.reduce_max(output_logits, 1))
        #
        # predict_labels = []
        # for j in xrange(self.batch_size):
        #     max_yes_index, max_yes_prob = -1, 0.0
        #     for i in xrange(len(output_labels)):
        #         if output_labels[i][j] % 2 == 1 and output_labels_probs[i][j] > max_yes_prob:
        #             max_yes_index, max_yes_prob = output_labels[i][j], output_labels_probs[i][j]
        #     predict_labels.append(max_yes_index)
        # predict_labels = np.array(predict_labels)
        #
        # target_labels = []
        # print len(self.targets)
        # for j in xrange(self.batch_size):
        #     for i in xrange(len(self.targets)):
        #         print self.targets[i][j]
        #         if self.targets[i][j] % 2 == 1:
        #             target_labels.append(self.targets[i][j])
        #             break
        # target_labels = np.array(target_labels)
        #
        # self.correct_pred = tf.equal(predict_labels, target_labels)
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))









        # self.accuracy =
        # output_labels = [tf.argmax(output, 1) for output in self.outputs] #[[batch-size]]
        # for j in xrange(tf.get_shape(output_labels[0])[0]):
        #     max_yes_index, max_yes_prob = -1, 0.0
        #     for i in xrange(len(output_labels)):
        #         if output_labels[i][j] % 2 == 1 and output
        #
        # for i in xrange(len(self.output_labels)):
        #
        #
        #
        # self.preds = self.RNN(self.x, self.fc_w, self.fc_b)
        #
        # self.output_label = tf.argmax(self.pred, 1)
        # self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _extract_argmax(self, output_projection=None):
        def loop_function(prev, _):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
            prev_symbol = tf.argmax(prev, 1)
            emb_prev = tf.nn.embedding_lookup(self.fake_embedding, prev_symbol)
            return emb_prev
        return loop_function

    def noembedding_rnn_decoder(self, decoder_inputs, init_state, cell):
        loop_function = self._extract_argmax(self.decoder_output_projection) if self.feed_previous else None
        emb_inp = (tf.nn.embedding_lookup(self.fake_embedding, i) for i in decoder_inputs)
        return rnn_decoder(emb_inp, init_state, cell, loop_function=loop_function)

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

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only=False):
        """
        seq mvmodel step operation
        :param session:
        :param encoder_inputs:
        :param decoder_inputs:
        :param target_weights:
        :param forward_only:
        :return: Gridient, loss, logits,
        """
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        for i in xrange(self.decoder_n_steps):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        # for target shift
        last_target = self.decoder_inputs[self.decoder_n_steps].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        if not forward_only:
            output_ops = [self.optimizer, self.cost]
        else:
            output_ops = self.logits
        outputs = session.run(output_ops, input_feed)
        if not forward_only:
            return outputs[0], outputs[1], None #Gradient, loss, no outputs
        else:
            return None, None, outputs #No gradient, no loss, outputs logits

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

    def get_batch(self, batch_encoder_inputs, batch_labels, batch_size=10):
        """
        format batch to fit input placeholder
        :param batch_encoder_inputs:
        :param batch_labels:
        :return:
        """
        self.batch_size = batch_size
        batch_decoder_inputs, batch_target_weights = [], []
        for j in xrange(np.shape(batch_labels)[1]):
            batch_decoder_inputs.append(np.array([batch_labels[i][j] for i in xrange(self.batch_size)]))
            ones_weights = np.ones(self.batch_size)
            batch_target_weights.append(ones_weights)
        batch_target_weights[-1] = np.zeros(self.batch_size)
        return batch_encoder_inputs, batch_decoder_inputs, batch_target_weights

    def predict(self, logits, min_no=True, all_min_no=True):
        """
        predict labels and its prob
        :param logits: logits of each step,shape=[classes, batch_size, 2* classes+1]
        :return: label,shape=[batch_size]
        """
        output_labels, output_labels_probs = [], []
        if not all_min_no:
            for batch_logits in logits:
                output_labels.append(np.argmax(batch_logits, 1))
                output_labels_probs.append(np.amax(batch_logits, 1))
        else:
            for i in xrange(np.shape(logits)[0]):
                batch_logits = logits[i]
                batch_size = np.shape(batch_logits)[0]
                output_labels.append(np.array([(i+1)*2]*batch_size))
                output_labels_probs.append(batch_logits[np.arange(batch_size),(i+1)*2])

        predict_labels = []
        for j in xrange(np.shape(logits)[1]):
            max_yes_index, max_yes_prob, min_no_index, min_no_prob = -1, 0.0, -1, 1.0
            for i in xrange(len(output_labels)):
                if output_labels[i][j] % 2 == 1 and output_labels_probs[i][j] > max_yes_prob:
                    max_yes_index, max_yes_prob = output_labels[i][j], output_labels_probs[i][j]
                if output_labels[i][j] > 0 and output_labels[i][j] % 2 == 0 and output_labels_probs[i][j] < min_no_prob:
                    min_no_index, min_no_prob = output_labels[i][j], output_labels_probs[i][j]
            if all_min_no: # get class by min probability meaning not this class
                predict_labels.append(min_no_index/2)
            elif max_yes_index == -1 and min_no: # extract index with min probablity meaning no
                predict_labels.append(min_no_index/2)
            else:
                predict_labels.append((max_yes_index+1)/2) #convert index to label
        predict_labels = np.array(predict_labels)
        return predict_labels



