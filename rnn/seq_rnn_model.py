import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_decoder, sequence_loss, rnn_decoder, embedding_attention_decoder
from tensorflow.python.util import nest

import numpy as np
import copy

def linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = tf.get_variable_scope()
  with tf.variable_scope(scope) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return tf.nn.bias_add(res, biases)


class SequenceRNNModel(object):
    def __init__(self, encoder_symbols_size, encoder_n_steps, encoder_n_hidden,
                 decoder_symbols_size, decoder_n_steps, decoder_n_hidden, batch_size=32,
                 learning_rate = 0.00001, keep_prob=1.0, is_training=True,
                 use_lstm=False, use_embedding=True, use_attention=True,
                 num_heads=1, init_embedding=None):
        self.encoder_symbols_size, self.encoder_n_steps, self.encoder_n_hidden = encoder_symbols_size, encoder_n_steps, encoder_n_hidden
        self.decoder_symbols_size, self.decoder_n_steps, self.decoder_n_hidden = decoder_symbols_size, decoder_n_steps, decoder_n_hidden
        self.n_classes = decoder_n_steps - 1
        self.batch_size = batch_size
        self.learning_rate, self.keep_prob, self.is_training = learning_rate, keep_prob if is_training else 1.0, is_training
        self.use_lstm, self.decoder_embedding_size = use_lstm, decoder_n_hidden
        self.init_embedding = init_embedding
        self.encoder_embedding_size = init_embedding.shape[1] if init_embedding else encoder_n_hidden
        if is_training:
            self.feed_previous = False
        else:
            self.feed_previous = True
        self.is_training, self.use_embedding, self.use_attention = is_training, use_embedding, use_attention
        if self.use_attention:
            self.num_heads = num_heads

    def single_cell(self, hidden_size):
        if self.use_lstm:
            return rnn.BasicLSTMCell(hidden_size)
        else:
            return rnn.GRUCell(hidden_size)

    def build_model(self):
        # encoder
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)) for i in xrange(self.encoder_n_steps)]
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

        self.outputs, self.decoder_hidden_state, self.attns_weights = self.self_embedding_attention_seq2seq(
            self.encoder_inputs,
            self.decoder_inputs[:self.decoder_n_steps], decoder_cell, self.encoder_symbols_size,
            self.decoder_symbols_size, self.decoder_embedding_size,
            input_init_embedding=self.init_embedding,
            output_projection=self.decoder_output_projection, feed_previous=self.feed_previous)


        # do wx+b for output, to generate decoder_symbols_size length
        for i in xrange(self.decoder_n_steps-1): #ignore last output, we only care 40 classes
            self.outputs[i] = tf.matmul(self.outputs[i], self.decoder_output_projection[0]) + self.decoder_output_projection[1]
        if self.feed_previous:
            # do softmax
            self.logits = tf.nn.softmax(self.outputs[:-1], dim=-1, name="output_softmax")
            if self.attns_weights is not None:
                self.attns_weights = self.attns_weights[:-1]

        # cost function
        if self.is_training:
            self.cost = sequence_loss(self.outputs, self.targets, self.target_weights)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    def _extract_argmax(self, embedding, output_projection=None):
        def loop_function(prev, _):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
            prev_symbol = tf.argmax(prev, 1)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            return emb_prev
        return loop_function

    def noembedding_rnn_decoder(self, decoder_inputs, init_state, cell):
        loop_function = self._extract_argmax(self.fake_embedding, self.decoder_output_projection) if self.feed_previous else None
        emb_inp = (tf.nn.embedding_lookup(self.fake_embedding, i) for i in decoder_inputs)
        return rnn_decoder(emb_inp, init_state, cell, loop_function=loop_function)

    def noembedding_attention_rnn_decoder(self, decoder_inputs, init_state, attention_states, cell, num_heads=1):
        loop_function = self._extract_argmax(self.fake_embedding, self.decoder_output_projection) if self.feed_previous else None
        emb_inp = [tf.nn.embedding_lookup(self.fake_embedding, i) for i in decoder_inputs]
        return self.self_attention_decoder(emb_inp, init_state, attention_states, cell, loop_function=loop_function, num_heads=num_heads)

    def self_embedding_attention_decoder(self, decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
        if output_size is None:
            output_size = cell.output_size
        if output_projection is not None:
            proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        with tf.variable_scope(
                        scope or "embedding_attention_decoder", dtype=dtype) as scope:

            embedding = tf.get_variable("embedding",
                                                    [num_symbols, embedding_size])
            loop_function = self._extract_argmax(
                embedding, output_projection) if feed_previous else None
            emb_inp = [
                tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs
                ]
            return self.self_attention_decoder(
                emb_inp,
                initial_state,
                attention_states,
                cell,
                output_size=output_size,
                num_heads=num_heads,
                loop_function=loop_function,
                initial_state_attention=initial_state_attention)

    def self_attention_decoder(self, decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
        if not decoder_inputs:
            raise ValueError("Must provide at least 1 input to attention decoder.")
        if num_heads < 1:
            raise ValueError("With less than 1 heads, use a non-attention decoder.")
        if attention_states.get_shape()[2].value is None:
            raise ValueError("Shape[2] of attention_states must be known: %s" %
                             attention_states.get_shape())
        if output_size is None:
            output_size = cell.output_size

        with tf.variable_scope(
                        scope or "attention_decoder", dtype=dtype) as scope:
            dtype = scope.dtype

            batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            attn_length = attention_states.get_shape()[1].value
            if attn_length is None:
                attn_length = tf.shape(attention_states)[1]
            attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = tf.reshape(attention_states,
                                       [-1, attn_length, 1, attn_size])
            hidden_features = []
            v = []
            attention_vec_size = attn_size  # Size of query vectors for attention.
            for a in xrange(num_heads):
                k = tf.get_variable("AttnW_%d" % a,
                                                [1, 1, attn_size, attention_vec_size])
                hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                v.append(
                    tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

            state = initial_state

            def attention(query):
                """Put attention masks on hidden using hidden_features and query."""
                ds = []  # Results of attention reads will be stored here.
                att_weights = [] # attention weights given specific query
                if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = tf.concat(query_list, 1)
                for a in xrange(num_heads):
                    with tf.variable_scope("Attention_%d" % a):
                        y = linear(query, attention_vec_size, True)
                        y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                                [2, 3])
                        a = tf.nn.softmax(s)
                        #a = tf.nn.l2_normalize(tf.multiply(s,s), 1)
                        # s_square = tf.multiply(s, s)
                        #s_square_sum = tf.expand_dims(tf.reduce_sum(s_square, axis=1), 1)
                        #a = s_square / s_square_sum
                        #s_sigmoid = tf.sigmoid(s)
                        #s_sigmoid_sum = tf.expand_dims(tf.reduce_sum(s_sigmoid, axis=1), 1)
                        #a = s_sigmoid / s_sigmoid_sum
                        #a = tf.nn.l2_normalize(tf.sigmoid(s), 1)
                        att_weights.append(a)
                        # Now calculate the attention-weighted vector d.
                        d = tf.reduce_sum(
                            tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                        ds.append(tf.reshape(d, [-1, attn_size]))
                return ds, att_weights

            outputs = []
            prev = None
            batch_attn_size = tf.stack([batch_size, attn_size])
            attns = [
                tf.zeros(
                    batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
                ]
            attns_weights = []
            for a in attns:  # Ensure the second shape of attention vectors is set.
                a.set_shape([None, attn_size])
            if initial_state_attention:
                attns, _ = attention(initial_state)
            for i, inp in enumerate(decoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # If loop_function is set, we use it instead of decoder_inputs.
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = loop_function(prev, i)
                # Merge input and previous attentions into one vector of the right size.
                input_size = inp.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError("Could not infer input size from input: %s" % inp.name)
                x = linear([inp] + attns, input_size, True)
                # Run the RNN.
                cell_output, state = cell(x, state)
                # Run the attention mechanism.
                if i == 0 and initial_state_attention:
                    with tf.variable_scope(
                            tf.get_variable_scope(), reuse=True):
                        attns, attns_weight = attention(state)
                else:
                    attns, attns_weight = attention(state)

                with tf.variable_scope("AttnOutputProjection"):
                    output = linear([cell_output] + attns, output_size, True)
                if loop_function is not None:
                    prev = output
                outputs.append(output)
                attns_weights.append(attns_weight)

        return outputs, state, attns_weights

    def self_embedding_attention_seq2seq(self, encoder_inputs,
                                         decoder_inputs,
                                         cell,
                                         num_encoder_symbols,
                                         num_decoder_symbols,
                                         encoder_embedding_size,
                                         decoder_embedding_size,
                                         input_init_embedding=None,
                                         num_heads=1,
                                         output_projection=None,
                                         feed_previous=False,
                                         dtype=None,
                                         scope=None,
                                         initial_state_attention=False):
        """Embedding sequence-to-sequence model with attention.
        This model first embeds encoder_inputs by a newly created embedding (of shape
        [num_encoder_symbols x input_size]). Then it runs an RNN to encode
        embedded encoder_inputs into a state vector. It keeps the outputs of this
        RNN at every step to use for attention later. Next, it embeds decoder_inputs
        by another newly created embedding (of shape [num_decoder_symbols x
        input_size]). Then it runs attention decoder, initialized with the last
        encoder state, on embedded decoder_inputs and attending to encoder outputs.
        Warning: when output_projection is None, the size of the attention vectors
        and variables will be made proportional to num_decoder_symbols, can be large.
        Args:
          encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
          decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
          cell: core_rnn_cell.RNNCell defining the cell function and size.
          num_encoder_symbols: Integer; number of symbols on the encoder side.
          num_decoder_symbols: Integer; number of symbols on the decoder side.
          embedding_size: Integer, the length of the embedding vector for each symbol.
          num_heads: Number of attention heads that read from attention_states.
          output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_decoder_symbols] and B has
            shape [num_decoder_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
          feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
          dtype: The dtype of the initial RNN state (default: tf.float32).
          scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_seq2seq".
          initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states.
        Returns:
          A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
              shape [batch_size x num_decoder_symbols] containing the generated
              outputs.
            state: The state of each decoder cell at the final time-step.
              It is a 2D Tensor of shape [batch_size x cell.state_size].
        """
        with tf.variable_scope(
                        scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
            dtype = scope.dtype
            # Encoder.
            encoder_cell = copy.deepcopy(cell)
            encoder_cell = rnn.core_rnn_cell.EmbeddingWrapper(
                encoder_cell,
                embedding_classes=num_encoder_symbols,
                embedding_size=encoder_embedding_size,
                initializer=tf.constant_initializer(input_init_embedding) if input_init_embedding else None)
            encoder_outputs, encoder_state = rnn.static_rnn(
                encoder_cell, encoder_inputs, dtype=dtype)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [
                tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
                ]
            attention_states = tf.concat(top_states, 1)

            # Decoder.
            output_size = None
            if output_projection is None:
                cell = rnn.core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
                output_size = num_decoder_symbols

            if isinstance(feed_previous, bool):
                return self.self_embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    decoder_embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    initial_state_attention=initial_state_attention)

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
        for i in xrange(self.encoder_n_steps):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in xrange(self.decoder_n_steps):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        # for target shift
        last_target = self.decoder_inputs[self.decoder_n_steps].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        self.input_feed = input_feed
        if not forward_only:
            output_ops = [self.optimizer, self.cost]
        else:
            output_ops = self.logits
        outputs = session.run(output_ops, input_feed)
        if not forward_only:
            return outputs[0], outputs[1], None, None #Gradient, loss, no outputs, no encoder_hidden
        else:
            attns_weights = session.run(self.attns_weights, input_feed)
            return None, None, outputs, attns_weights #No gradient, no loss, outputs logits, encoder_hidden

    def get_batch(self, batch_inputs, batch_labels, batch_size=10):
        """
        format batch to fit input placeholder
        :param batch_encoder_inputs:
        :param batch_labels:
        :return:
        """
        batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = [], [], []
        for j in xrange(np.shape(batch_inputs)[1]):
            batch_encoder_inputs.append(np.array([batch_inputs[i][j] for i in xrange(self.batch_size)]))
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



