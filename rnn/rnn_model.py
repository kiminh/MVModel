import tensorflow as tf
from tensorflow.contrib import rnn
from model_data import read_data
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string('model_path', '/home/shangmingyang/PycharmProjects/TF/model', 'file path for saving model')
tf.flags.DEFINE_string('data_path', '/home/shangmingyang/PycharmProjects/TF/data', 'file dir for saving features and labels')

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

FLAGS = tf.flags.FLAGS


learning_rate = 0.0001
training_iters = 3183 * 500
batch_size = 10
display_step = 100
save_step = 3183 * 500

# Network Parameters
# TODO change n_steps and n_input and n_classes
n_steps = 11 # timesteps
n_input = 2048 # model_data data input (img shape: 28*28)
n_hidden = 128 # hidden layer num of features
n_classes = 40 # model_data total classes (0-9 digits)

model_data = read_data(FLAGS.data_path, n_steps)

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    # lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn.GRUCell(n_hidden)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']




pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = model_data.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        # if step % save_step == 0:
        #     saver.save(sess, FLAGS.model_path)
        step += 1

    print("Optimization Finished!")

    # Calculate accuracy for 128 model_data test images
    # test_len = 256
    test_data = model_data.test.fcs.reshape((-1, n_steps, n_input))
    test_label = model_data.test.labels
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
    print("using %d views" % (n_steps))

def test(model_path=FLAGS.model_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        test_data = model_data.test.fcs.reshape((-1, n_steps, n_input))
        test_label = model_data.test.labels
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

# class RNNModel(object):
#     def __init__(self, lstm_size, lstm_input_size, class_num, fcs, output_keep_pro=0.9, is_training=True):
#         self.lstm_size, class_num, lstm_input_size = lstm_size, class_num, lstm_input_size # lstm size in rnn model
#         self.input_data = tf.placeholder(tf.float32, [None, lstm_size, lstm_input_size])
#
#         self.input_label = tf.placeholder(tf.int32, [1])
#
#         w = {
#             'out': tf.Variable(tf.random_normal([lstm_size, class_num]))
#         }
#         b = {
#             'out': tf.Variable(tf.random_normal([class_num]))
#         }
#
#     def RNN(self, x, weights, biases):
#         x = tf.unstack(x, self.lstm_input_size, 1)
#         lstm_cell = rnn.LayerNormBasicLSTMCell(self.lstm_size, forget_bias=1.0)
#         outputs, states = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)
#
#         return tf.matmul(outputs[-1], weights['out'] + biases['out'])

