import tensorflow as tf
import numpy as np

from seq_rnn_model import SequenceRNNModel
import data_utils
import model_data
import csv

tf.flags.DEFINE_string('data_path', '/home3/lhl/tensorflow-vgg-master/feature', 'file dir for saving features and labels')
tf.flags.DEFINE_string("save_seq_mvmodel_path", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/seq_mvmodel.ckpt", "file path to save model")
tf.flags.DEFINE_string('seq_mvmodel_path', '/home1/shangmingyang/data/3dmodel/seq_mvmodel.ckpt-200', 'trained mvmodel path')
tf.flags.DEFINE_string('test_acc_file', 'seq_acc.csv', 'test acc file')
tf.flags.DEFINE_boolean('train', True, 'train mode')

FLAGS = tf.flags.FLAGS

learning_rate = 0.0001
training_iters = 3183 * 100
batch_size = 10
display_step = 100
save_step =  3183 * 1
need_save = True

training_epoches = 500
display_epoch = 1
save_epoch = 50

# Network Parameters
n_steps = 12 # timesteps
n_input = 4096 # model_data data input (img shape: 28*28)
#n_hidden = 128 # hidden layer num of features
n_classes = 40 # model_data total classes (0-9 digits)

def main(unused_argv):
    if FLAGS.train:
        train()
    else:
        test()

def train():
    data =  model_data.read_data(FLAGS.data_path)
    seq_rnn_model = SequenceRNNModel(4096, 12, 64, 1, 41, 64, batch_size=batch_size, is_training=True)
    with tf.Session() as sess:
        seq_rnn_model.build_model()
        saver = tf.train.Saver(max_to_keep=10)
        init = tf.global_variables_initializer()
        sess.run(init)

        epoch = 1
        while epoch <= training_epoches:
            batch = 1
            while batch * batch_size <= data.train.size():
                batch_encoder_inputs, batch_decoder_inputs = data.train.next_batch(batch_size)
                target_labels = get_target_labels(batch_decoder_inputs)
                batch_encoder_inputs = batch_encoder_inputs.reshape((batch_size, n_steps, n_input))
                batch_encoder_inputs, batch_decoder_inputs, batch_target_weights = seq_rnn_model.get_batch(batch_encoder_inputs, batch_decoder_inputs, batch_size=batch_size)
                _, loss, outputs = seq_rnn_model.step(sess, batch_encoder_inputs, batch_decoder_inputs, batch_target_weights,forward_only=False)
                # predict_labels = seq_rnn_model.predict(outputs)
                # acc = accuracy(predict_labels, target_labels)
                print("epoch %d batch %d: loss=%f" %(epoch, batch, loss))
                batch += 1
            # if epoch % display_epoch == 0:
            #     print("epoch %d:display" %(epoch))
            if epoch % save_epoch == 0:
                saver.save(sess, FLAGS.save_seq_mvmodel_path, global_step=epoch)
            #     # do test using test dataset
            #     test_encoder_inputs, test_decoder_inputs = data.test.next_batch(data.test.size())
            #     target_labels = get_target_labels(test_decoder_inputs)
            #     test_encoder_inputs = test_encoder_inputs.reshape((-1, n_steps, n_input))
            #     test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs, test_decoder_inputs, batch_size=data.test.size())
            #     _, _, outputs = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights, forward_only=True) # don't do optimize
            #     predict_labels = seq_rnn_model.predict(outputs)
            #     acc = accuracy(predict_labels, target_labels)
            #     print("epoch %d:save, acc=%f" %(epoch, acc))
            epoch += 1

def test():
    data = model_data.read_data(FLAGS.data_path)
    seq_rnn_model = SequenceRNNModel(4096, 12, 128, 1, 41, 128, batch_size=data.train.size(), is_training=False)
    with tf.Session() as sess:
        seq_rnn_model.build_model()
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.seq_mvmodel_path)
        print([v.name for v in tf.trainable_variables()])
        train_encoder_inputs, train_decoder_inputs = data.train.next_batch(data.train.size(), shuffle=False)
        #test_encoder_inputs, test_decoder_inputs = data.test.next_batch(data.test.size(), shuffle=False)
        #target_labels = get_target_labels(test_decoder_inputs)
        #test_encoder_inputs = test_encoder_inputs.reshape((-1, n_steps, n_input))
        train_encoder_inputs = train_encoder_inputs.reshape((-1, n_steps, n_input))
        #test_encoder_inputs, test_decoder_inputs, test_target_weights = seq_rnn_model.get_batch(test_encoder_inputs,
        #                                                                                        test_decoder_inputs,
        #                                                                                        batch_size=data.test.size())
        train_encoder_inputs, train_decoder_inputs, train_target_weights = seq_rnn_model.get_batch(train_encoder_inputs,
                                                                                                train_decoder_inputs,
                                                                                                batch_size=data.train.size())
        _, _, outputs_train, hidden_train = seq_rnn_model.step(sess, train_encoder_inputs, train_decoder_inputs, train_target_weights,
                                            forward_only=True)
        #np.save("hidden_feature_train_embedding", hidden_train)
        #_, _, outputs, hidden = seq_rnn_model.step(sess, test_encoder_inputs, test_decoder_inputs, test_target_weights,
        #                                   forward_only=True)  # don't do optimize
        print(np.shape(outputs_train), np.shape(hidden_train))
        np.save('hidden_feature_train_embedding', hidden_train)
        predict_labels = seq_rnn_model.predict(outputs, all_min_no=False)
        acc = accuracy(predict_labels, target_labels)

        with open(FLAGS.test_acc_file, 'a') as f:
            w = csv.writer(f)
            w.writerow([FLAGS.seq_mvmodel_path, acc])
        print("model:%s, acc=%f" % (FLAGS.seq_mvmodel_path, acc))


def get_target_labels(seq_labels):
    target_labels = []
    for i in xrange(np.shape(seq_labels)[0]): #loop batch_size
        for j in xrange(np.shape(seq_labels)[1]): #loop label
            if seq_labels[i][j] % 2 == 1:
                target_labels.append((seq_labels[i][j]+1)/2)
                break
    return target_labels

def accuracy(predict, target):
    return np.mean(np.equal(predict, target))


if __name__ == '__main__':
    tf.app.run()
