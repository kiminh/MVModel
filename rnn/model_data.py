import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.learn.python.learn.datasets import base

tf.flags.DEFINE_string('data_dir', '/home/shangmingyang/PycharmProjects/TF/data', 'dir path saving model features file and labels file for training and testing')
tf.flags.DEFINE_string('train_feature_file', 'feature_googlenet_train.npy', 'file path saving model features for training')
tf.flags.DEFINE_string('train_label_file', 'train_label.npy', 'file path saving model labels for training')
tf.flags.DEFINE_string('test_feature_file', 'feature_googlenet_val.npy', 'file path saving model features for testing')
tf.flags.DEFINE_string('test_label_file', 'test_label.npy', 'file path saving model labels for testing')

FLAGS = tf.flags.FLAGS

class DataSet(object):
    def __init__(self, filenames, fcs, labels):
        self._filenames, self._fcs, self._labels = filenames, fcs, labels
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = fcs.shape[0]

    @property
    def labels(self):
        return self._labels
    @property
    def fcs(self):
        return self._fcs

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
                ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._fcs = self._fcs[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            fcs_rest_part = self._fcs[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._fcs = self._fcs[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            fcs_new_part = self._fcs[start:end]
            labels_new_part = self._labels[start:end]

            return np.concatenate((fcs_rest_part, fcs_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._fcs[start:end], self._labels[start:end]

def read_data(data_dir, n_views=12):
    print("read data from %s" %data_dir)
    train_fcs = np.load(os.path.join(data_dir, FLAGS.train_feature_file))
    train_fcs = multiview(train_fcs, n_views)
    train_labels = np.load(os.path.join(data_dir, FLAGS.train_label_file))
    train_labels = onehot(train_labels)

    test_fcs = np.load(os.path.join(data_dir, FLAGS.test_feature_file))
    test_fcs = multiview(test_fcs, n_views)
    test_labels = np.load(os.path.join(data_dir, FLAGS.test_label_file))
    test_labels = onehot(test_labels)

    train_dataset = DataSet(None, train_fcs, train_labels)
    test_dataset = DataSet(None, test_fcs, test_labels)

    print("read data finished")
    return base.Datasets(train=train_dataset, test=test_dataset, validation=None)

def multiview(fcs, n_views=12):
    fcs2 = np.zeros(shape=[fcs.shape[0], n_views, fcs.shape[2]])
    for i in xrange(len(fcs)):
        fcs2[i] = fcs[i][:n_views]
    return fcs2

def onehot(labels):
    label_count = np.shape(labels)[0]
    labels2 = np.zeros(shape=[label_count, 40]) # TODO shape=[batch_size, n_classes]
    labels2[np.arange(label_count), labels] = 1
    return labels2

def _fake_write_data(data_dir):
    train_fcs = np.zeros(shape=[2, 12, 13])
    train_labels = np.ones(shape=[2])

    test_fcs = np.ones(shape=[2,12,13])
    test_labels = np.zeros(shape=[2])

    np.save(os.path.join(data_dir, FLAGS.train_feature_file[ : FLAGS.train_feature_file.find('.')]), train_fcs)
    np.save(os.path.join(data_dir, FLAGS.train_label_file[ : FLAGS.train_label_file.find('.')]), train_labels)

    np.save(os.path.join(data_dir, FLAGS.test_feature_file[ : FLAGS.test_feature_file.find('.')]), test_fcs)
    np.save(os.path.join(data_dir, FLAGS.test_label_file[ : FLAGS.test_label_file.find('.')]), test_labels)


def run_readdata_demo(data_dir):
    model_data = read_data(data_dir)
    train_data = model_data.train
    test_data = model_data.test
    fc1, label1 = test_data.next_batch(1)
    print("fc1:", np.shape(fc1))
    print("label1:", label1)
    print("shape:", np.shape(label1))
    # print("shape:", np.shape(batch1))

if __name__ == '__main__':
    # _fake_write_data(FLAGS.data_dir) #only run once to fake data
    # run_readdata_demo(FLAGS.data_dir)
    # data = np.load(os.path.join(FLAGS.data_dir, FLAGS.train_feature_file))
    data = np.ones(shape=[10,12,13])
    data = multiview(data, 1)
    print("shape:", np.shape(data))