import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.learn.python.learn.datasets import base

# server config
# tf.flags.DEFINE_string('train_data_dir', '/home3/lxh/modelnet/modelnet40v1_2/train', 'file path saving model features for training')
# tf.flags.DEFINE_string('train_label_file', 'train_label.npy', 'file path saving model labels for training')
# tf.flags.DEFINE_string('test_data_dir', '/home3/lxh/modelnet/modelnet40v1_2/test', 'file path saving model features for testing')
# tf.flags.DEFINE_string('test_label_file', 'test_label.npy', 'file path saving model labels for testing')
tf.flags.DEFINE_string('train_data_dir', '/home/shangmingyang/PycharmProjects/TF/modelnet/train', 'file path saving model features for training')
tf.flags.DEFINE_string('test_data_dir', '/home/shangmingyang/PycharmProjects/TF/modelnet/test', 'file path saving model features for testing')

FLAGS = tf.flags.FLAGS

class DataSet(object):
    def __init__(self, views, labels):
        self._views, self._labels = views, labels
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = labels.shape[0]

    def views(self):
        return np.array([self.get_all_view_by_first(v) for v in self._views]).reshape([-1])

    @property
    def labels(self):
        return self._labels

    def size(self):
        return np.shape(self._labels)[0]

    def get_all_view_by_first(self, first_view_name):
        prefix_name = first_view_name[ : first_view_name.find('001.jpg')]
        return [prefix_name + "%03d" %i + ".jpg" for i in range(1, 13)]


    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """
        Return the next `batch_size` examples from this data set.
        NOTE: the view's result size is [batch_size*12], because we should enrich one view to 12 views of a model
        """
        # TODO fake data when fake_data==True
        # if fake_data:
        #     fake_image = [1] * 784
        #     if self.one_hot:
        #         fake_label = [1] + [0] * 9
        #     else:
        #         fake_label = 0
        #     return [fake_image for _ in xrange(batch_size)], [
        #         fake_label for _ in xrange(batch_size)
        #         ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._views = self._views[perm0]
            self._labels = self._labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            views_rest_part = self._views[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._views = self._views[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            views_new_part = self._views[start:end]
            labels_new_part = self._labels[start:end]

            enriched_views = np.array([self.get_all_view_by_first(first_view) for first_view in np.concatenate((views_rest_part, views_new_part), axis=0)])
            return enriched_views.reshape([-1]), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            enriched_views = np.array([self.get_all_view_by_first(first_view) for first_view in self._views[start:end]])
            return enriched_views.reshape([-1]), self._labels[start:end]

def read_data(data_dir, n_views=12):
    print("read data from %s" %data_dir)
    train_views_files = os.listdir(FLAGS.train_data_dir)
    train_views_files.sort()
    train_views_files = train_views_files[::n_views]
    train_views_files = [os.path.join(FLAGS.train_data_dir, f) for f in train_views_files]
    train_numbers_in_class = [80 for i in range(40)]
    train_numbers_in_class[6], train_numbers_in_class[10] = 64, 79
    train_labels = [[j for i in range(train_numbers_in_class[j])] for j in range(40)]
    train_labels = sum(train_labels, [])
    train_labels = onehot(train_labels)

    test_views_files = os.listdir(FLAGS.test_data_dir)
    test_views_files.sort()
    test_views_files = test_views_files[::n_views]
    test_views_files = [os.path.join(FLAGS.test_data_dir, f) for f in test_views_files]
    test_labels = [[j for i in range(20)] for j in range(40)]
    test_labels = sum(test_labels, [])
    test_labels = onehot(test_labels)

    train_dataset = DataSet(np.array(train_views_files), train_labels)
    test_dataset = DataSet(np.array(test_views_files), test_labels)

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
    train_views = np.zeros(shape=[2, 12, 13])
    train_labels = np.ones(shape=[2])

    test_views = np.ones(shape=[2,12,13])
    test_labels = np.zeros(shape=[2])

    np.save(os.path.join(data_dir, FLAGS.train_feature_file[ : FLAGS.train_feature_file.find('.')]), train_views)
    np.save(os.path.join(data_dir, FLAGS.train_label_file[ : FLAGS.train_label_file.find('.')]), train_labels)

    np.save(os.path.join(data_dir, FLAGS.test_feature_file[ : FLAGS.test_feature_file.find('.')]), test_views)
    np.save(os.path.join(data_dir, FLAGS.test_label_file[ : FLAGS.test_label_file.find('.')]), test_labels)


def run_readdata_demo(data_dir):
    model_data = read_data(data_dir)
    train_data = model_data.train
    test_data = model_data.test
    test_views, test_labels = test_data.next_batch(1)
    print(test_views)
    print("views:", np.shape(test_views))
    print("labels:", np.shape(test_labels))

if __name__ == '__main__':
    # _fake_write_data(FLAGS.data_dir) #only run once to fake data
    run_readdata_demo("")
    # data = np.load(os.path.join(FLAGS.data_dir, FLAGS.train_feature_file))
