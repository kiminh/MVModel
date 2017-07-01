import numpy as np
import itertools as it

from tensorflow.contrib.learn.python.learn.datasets import base

import data_utils

class FakeDataSet(object):
    def __init__(self, lan1, lan2):
        self.lan1, self.lan2 = lan1, lan2
        self._epochs_completed, self._index_in_epoch = 0, 0
        self._num_examples = lan1.shape[0]

    def size(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self.lan1 = self.lan1[perm0]
            self.lan2 = self.lan2[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            fcs_rest_part = self.lan1[start:self._num_examples]
            labels_rest_part = self.lan2[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self.lan1 = self.lan1[perm]
                self.lan2 = self.lan2[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            fcs_new_part = self.lan1[start:end]
            labels_new_part = self.lan2[start:end]

            return np.concatenate((fcs_rest_part, fcs_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.lan1[start:end], self.lan2[start:end]

    def label2sequence(self, label_onehot):
        label = np.argmax(label_onehot) + 1
        sequence = [data_utils.GO_ID]
        for i in xrange(1, np.shape(label_onehot)[0]+1):
            if label != i:
                sequence.append(2*i)
            else:
                sequence.append(2*i-1)
        return np.array(sequence)

    def batch_label2sequence(self, labels_onehot):
        return np.array([self.label2sequence(label_onehot) for label_onehot in labels_onehot])

def read_data(num=7, test_num=10):
    lan1 = list(it.permutations(range(1,num+1), num))
    for i in xrange(len(lan1)):
        lan1[i] = list(lan1[i])
    lan_map = zip(range(1, num+1), range(1, num+1)[::-1])
    data_size = len(lan1)

    def lan1_lan2(lan1):
        return [data_utils.GO_ID] + [lan_map[l-1][1] for l in lan1]
    train_lan1, test_lan1 = lan1[:data_size-test_num], lan1[-test_num:]
    train_lan2, test_lan2 = [lan1_lan2(l_train) for l_train in train_lan1], [lan1_lan2(l_test) for l_test in test_lan1]

    train_data = FakeDataSet(np.array(train_lan1), np.array(train_lan2))
    test_data = FakeDataSet(np.array(test_lan1), np.array(test_lan2))

    return base.Datasets(train=train_data, test=test_data, validation=None)

if __name__ == '__main__':
    data = read_data(num=7)
    print data.train.next_batch(2)
