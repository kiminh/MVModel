from scipy.spatial.distance import euclidean, cosine
import numpy as np

def retrival_distance(test_feature_file, train_feature_file, savepath='test_train'):
    test_data, train_data = np.load(test_feature_file), np.load(train_feature_file)
    result = [euclidean(test, train) for train in train_data for test in test_data]
    result = np.reshape(np.array(result), [test_data.shape[0], train_data.shape[0]])
    np.save(savepath, result)

def retrival_all_distance(test_feature_file, train_feature_file, savepath='all_all'):
    test_data, train_data = np.load(test_feature_file), np.load(train_feature_file)
    all_data = np.concatenate((test_data, train_data), axis=0)
    result = [euclidean(test1, test2) for test2 in all_data for test1 in all_data]
    np.save(savepath, result)

if __name__ == '__main__':
    retrival_all_distance('/home/shangmingyang/projects/MVModel/rnn/test_hidden.npy', '/home/shangmingyang/projects/MVModel/rnn/train_hidden.npy', 'all_all_euclidean')


