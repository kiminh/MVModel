from scipy.spatial.distance import euclidean, cosine
import numpy as np
from rank_metrics import mean_average_precision

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

def generate_labels_all(labels_test_file, labels_train_file, save_all_file):
    labels_test, labels_train = np.load(labels_test_file), np.load(labels_train_file)
    labels_all = np.concatenate((labels_test, labels_train))
    np.save(save_all_file, labels_all)

def retrival_metrics_all(sims_file, labels_file):
    sims, labels = np.load(sims_file), np.load(labels_file)
    #sims = np.array([[2,0.5,1],[0.5,3,2],[1,2,4]])
    #labels = np.array([0,1,0])
    mAP_input = np.zeros(sims.shape)
    for i in xrange(sims.shape[0]):
        sims_row, curr_label = sims[i], labels[i]
        labels_row = labels[np.argsort(sims_row)]
        for j in xrange(labels_row.shape[0]):
            if labels_row[j] == curr_label:
                mAP_input[i][j] = 1
    print mAP_input
    print("mAP:%f" %mean_average_precision(mAP_input.tolist()))


if __name__ == '__main__':
    #retrival_all_distance('/home/shangmingyang/projects/MVModel/rnn/test_hidden.npy', '/home/shangmingyang/projects/MVModel/rnn/train_hidden.npy', 'all_all_euclidean')
    #generate_labels_all('/home3/lhl/modelnet40_total_v2/test_label.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels')
    retrival_metrics_all('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_all_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels.npy')
