from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import auc
import numpy as np
from rank_metrics import mean_average_precision, average_precision

def retrival_distance(test_feature_file, train_feature_file, savepath='test_train'):
    test_data, train_data = np.load(test_feature_file), np.load(train_feature_file)
    result = [euclidean(test, train) for test in test_data for train in train_data]
    result = np.reshape(np.array(result), [test_data.shape[0], train_data.shape[0]])
    np.save(savepath, result)

def generate_distance_test2test(test_feature_file, savepath='test_test'):
    test_data = np.load(test_feature_file)
    result = [euclidean(test1, test2) for test1 in test_data for test2 in test_data]
    result = np.reshape(np.array(result), [test_data.shape[0], test_data.shape[0]])
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
    #sims = np.array([[0,0.5,1],[0.5,0,2],[1,2,0]])
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

def retrival_metrics_test2train(sims_file, test_labels_file, train_labels_file):
    sims, labels_test, labels_train = np.load(sims_file), np.load(test_labels_file), np.load(train_labels_file)
    mAP_input = np.zeros(sims.shape)
    for i in xrange(sims.shape[0]):
        sims_row, curr_label = sims[i], labels_test[i]
        labels_row = labels_train[np.argsort(sims_row)]
        for j in xrange(labels_row.shape[0]):
            if labels_row[j] == curr_label:
                mAP_input[i][j] = 1
    print mAP_input
    print("mAP:%f" %mean_average_precision(mAP_input.tolist()))

def PR_test2test(sims_file, labels_file):
    sims, labels = np.load(sims_file), np.load(labels_file)
    Ps, Rs = [], []
    for i in xrange(sims.shape[0]):
        sims_row, curr_label = sims[i], labels[i]
        labels_row = labels[np.argsort(sims_row)]
        y_true = [1 if curr_label==l else 0 for l in labels_row]
        y_pred = [1 for _ in xrange(len(y_true))]
        P, R = PR(y_true, y_pred)
        Ps.append(P)
        Rs.append(R)
    Ps, Rs = np.array(Ps), np.array(Rs)
    mean_P, mean_R = np.mean(Ps, axis=0), np.mean(Rs, axis=0)
    area = auc(mean_R, mean_P)
    np.save("P", mean_P)
    np.save("R", mean_R)
    return Ps, Rs, area

def PR(y_true, y_pred):
    """
    generate precison and recall array for drawing PR-curve
    y_true: array, value is 0 or 1, 1 means the model has same class with query
    y_pred: array, value is 0 or 1, 1 means the model has same class with query
    @return:
        P: array, size is 1 bigger than y_true and y_pred, first is 1
        R: array, size is 1 bigger than y_true and y_pred, first is 0
    example:
        y_true=[1,1,0,0]
        y_pred=[1,0,1,0]
        P=[1,1,1,1/2,1/2]
        R=[0,1/2,1/2,1/2,1/2]
    """
    P, R = [], []
    sum_true, TP = 0, 0
    TP, FP = 0, 0
    for gd, pred in zip(y_true, y_pred):
        if gd == pred == 1:
            TP += 1
        elif pred == 1 and gd == 0:
            FP += 1
        if gd == 1:
            sum_true += 1
        P.append(TP*1.0/(TP+FP))
        R.append(TP)
    R = [r*1.0/sum_true for r in R]
    P, R = [1.0] + P, [0.0] + R
    return P, R

def PR_curve(P, R):
    import pylab as pl
    pl.plot(R, P)
    pl.show()

if __name__ == '__main__':
    #retrival_distance('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/train_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_train_euclidean')
    #generate_distance_test2test('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/train_hidden.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/train_train_euclidean')
    #retrival_all_distance('/home/shangmingyang/projects/MVModel/rnn/test_hidden.npy', '/home/shangmingyang/projects/MVModel/rnn/train_hidden.npy', 'all_all_euclidean')
    #generate_labels_all('/home3/lhl/modelnet40_total_v2/test_label.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels')
    #retrival_metrics_all('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_all_euclidean.npy', '/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/all_labels.npy')
    #retrival_metrics_test2train('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_train_euclidean.npy', '/home3/lhl/modelnet40_total_v2/test_label.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy')
    #retrival_metrics_all('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/train_train_euclidean.npy', '/home3/lhl/modelnet40_total_v2/train_label.npy')
    _,_,area=PR_test2test('/home1/shangmingyang/data/3dmodel/mvmodel_result/retrival/test_test_euclidean.npy', '/home3/lhl/modelnet40_total_v2/test_label.npy')
    print area
