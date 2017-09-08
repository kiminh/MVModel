import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='modelnet10', help='dataset used to train and test')
args = parser.parse_args()


train_cmd_base = 'python train.py --train=True --n_hidden=512 --decoder_embedding_size=256 --n_views=12 --use_lstm=False --training_epoches=50 --save_epoches=1 --learning_rate=0.0001 --batch_size=32 --n_max_keep_model=100 '
train_cmd = train_cmd_base

data_paths = {"modelnet10": ["/home3/lhl/tensorflow-vgg-master-total/feature/train_12p_vgg19_epo48_do05_sigmoid7_feature_class10.npy", "/home3/lhl/modelnet10_v2/feature10/train_labels_modelnet10.npy", "/home3/lhl/tensorflow-vgg-master-total/feature/test_12p_vgg19_epo48_do05_sigmoid7_feature_class10.npy", "/home3/lhl/modelnet10_v2/feature10/test_labels_modelnet10.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/modelnet10/mvmodel.ckpt"],
                "modelnet40": ["/home3/lhl/tensorflow-vgg-master-total/feature/train_12p_vgg19_ori_feature_total.npy", "/home3/lhl/modelnet40_total_v2/train_label.npy", "/home3/lhl/tensorflow-vgg-master-total/feature/test_12p_vgg19_ori_feature_total.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/modelnet40/mvmodel.ckpt"],

                #"modelnet40": ["/home3/lhl/tensorflow-vgg-master-total/feature/train_12p_vgg19_epo10_do05_sigmoid7_feature_total.npy", "/home3/lhl/modelnet40_total_v2/train_label.npy", "/home3/lhl/tensorflow-vgg-master-total/feature/test_12p_vgg19_epo10_do05_sigmoid7_feature_total.npy", "/home3/lhl/modelnet40_total_v2/test_label.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/embedding/checkpoint/modelnet40"],
                "shapenet55": ["/home3/lhl/tensorflow-vgg-master-shapenet/feature/train_feature_SN55_epo17.npy", "/home1/shangmingyang/data/3dmodel/shapenet/shapenet55_v1_train_labels.npy", "/home3/lhl/tensorflow-vgg-master-shapenet/feature/test_feature_SN55_epo17.npy", "/home1/shangmingyang/data/3dmodel/shapenet/shapenet55_v1_test_labels.npy", "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/shapenet55_hidden512_0.0001/mvmodel.ckpt"]}

if args.dataset == "modelnet10":
    path = data_paths["modelnet10"]
    train_cmd = train_cmd_base + '--n_classes=10 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s'%(path[0], path[1], path[2], path[3], path[4])
elif args.dataset == "modelnet40":
    path = data_paths["modelnet40"]
    train_cmd = train_cmd_base + '--n_classes=40 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s'%(path[0], path[1], path[2], path[3], path[4])
elif args.dataset == 'shapenet55':
    path = data_paths["shapenet55"]
    train_cmd = train_cmd_base + '--n_classes=55 --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s --save_seq_embeddingmvmodel_path=%s'%(path[0], path[1], path[2], path[3], path[4])
else:
    print("dataset muse one of [modelnet10, modelnet40, shapenet55], can not be %s" %(FLAGS.dataset))
    sys.exit()
print(train_cmd)
os.system(train_cmd)

