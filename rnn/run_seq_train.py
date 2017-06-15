import os

try:
    os.system('python seq_train.py --data_path=%s --data_dir=%s --train_feature_file=%s --train_label_file=%s --test_feature_file=%s --test_label_file=%s'
              %('/home/shangmingyang/PycharmProjects/MVModel/ignore/data/', '/home/shangmingyang/PycharmProjects/MVModel/ignore/data/', 'example_feature.npy', 'example_label.npy', 'example_feature.npy', 'example_label.npy'))
except Exception as e:
    print("Exception:", e)
