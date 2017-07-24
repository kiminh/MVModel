import os
checkpoint_file = "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/modelnet10_6views/checkpoint"
with open(checkpoint_file) as f:
    data = f.readlines()[1:]
    data = [line.split(":")[1] for line in data]
    data = [line[2: -2] for line in data]

print data

for model_path in data:
    os.system('python train.py --train=False --n_views=6 --seq_embeddingmvmodel_path=%s --test_acc_file=%s --n_hidden=%d --decoder_embedding_size=%d --enrich_data=%s --use_lstm=False' %(model_path, '/home1/shangmingyang/data/3dmodel/mvmodel_result/modelnet10/modelnet10_6views.csv', 128, 256, "False"))
