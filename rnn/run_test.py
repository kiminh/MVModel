import os
checkpoint_file = "/home1/shangmingyang/data/3dmodel/trained_seq_mvmodel/embedding/checkpoint"
with open(checkpoint_file) as f:
    data = f.readlines()[1:]
    data = [line.split(":")[1] for line in data]
    data = [line[2: -2] for line in data]

#data=data[6:]
print data

for model_path in data:
    os.system('python train.py --train=False --seq_embeddingmvmodel_path=%s --test_acc_file=%s --n_hidden=%d --decoder_embedding_size=%d --enrich_data=%si --use_lstm=False' %(model_path, 'embedding.csv', 128, 4096, "False"))
