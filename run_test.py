import os
checkpoint_file = "/home1/shangmingyang/data/3dmodel/trained_mvmodel/1.0_0.5_4/checkpoint"
with open(checkpoint_file) as f:
    data = f.readlines()[1:]
    data = [line.split(":")[1] for line in data]
    data = [line[2: -2] for line in data]

data=data[6:]
print data

for model_path in data:
    os.system('python test.py --test_model_path=%s >>test_result_1.0_0.5.txt' %(model_path))
