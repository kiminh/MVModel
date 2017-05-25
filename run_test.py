import os
checkpoint_file = "/home1/shangmingyang/data/3dmodel/trained_mvmodel/0.9_0.9_4/checkpoint"
with open(checkpoint_file) as f:
    data = f.readlines()[1:]
    data = [line.split(":")[1] for line in data]
    data = [line[2: -2] for line in data]

data=data[-4:]
print data

for model_path in data:
    os.system('python test.py --test_model_path=%s >>test_result.txt' %(model_path))
