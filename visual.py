import numpy as np
import os
from skimage import io
import skimage

modelnet_path_train = '/home3/lxh/modelnet40v1_2/train/'
modelnet_path_test = '/home3/lxh/modelnet40v1_2/test/'

save_path = '/home1/shangmingyang/data/3dmodel/classify_wrong'

train_files = os.listdir(modelnet_path_train)
train_files.sort()
test_files = os.listdir(modelnet_path_test)
test_files.sort()

class_label = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flowerpot',\
        'glassbox','guitar','keyboard','lamp','laptop','mantel','monitor','nightstand','person','piano','plant','radio','rangehood','sink',\
        'sofa','stairs','stool','table','tent','toilet','tvstand','vase','wardrobe','xbox']

data = np.load('classify_result.npy').tolist()
for i in xrange(len(data)):
    m = data[i]
    output_label, gb_label, output_prob, gb_prob = int(m[0]), int(m[1]), m[2], m[3]

    if gb_label != output_label:
        pre_filename = test_files[i * 12]
        save_filename = pre_filename[: pre_filename.find('.jpg')] + '_' + class_label[output_label] + '_' + str(output_prob) + '_' + str(gb_prob) + '.jpg'
        print("save to ", save_filename)
        pre_image = skimage.io.imread(os.path.join(modelnet_path_test, pre_filename))
        skimage.io.imsave(os.path.join(save_path, save_filename), pre_image)


