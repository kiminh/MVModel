import numpy as np
from matplotlib import pyplot as plt

random_image = np.random.random([100, 500])
# print(np.shape(random_image))

# plt.imshow(random_image, cmap='gray', interpolation='nearest')

# linear0 = np.linspace(0, 1, 2500).reshape((50, 50))
# linear1 = np.linspace(0, 255, 2500).reshape((50, 50)).astype(np.uint8)
#
# print("Linear0:", linear0.dtype, linear0.min(), linear0.max())
# print("Linear1:", linear1.dtype, linear1.min(), linear1.max())
#
# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.imshow(linear0, cmap='gray')
# ax1.imshow(linear1, cmap='gray')
# plt.show()

features_relu = np.load('../ignore/test_12p_vgg19_29epo_feature.npy')
features_tanh = np.load('../ignore/test_12p_vgg19_epo29_tanh7_feature.npy')


def show_model_features(index=0, type='tanh'):
    # do transformation
    data_relu = features_relu[index]
    data_tanh = features_tanh[index]

    max_f_relu, min_f_relu = np.max(data_relu), np.min(data_relu)
    max_f_tanh, min_f_tanh = np.max(data_tanh), np.min(data_tanh)
    if type == 'tanh' or type=='relu':
        for i in xrange(np.shape(data_relu)[0]):
            for j in xrange(np.shape(data_relu)[1]):
                data_relu[i][j] = normalization(min_f_relu, max_f_relu, data_relu[i][j])
        max_f_relu, min_f_relu = np.max(data_relu), np.min(data_relu)
        for i in xrange(np.shape(data_tanh)[0]):
            for j in xrange(np.shape(data_tanh)[1]):
                data_tanh[i][j] = normalization(min_f_tanh, max_f_tanh, data_tanh[i][j])
        max_f_tanh, min_f_tanh = np.max(data_tanh), np.min(data_tanh)
    for i in xrange(7):
        data_relu = np.concatenate((data_relu, data_relu))
    for i in xrange(7):
        data_tanh = np.concatenate((data_tanh, data_tanh))

    fig_relu, ax_relu = plt.subplots()
    fig_relu.suptitle("relu")
    cax_relu = ax_relu.imshow(data_relu, cmap='gray', interpolation='nearest')
    cbar_relu = fig_relu.colorbar(cax_relu, ticks=[min_f_relu, max_f_relu])
    cbar_relu.ax.set_yticklabels([str(min_f_relu), str(max_f_relu)])  # vertically oriented colorbar

    fig_tanh, ax_tanh = plt.subplots()
    fig_tanh.suptitle("tanh")
    cax_tanh = ax_tanh.imshow(data_tanh, cmap='gray', interpolation='nearest')
    cbar_tanh = fig_tanh.colorbar(cax_tanh, ticks=[min_f_tanh, max_f_tanh])
    cbar_tanh.ax.set_yticklabels([str(min_f_tanh), str(max_f_tanh)])

def normalization(min_v, max_v, v):
    return 1.0*(v-min_v)/(max_v-min_v)

# 4, 7, 12
# 64,65,66
if __name__ == '__main__':
    show_model_features(index=66)
    plt.show()
