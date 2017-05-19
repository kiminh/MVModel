
class TrainingConfig(object):
    def __init__(self):
        self.train_inception_learning_rate = 0.0005
        self.max_checkpoints_to_keep = 5
        self.batch_size = 1
        self.training_epoches = 100
        self.display_epoches = 8
        self.save_epoches = 10
        self.cnn_keep_prob = 1.0
        self.rnn_keep_prob = 1.0


class ModelConfig(object):
    def __init__(self):
        self.n_fcs = 4096 #dimension of feature vector
        self.n_views = 12 #number of model views

        # rnn config
        self.n_hidden = 128
        self.n_classes = 40
