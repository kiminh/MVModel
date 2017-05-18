
class TrainingConfig(object):
    def __init__(self):
        self.train_inception_learning_rate = 0.0005
        self.max_checkpoints_to_keep = 5


class ModelConfig(object):
    def __init__(self):
        self.n_fcs = 4096 #dimension of feature vector
        self.n_views = 12 #number of model views