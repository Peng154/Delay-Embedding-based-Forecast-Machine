from configs import config


class Lorenz96Config(config.Config):
    def __init__(self):
        super(Lorenz96Config, self).__init__()

        # ##################
        # # for lorenz 96 #
        # ##################
        self.MODEL_NAME = 'lorenz96'  # the name of data

        self.DROP_RATE = 0

        self.EMBEDDING_LEN = 19

        self.K = 60

        self.F = 5

        self.TRAIN_LEN = 40

        self.SPATIAL_NODES = [512, 256, self.K]

        self.MERGE_MAP_NODES = [512, 256, 128, self.EMBEDDING_LEN]

        self.Y_IDX = 0

        self.MODULE_LAST_ACITVATION = False

        self.DATASET_RATE = 0

        self.TRAINING = True

        self.ENCODING_LAYER_NUMS = 2  

        self.TEMPORAL_DIM = self.K 

        self.DIFF = self.TEMPORAL_DIM * 4

        self.NUM_HEADS = 3

        self.BATCH_SIZE = 4

        self.EPOCHS = 85

        self.LOSS_WEIGHTS = {'consistent_loss': 1}


        self.MERGE_ONLY = False  

        self.MERGE_FUNC = 'concat' 

        self.LR = 1e-3

        self.WEIGHT_DECAY = 0 

        self.BN = False

        self.ADD_NOISE = False

        self.DATA_NOISE_STRENGTH = 0

        if self.MERGE_ONLY:
            self.MODEL_NAME = self.MODEL_NAME + '_merge_only'

    def LR_SCHEDULER(self, epoch):
        if epoch <= 20:
            return self.LR
        elif epoch <= 40:
            return self.LR / 3.0
        elif epoch <= 65:
            return self.LR / 10.0
        else:
            return self.LR / 30.0


    @property
    def name(self):
        if self.ADD_NOISE:
            return '{}_{}_{}_Yidx_{}_noise_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX, self.DATA_NOISE_STRENGTH)
        else:
            return '{}_{}_{}_Yidx_{}'.format(self.MODEL_NAME, self.TRAIN_LEN, self.EMBEDDING_LEN, self.Y_IDX)
