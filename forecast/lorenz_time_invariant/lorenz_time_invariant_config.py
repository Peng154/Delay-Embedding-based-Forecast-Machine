from configs import config


class LorenzTimeInvariantConfig(config.Config):
    def __init__(self):
        super(LorenzTimeInvariantConfig, self).__init__()

        # ##################
        # # for lorenz #
        # ##################
        self.MODEL_NAME = 'lorenz'  # the name of data

        self.DROP_RATE = 0

        self.EMBEDDING_LEN = 19

        self.K = 90

        self.TRAIN_LEN = 40

        self.SPATIAL_NODES = [512, 256, self.K]

        self.MERGE_MAP_NODES = [512, 256, 128, self.EMBEDDING_LEN]

        self.Y_IDX = 0

        self.MODULE_LAST_ACITVATION = False

        self.DATASET_RATE = 0

        self.TRAINING = True  # whther is training

        self.ENCODING_LAYER_NUMS = 2  # the stack number of self-attention 

        self.TEMPORAL_DIM = self.K 

        self.DIFF = self.TEMPORAL_DIM * 4

        self.NUM_HEADS = 3

        self.BATCH_SIZE = 4

        self.EPOCHS = 85

        self.LOSS_WEIGHTS = {'consistent_loss': 1}

        # def LR_SCHEDULER(self, epoch):
        #     if epoch <= 10:
        #         return self.LR
        #     elif epoch <= 35:
        #         return self.LR / 3.0
        #     elif epoch <= 55:
        #         return self.LR / 10.0
        #     else:
        #         return self.LR / 30.0

        # ##################
        # #     end        #
        # ##################

        # self.KERNEL_INITIALIZER = 'truncated_normal'

        self.MERGE_ONLY = False  # whether (without temporal module) or full model

        self.MERGE_FUNC = 'concat'  # or 'add' , the way to aggregate spatial and temporal features 

        self.LR = 1e-3

        self.WEIGHT_DECAY = 0  

        self.BN = False

        # whether to add noise
        self.ADD_NOISE = False

        # the noise strength
        self.DATA_NOISE_STRENGTH = 0

        # modify model name
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
