class Config(object):

    def __init__(self):
        ##############################
        # the config parameters related to data
        ##############################

        self.TRAIN_LEN = 60  # the training length of input data

        self.EMBEDDING_LEN = 19  # the length of embedding, generally the prediction length is EMBEDDING_LEN- 1

        self.K = 90  # the dimension of input data

        self.Y_IDX = 0  # the idx of target variable to be predicted

        ##############################
        # the config parameters related to model and training
        ##############################
        self.MODEL_NAME = None

        self.LR = None  # the initial learning rate

        self.WEIGHT_DECAY = None  # the weight of L2-normalization

        self.ACITVATION = 'relu'  # the type of activation function

        self.KERNEL_INITIALIZER = 'glorot_normal'  # the way of initializing the inner weights

        self.FROM_EPOCH = 0  # which epoch to load, always used during continuing the train process from last checkpoint

        self.EPOCHS = 100  # total epoches to train

        self.BATCH_SIZE = 4  # batch size

        self.BN = False  # whether to make bach normalization

        self.LR_DECAY = 1e-5  # decay of learning rate

        self.VALIDATION_STEPS = 25

