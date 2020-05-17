import tensorflow.keras.models as KM
import tensorflow as tf
import re
import numpy as np
import tensorflow.keras.layers as KL
from utils import layers
import os
import datetime

def time_distributed_graph(input, nodes, last_activation, last_bn, activation, name_prefix,
                           kernel_initialzer, weight_decay, bn):

    nodes_count = len(nodes)

    x = input
    for i in range(nodes_count - 1):
        x = KL.TimeDistributed(KL.Dense(nodes[i],
                                        activation=activation,
                                        kernel_initializer=kernel_initialzer,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                        name=name_prefix + '_Dense{}'.format(i+1)))(x)
        # x = LayerNormalization(epsilon=1e-6)(x)

    activation = activation if last_activation else None  
    x = KL.TimeDistributed(KL.Dense(nodes[-1],
                                    activation=activation,
                                    kernel_initializer=kernel_initialzer,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                    name=name_prefix + '_Dense{}'.format(nodes_count)))(x)
    # if last_bn:
    #     x = LayerNormalization(epsilon=1e-6)(x)  

    return x


class DataGeneratorForLengthCmp(tf.keras.utils.Sequence):
    def __init__(self, data_x, y, idxs, config, shuffle=True):
        self.data_x = data_x
        self.y = y
        self.idxs = idxs
        self.config = config
        self.shuffle = shuffle

    def __len__(self):
        return len(self.idxs) // self.config.BATCH_SIZE

    def __getitem__(self, item):
        batch = self.get_batch(item)

        return [batch[0][0], batch[1][0], batch[2]], []

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []
        # print('t_idx:{}'.format(t_idx))
        for i in range(self.config.TRAIN_LEN):
            try:
                y_matrix.append(self.data_x[
                                t_idx - self.config.TRAIN_LEN + i:
                                t_idx - self.config.TRAIN_LEN + i + self.config.EMBEDDING_LEN, self.config.Y_IDX])
            except Exception:
                print(self.data_x.shape)
                print(t_idx - self.config.TRAIN_LEN + i, t_idx - self.config.TRAIN_LEN + i + self.config.EMBEDDING_LEN, self.config.Y_IDX)
        y_matrix = np.stack(y_matrix)

        x = self.data_x[t_idx - self.config.TRAIN_LEN: t_idx]
        return x, y_matrix, self.y[t_idx: t_idx + self.config.EMBEDDING_LEN - 1]


    def get_batch(self, b_idx):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(self.config.BATCH_SIZE):
            t_idx = self.idxs[b_idx * self.config.BATCH_SIZE + i]
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def get_long_term_item(self, iterations_count):
        for i in range(len(self.idxs)):
            yield self.get_long_term_batch(i, iterations_count)

    def get_long_term_batch(self, b_idx, iteration_count):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(iteration_count):
            t_idx = self.idxs[b_idx] + i * (self.config.EMBEDDING_LEN - 1)
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_x, y, idxs, config, shuffle=True):
        self.data_x = data_x
        self.y = y
        self.idxs = idxs
        self.config = config
        self.shuffle = shuffle

    def __len__(self):
        return len(self.idxs) // self.config.BATCH_SIZE

    def __getitem__(self, item):
        batch = self.get_batch(item)

        return [batch[0][0], batch[1][0], batch[2]], []

    def get_item(self):
        for i in range(len(self.idxs) // self.config.BATCH_SIZE):
            yield self.get_batch(i)

    def get_sample(self, t_idx):
        y_matrix = []
        # print('t_idx:{}'.format(t_idx))
        for i in range(self.config.TRAIN_LEN):
            try:
                y_matrix.append(self.data_x[t_idx + i: t_idx + i + self.config.EMBEDDING_LEN, self.config.Y_IDX])
            except BaseException:
                print(self.data_x.shape)
                print(t_idx + i, t_idx + i + self.config.EMBEDDING_LEN, self.config.Y_IDX)
        y_matrix = np.stack(y_matrix)

        x = self.data_x[t_idx: t_idx + self.config.TRAIN_LEN]
        return x, y_matrix, self.y[t_idx + self.config.TRAIN_LEN:
                                          t_idx + self.config.TRAIN_LEN +
                                          self.config.EMBEDDING_LEN - 1]

    def get_batch(self, b_idx):
        xs = []
        y_matrixs = []
        ys = []
        for i in range(self.config.BATCH_SIZE):
            t_idx = self.idxs[b_idx * self.config.BATCH_SIZE + i]
            x, y_matrix, y = self.get_sample(t_idx)
            xs.append(x)
            y_matrixs.append(y_matrix)
            ys.append(y)
        xs = np.stack(xs)
        y_matrixs = np.stack(y_matrixs)
        ys = np.stack(ys)

        return [xs], [y_matrixs], ys

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)


class STDelayModel(object):
    def __init__(self, config, mode='training', log_dir_suffix=None):
        self.config = config
        self.mode = mode
        self.model = self.build_model()
        self.model.summary()
        self.epoch = 0
        self.log_dir_suffix = log_dir_suffix
        self.log_dir = self.get_log_dir()


    def build_model(self):
        t_layer = KL.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]), name='t_layer')

        input = KL.Input(shape=(self.config.TRAIN_LEN, self.config.K), name='input', dtype=tf.float32)

        if self.mode == 'training':
            gt_y_matrix = KL.Input(shape=(self.config.TRAIN_LEN, self.config.EMBEDDING_LEN), name='gt_y_matrix',
                                   dtype=tf.float32)
            gt_y = KL.Input(shape=(self.config.EMBEDDING_LEN - 1,), name='gt_y', dtype=tf.float32)

        input_drop = KL.SpatialDropout1D(rate=self.config.DROP_RATE, name='input_drop')(input) 
        # input_T = t_layer(input_drop)

        if not self.config.MERGE_ONLY: 
            spatial_features = time_distributed_graph(input_drop, self.config.SPATIAL_NODES,
                                                      last_activation=self.config.MODULE_LAST_ACITVATION,
                                                      last_bn=True, activation=self.config.ACITVATION,
                                                      name_prefix='spatial',
                                                      kernel_initialzer=self.config.KERNEL_INITIALIZER,
                                                      weight_decay=self.config.WEIGHT_DECAY, bn=self.config.BN)

            positioned_input = layers.PositionEncodingLayer(self.config.TEMPORAL_DIM,
                                                            self.config.TRAIN_LEN)(input_drop)

            temporal_features = positioned_input
            for _ in range(self.config.ENCODING_LAYER_NUMS):
                temporal_features = layers.encoder_graph(self.config.TEMPORAL_DIM,
                                                         self.config.NUM_HEADS,
                                                         self.config.DIFF,
                                                         self.config.TRAINING,
                                                         rate=0,
                                                         x=temporal_features)

            merge_features = None
            if self.config.MERGE_FUNC == 'add':
                assert self.config.TEMPORAL_NODES[-1] == self.config.TRAIN_LEN and \
                       self.config.SPATIAL_NODES[-1] == self.config.K, 'if add features, dimensions must be equal'
                merge_features = KL.Add(name='merge_features')([spatial_features, temporal_features])
            elif self.config.MERGE_FUNC == 'concat':
                # merge_features = KL.Concatenate(name='merge_features')([spatial_features, temporal_features_t])
                merge_features = KL.Concatenate(name='merge_features')([spatial_features, temporal_features])
            else:
                print('unknown func')
                exit(0)
        else:  
            merge_features = input_drop

        delay_output = time_distributed_graph(merge_features, self.config.MERGE_MAP_NODES, last_activation=False,
                                              last_bn=False, activation=self.config.ACITVATION,
                                              name_prefix='merge_delay',
                                              kernel_initialzer=self.config.KERNEL_INITIALIZER,
                                              weight_decay=self.config.WEIGHT_DECAY, bn=self.config.BN)

        predict_y = layers.Matrix2Y(self.config, name='predict_y')(delay_output)

        if self.mode == 'training':
            known_y_loss = layers.KnownYLoss(self.config, name='known_y_loss')([gt_y_matrix, delay_output])
            
            consistent_loss = layers.TimeConsistentLoss(self.config, name='consistent_loss')(delay_output)

            predict_y_loss = layers.PredictYLoss(name='predict_y_loss')([gt_y, predict_y])

            return KM.Model(inputs=[input, gt_y_matrix, gt_y], outputs=[delay_output, predict_y, known_y_loss,
                                                                  consistent_loss, predict_y_loss])

        else:
            return KM.Model(inputs=input, outputs=[delay_output, predict_y])

    def compile(self):
        print(self.model.outputs)
        optimizer = tf.keras.optimizers.Adam(lr=self.config.LR)

        losses = ['known_y_loss', 'consistent_loss']
        # print(self.model.losses)
        # add loss
        for loss_name in losses:
            layer = self.model.get_layer(loss_name)
            # if layer.output not in self.model.losses:
            loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.)
            self.model.add_loss(loss)
        print(self.model.losses)

        self.model.compile(optimizer=optimizer, loss=[None for _ in self.model.outputs])

        losses.extend(['predict_y_loss'])
        # add loss metric
        for loss_name in losses:
            # self.model.metrics_names.append(loss_name)
            layer = self.model.get_layer(loss_name)
            loss = layer.output * self.config.LOSS_WEIGHTS.get(loss_name, 1.)
            # self.model.metrics_tensors.append(loss)
            self.model.add_metric(loss, name=loss_name, aggregation='mean')

        print(self.model.metrics_names)

    def load_weights(self, model_path):
        
        self.model.load_weights(model_path, by_name=True)

        regex = '(.*)/weights_epoch:(\d{2,5})_.*'
        m = re.match(regex, model_path)
        self.log_dir = m.group(1)
        self.epoch = int(m.group(2))

        print('log_dir', self.log_dir)
        print('start_epoch', self.epoch)

    def get_log_dir(self):
        time_stamp = datetime.datetime.now()
        suffix = '({})'.format(self.log_dir_suffix) if self.log_dir_suffix is not None else ''
        log_dir = '../../logs/' + time_stamp.strftime('%Y_%m_%d-%H_%M_%S') + suffix
        return log_dir

    def train(self, train_generator, val_generator):
        os.makedirs(self.log_dir, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                        write_images=False),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(
                    self.log_dir,
                    'weights_epoch:{epoch:04d}_loss:{loss:.3f}_val_loss:{val_loss:.3f}_predict_loss:{predict_y_loss:.3f}.h5'),
                verbose=0, save_weights_only=True, period=1),
            tf.keras.callbacks.LearningRateScheduler(self.config.LR_SCHEDULER),
        ]

        val_steps = self.config.VALIDATION_STEPS
        if val_generator is None:
            val_steps = None

        self.model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=self.config.EPOCHS,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=True
        )
