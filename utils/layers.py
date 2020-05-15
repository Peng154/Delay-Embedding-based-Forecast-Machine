# import keras
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from tensorflow.python.keras import layers as tf_layers
# import keras.engine as KE
# from keras_layer_normalization import LayerNormalization
import numpy as np
import tensorflow as tf


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def get_same_idxs(sample_time_len, embedding_len):
    """

    :param sample_time_len:
    :param embedding_len:
    :return:
    """
    same_idxs = []
    if embedding_len > sample_time_len:
        for start_col_idx in range(1, embedding_len):
            idxs = []
            row_num = min(start_col_idx + 1, sample_time_len)
            for row_idx in range(row_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = row_idx
                idx[0, 1] =start_col_idx - row_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        for start_col_idx in range(embedding_len - sample_time_len + 1, embedding_len - 1):
            idx_count = embedding_len - start_col_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 0] = sample_time_len - 1 - i
                idx[0, 1] = start_col_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))
    else:
        for start_row_idx in range(1, sample_time_len):
            idxs = []
            col_num = min(start_row_idx+1, embedding_len)
            for col_idx in range(col_num):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = col_idx
                idx[0, 0] = start_row_idx - col_idx
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

        for start_row_idx in range(sample_time_len - embedding_len + 1, sample_time_len - 1):
            idx_count = sample_time_len - start_row_idx
            idxs = []
            for i in range(idx_count):
                idx = np.zeros((1, 2), dtype=np.int32)
                idx[0, 1] = embedding_len - 1 - i
                idx[0, 0] = start_row_idx + i
                idxs.append(idx)
            same_idxs.append(np.concatenate(idxs))

    return same_idxs


def get_known_mask(time_len, embedding_len):
    """
    get the mask of output embedding matrix for computing the known message loss
    :param time_len:
    :param embedding_len:
    :return:
    """
    mask = np.ones(shape=[time_len, embedding_len], dtype=np.float32)

    for i in range(embedding_len - 1):
        mask[time_len - embedding_len + 1 + i, -(i+1):] = 0.0

    return mask


class Matrix2Y(KL.Layer):
    def __init__(self, config, **kwargs):
        """
        get the value of target variable from delay embedding matrix
        :param config:
        :param kwargs:
        """
        super(Matrix2Y, self).__init__(**kwargs)
        self.config = config
        self.same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)[config.TRAIN_LEN - 1:]
        # print(len(self.same_idxs))

    def cal_mean_y(self, delay_matrix):
        predict_y = []
        for i, same_idx in enumerate(self.same_idxs):
            same_y_s = tf.gather_nd(delay_matrix, same_idx)
            mean_y = tf.reduce_mean(same_y_s)
            predict_y.append(mean_y)
        predict_y.append(delay_matrix[-1, -1])
        predict_y = tf.stack(predict_y)
        # print(predict_y.shape)
        return predict_y

    def call(self, inputs, **kwargs):
        return batch_slice(inputs, self.cal_mean_y, self.config.BATCH_SIZE)


class PredictYLoss(KL.Layer):
    def call(self, inputs, **kwargs):
        y_true, y_pred = inputs
        loss = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
        return loss


class KnownYLoss(KL.Layer):
    def __init__(self, config, **kwargs):
        """
        :param config:
        :param kwargs:
        """
        super(KnownYLoss, self).__init__(**kwargs)
        self.config = config
        self.mask = tf.constant(get_known_mask(self.config.TRAIN_LEN, self.config.EMBEDDING_LEN))

    def call(self, inputs, **kwargs):
        y_true, y_pred = inputs
        errors = tf.reduce_mean(tf.square(y_true - y_pred), axis=0)
        errors = tf.multiply(errors, self.mask)
        return tf.reduce_mean(errors)


class TimeConsistentLoss(KL.Layer):
    def __init__(self, config, **kwargs):
        super(TimeConsistentLoss, self).__init__(**kwargs)
        self.config = config
        self.same_idxs = get_same_idxs(config.TRAIN_LEN, config.EMBEDDING_LEN)

    def time_consistent_loss(self, delay_matrix):
        """

        :param delay_matrix: [time_len, embedding_len]
        :return:
        """
        consistent_loss = []
        for i, same_idx in enumerate(self.same_idxs):
            same_y_s = tf.gather_nd(delay_matrix, same_idx)
            mean_y = tf.reduce_mean(same_y_s)
            loss = tf.reduce_mean(tf.square(same_y_s - mean_y))
            consistent_loss.append(loss)

        consistent_loss = tf.stack(consistent_loss)
        consistent_loss = tf.reduce_mean(consistent_loss)
        return consistent_loss

    def call(self, inputs, **kwargs):
        batch_time_consistent_loss = K.mean(batch_slice(inputs, self.time_consistent_loss, self.config.BATCH_SIZE))
        # self.add_loss(batch_time_consistent_loss, inputs=inputs)
        return batch_time_consistent_loss


def scaled_dot_product_attention(q, k, v, mask):

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    #  matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(KL.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        # self.wq = KL.Dense(d_model)
        # self.wk = KL.Dense(d_model)
        # self.wv = KL.Dense(d_model)
        #
        # self.dense = KL.Dense(d_model)

    def build(self, input_shape):
        # print(input_shape)
        self.wq = self.add_weight(name='q_kernel',
                                  shape=(input_shape[0][-1], self.d_model),
                                  initializer='glorot_normal',
                                  trainable=True)
        self.q_bias = self.add_weight(name='q_bias',
                                      shape=(self.d_model,),
                                      initializer='zeros',
                                      trainable=True)

        self.wk = self.add_weight(name='k_kernel',
                                  shape=(input_shape[0][-1], self.d_model),
                                  initializer='glorot_normal',
                                  trainable=True)
        self.k_bias = self.add_weight(name='k_bias',
                                      shape=(self.d_model,),
                                      initializer='zeros',
                                      trainable=True)

        self.wv = self.add_weight(name='v_kernel',
                                  shape=(input_shape[0][-1], self.d_model),
                                  initializer='glorot_normal',
                                  trainable=True)
        self.v_bias = self.add_weight(name='v_bias',
                                      shape=(self.d_model,),
                                      initializer='zeros',
                                      trainable=True)

        self.dense_w = self.add_weight(name='dense_kernel',
                                  shape=(input_shape[0][-1], self.d_model),
                                  initializer='glorot_normal',
                                  trainable=True)
        self.dense_bias = self.add_weight(name='dense_bias',
                                          shape=(self.d_model,),
                                          initializer='zeros',
                                          trainable=True)


    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # def call(self, v, k, q, mask):
    def call(self, inputs, **kwargs):
        v, k, q = inputs

        batch_size = tf.shape(q)[0]

        q = K.dot(q, self.wq) + self.q_bias  # (batch_size, seq_len, d_model)
        k = K.dot(k, self.wk) + self.k_bias  # (batch_size, seq_len, d_model)
        v = K.dot(v, self.wv) + self.v_bias  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, _ = scaled_dot_product_attention(
            q, k, v, None)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = K.dot(concat_attention, self.dense_w) + self.dense_bias  # (batch_size, seq_len_q, d_model)

        return output


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      KL.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      KL.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def feed_forword_graph(d_model, dff, x):
    x = KL.Dense(dff, activation='relu')(x)  # (batch_size, seq_len, dff)
    x = KL.Dense(d_model)(x)  # (batch_size, seq_len, d_model)
    return x


def encoder_graph(d_model, num_heads, dff, training, rate, x):
    attn_output = MultiHeadAttention(d_model, num_heads)([x, x, x])  # (batch_size, input_seq_len, d_model)
    attn_output = KL.Dropout(rate=rate)(attn_output, training=training)
    out1 = tf_layers.LayerNormalization(epsilon=1e-6)(KL.Add()([x, attn_output]))  # (batch_size, input_seq_len, d_model)
    # out1 = x + attn_output  # (batch_size, input_seq_len, d_model)
    #
    ffn_output = feed_forword_graph(d_model, dff, out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = KL.Dropout(rate=rate)(ffn_output, training=training)
    out2 = tf_layers.LayerNormalization(epsilon=1e-6)(KL.Add()([out1, ffn_output]))  # (batch_size, input_seq_len, d_model)
    # # out2 = out1 + ffn_output  # (batch_size, input_seq_len, d_model)
    return out2


class EncoderLayer(KL.Layer):
    def __init__(self, d_model, num_heads, dff, training, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.training = training

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf_layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf_layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = KL.Dropout(rate)
        self.dropout2 = KL.Dropout(rate)

    # def call(self, x, training):
    def call(self, inputs, **kwargs):
        x = inputs
        print(x.shape)
        attn_output = self.mha([x, x, x])  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=self.training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # out1 = x + attn_output  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=self.training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        # out2 = out1 + ffn_output  # (batch_size, input_seq_len, d_model)

        return out2


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionEncodingLayer(KL.Layer):
    def __init__(self, d_model, maximum_position_encoding, **kwargs):
        """
        add position encoding to the data for self-attention
        :param d_model:
        :param maximum_position_encoding:
        :param kwargs:
        """
        super(PositionEncodingLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    def call(self, inputs, **kwargs):
        x = inputs
        seq_len = tf.shape(x)[1]

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        return x

class Encoder(KL.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, training, rate=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.training = training

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, self.training, rate)
                           for _ in range(num_layers)]

        self.dropout = KL.Dropout(rate)

    # def call(self, x, training, mask):
    def call(self, inputs, **kwargs):
        x = inputs
        seq_len = tf.shape(x)[1]

        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)