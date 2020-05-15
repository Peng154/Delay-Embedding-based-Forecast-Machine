import numpy as np


def get_y_from_matrix(config, matrix, weighted=False):
    """
    get the predicted values of target variable with prediction length L-1
    :param config:
    :param matrix: shape=[L, m]
    :param weighted: whether to calculate the mean value with different weights.
    :return: predicted values of target variable
    """
    predict_y = []

    for i in range(config.EMBEDDING_LEN - 1):
        y = []
        start_col = config.TRAIN_LEN - config.EMBEDDING_LEN + i + 1
        # print(start_col)
        for j in range(config.EMBEDDING_LEN - 1 - i):
            y.append(matrix[config.EMBEDDING_LEN - 1 - j, start_col + j])
        # print(len(y))
        if weighted:
            y_count = len(y)
            y = np.array(y)
            weights = np.arange(1, y_count+1, dtype=np.float32) / np.sum(np.arange(1, y_count+1, dtype=np.float32))
            # print(weights)
            # print(weights.sum())
            predict_y.append(np.sum(weights * y))
        else:
            predict_y.append(np.mean(y))
    return np.array(predict_y)
