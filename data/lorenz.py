import numpy as np


def my_lorenz(n, time=100, step=0.02, c=0.1, time_invariant=True):
    """
    generate the lorenz data.
    :param n: the number of sub-systems
    :param time:
    :param step:
    :param c:
    :param time_invariant: whether time-invariant or time-variant
    :return: return a lorenz data matrix with size[n*3, time // step], n*3 is the dimension, and time // step
    is the time length.
    """
    length = int(time / step)  #
    x = np.zeros((n * 3, length), dtype=np.float32)  # initialize data matrix
    x[:, 0] = np.random.rand(n * 3)  # randomly initialze the values at first time point
    sigma = 10.0

    for i in range(1, length):

        if not time_invariant:
            sigma = 10.0 + 0.02 * (i % 100)

        x[0, i] = x[0, i - 1] + step * (sigma * (x[1, i - 1] - x[0, i - 1]) + c * x[(n - 1) * 3, i - 1])
        x[1, i] = x[1, i - 1] + step * (28 * x[0, i - 1] - x[1, i - 1] - x[0, i - 1] * x[2, i - 1])
        x[2, i] = x[2, i - 1] + step * (-8 / 3 * x[2, i - 1] + x[0, i - 1] * x[1, i - 1])

        for j in range(1, n):
            x[3 * j, i] = x[3 * j, i - 1] + step * (
                        10 * (x[3 * j + 1, i - 1] - x[3 * j, i - 1]) + c * x[3 * (j - 1), i - 1])
            x[3 * j + 1, i] = x[3 * j + 1, i - 1] + step * (
                        28 * x[3 * j, i - 1] - x[3 * j + 1, i - 1] - x[3 * j, i - 1] * x[3 * j + 2, i - 1])
            x[3 * j + 2, i] = x[3 * j + 2, i - 1] + step * (
                        -8 / 3 * x[3 * j + 2, i - 1] + x[3 * j, i - 1] * x[3 * j + 1, i - 1])

    return x

# if __name__ == '__main__':
#     temp = my_lorenz(30)
#     print(temp.shape)
#     print((temp[:, 0] == 0).sum())