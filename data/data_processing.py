import pickle
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
import os
import data.lorenz as lorenz

# TODO: you should change the base path of data dir
DATA_BASE_DIR = '/home/ph/projects/DEFM_published/logs/data'


def add_noise(raw_data, config):
    """
    add white noise to the data matrix
    :param raw_data: data matrix with size [m, K]
    :param config:
    :return: the data matrix with whtie noise noised_data:[m ,K], the original y raw_y:[m,]
    """
    raw_y = np.copy(raw_data[:, config.Y_IDX])
    if config.ADD_NOISE:
        np.random.seed(234)  # with same white noise
        noise = np.random.normal(size=raw_data.shape)  
        noise_data = raw_data + noise * config.DATA_NOISE_STRENGTH  # change the stength of noise
        return noise_data, raw_y
    else:
        return raw_data, raw_y


def get_data_idxs_as_predict_idx_no_overlap(rate=1, interval=100):
    """
    return the idxs of data, the idx is the start idx of to be predicted data,
    make sure all data has no overlap（train_len + pred_len < interval）
    :param rate: the rate of training data
    :param interval: the time point interval between two neighbouring data samples
    :return: train_idxs and val_idxs
    """

    np.random.seed(123)

    idxs = []
    for i in range(1050):
        idxs.append((i+1)*interval)
    idxs = np.array(idxs)
    np.random.shuffle(idxs)  # shuffle the idxs

    train_idxs = idxs[:1000]
    val_idxs = idxs[1000:]
    total_len = train_idxs.shape[0]
    print(total_len)

    count = int(total_len * rate)
    train_idxs = train_idxs[:count]
    return train_idxs, val_idxs


def get_data_idxs_for_lorenz(threshold=5.0, time_invariant=True, rate=1):
    np.random.seed(123)
    if not time_invariant:
        idxs = np.arange(2900)
        np.random.shuffle(idxs)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]
    else:
        idxs = np.arange(5000)
        np.random.shuffle(idxs)
        train_idxs = idxs[:1000]
        val_idxs = idxs[1000:1050]

    total_len = train_idxs.shape[0]
    print(total_len)

    count = int(total_len * rate)
    train_idxs = train_idxs[:count]
    return train_idxs, val_idxs


def get_data_idxs_for_gene(data, config):
    """
    return the data idxs of gene data
    :param data: raw data
    :param config:
    :return:
    """
    total_num = data.shape[0] - (config.TRAIN_LEN + config.EMBEDDING_LEN - 1) + 1
    return np.arange(total_num), None


def get_data_idxs_for_solar(data, batch, train_rate, config, batch_size=1e3, select_num=3e2):
    """
    getting the data idxs for solar dataset
    :param data: [m, K]
    :param batch: [0~]
    :param train_rate: 0-1
    :return: train_idxs, val_idxs
    """

    total_num = int(batch_size)
    select_num = int(select_num)

    assert total_num * (batch + 1) < data.shape[0], "batch_num is too much!"
    assert batch >= 0, "batch_num must > 0!"

    print(total_num * batch)
    print(total_num * (batch + 1))
    idxs = np.arange(total_num * batch, total_num * (batch + 1))
    # idxs = np.arange(select_num)

    np.random.seed(123)
    np.random.shuffle(idxs)

    idxs = idxs[:select_num]

    train_idxs = idxs[:int(train_rate * select_num)]
    val_idxs = idxs[int(train_rate * select_num):]

    print(train_idxs.dtype)

    return train_idxs, val_idxs


def get_data_idxs_for_wind_speed(data, batch, train_rate, config, batch_size=5e3, select_num=3e2):
    """
    get the training idxs and validation idxs for wind speed data
    :param data: [m, K]
    :param batch: [0~]
    :param train_rate: 0-1
    :return: train_idxs, val_idxs
    """
    # total_num = data.shape[0] - (config.TRAIN_LEN + config.EMBEDDING_LEN - 1) + 1

    total_num = int(batch_size)
    select_num = int(select_num)

    assert total_num*(batch + 1) < data.shape[0], "batch_num is too much!"
    assert batch >= 0, "batch_num must > 0!"

    print(total_num*batch)
    print(total_num*(batch+1))
    idxs = np.arange(total_num*batch, total_num*(batch+1))
    # idxs = np.arange(select_num)

    np.random.seed(123)
    np.random.shuffle(idxs)

    idxs = idxs[:select_num]

    train_idxs = idxs[:int(train_rate * select_num)]
    val_idxs = idxs[int(train_rate * select_num):]

    print(train_idxs.dtype)

    return train_idxs, val_idxs


def get_data_idxs_for_normal(data, train_rate, config, select_num=None, shuffle=True):
    """
    get the training idxs and validation idxs for normal data without specific functions
    :param data:
    :param train_rate:
    :param config:
    :param select_num: the size of data will be selected after shuffle, if None will select all data
    :param shuffle: whether to shuffle
    :return:
    """
    total_num = data.shape[0] - (config.TRAIN_LEN + config.EMBEDDING_LEN - 1) + 1
    idxs = np.arange(total_num)

    if shuffle:
        np.random.seed(123)
        np.random.shuffle(idxs)

    if select_num is not None:
        assert select_num < total_num, 'select num must smaller than total'
        idxs = idxs[:select_num]
        total_num = select_num

    if train_rate != 1:
        train_idxs = idxs[:int(train_rate * total_num)]
        val_idxs = idxs[int(train_rate * total_num): total_num]
    else:
        train_idxs = idxs
        val_idxs = None

    # print(train_idxs)
    return train_idxs, val_idxs


def get_data_idxs_for_traffic(data, train_rate, config, select_num=None, y_idxs=None):
    """
    first will drop the data with traffic speed of target sensor is 0.
    get the training idxs and validation idxs for traffic data
    :param data:
    :param train_rate:
    :param config:
    :param select_num:
    :param y_idxs: if is a list, drop the union set of data which traffic speed of target sensors (y_idxs) is 0.
    :return:
    """
    sample_len = config.TRAIN_LEN + config.EMBEDDING_LEN - 1  
    total_num = data.shape[0]  

    # the valid idxs of data that target variable that without 0.
    valid_idxs = np.ones(shape=(total_num,), dtype=np.float32)  

    if y_idxs is not None:  
        for y_idx in y_idxs:  # calculate the union set
            target_vars = data[:, y_idx]  

            for i, var in enumerate(target_vars):  
                if var == 0.0:
                    start = np.maximum(0, i - (sample_len - 1))
                    end = i
                    valid_idxs[start:end] = 0.0
    else:
        target_vars = data[:, config.Y_IDX]  

        for i, var in enumerate(target_vars):  
            if var == 0.0:
                start = np.maximum(0, i - (sample_len - 1))
                end = i
                valid_idxs[start:end] = 0.0

    total_num = total_num - sample_len + 1
    valid_idxs = valid_idxs[:total_num]

    valid_idxs = np.where(valid_idxs)[0]  
    total_num = valid_idxs.shape[0]  

    # used the first 10000 samples
    valid_idxs = valid_idxs[:10000]
    total_num = 10000

    np.random.seed(123)
    np.random.shuffle(valid_idxs)

    if select_num is not None:
        assert select_num < total_num, 'select num must smaller than total'
        valid_idxs = valid_idxs[:select_num]
        total_num = select_num

    train_idxs = valid_idxs[:int(train_rate * total_num)]
    val_idxs = valid_idxs[int(train_rate * total_num):]

    return train_idxs, val_idxs



def load_gene_data():
    """
    load the gene data
    :return: gene [23, 84]
    """
    gene = pd.read_csv(os.path.join(DATA_BASE_DIR, 'gene/circadian_geneexp_data.txt'), delimiter=',', header=None)
    # print(gene.shape)
    # print(gene[0])
    name = gene[0].map(lambda x: x.split('\t')[0]).values
    # print(name)
    gene = gene.iloc[:, 1:].values.T
    print(gene.shape)

    return name, gene


def load_lorenz_data(time_invariant=True, n=30, time=100):
    """
    load lorenz data
    :return: data [time_len, dim]
    """
    data = None
    # if time_invariant:
    file_name = 'lorenz/lorenz{}_d{}_t{}.pkl'.format('' if time_invariant else '_time_variant', n*3, int(time / 0.02))
    if not os.path.exists(os.path.join(DATA_BASE_DIR, file_name)):
        print(file_name)
        print('generating lorenz time {} data...'.format('invariant' if time_invariant else 'variant'))
        data = lorenz.my_lorenz(n, time=time, time_invariant=time_invariant)
        with open(os.path.join(DATA_BASE_DIR, file_name), 'wb') as file:
            pickle.dump(data, file)
    else:
        print('loading lorenz time {} data...'.format('invariant' if time_invariant else 'variant'))
        with open(os.path.join(DATA_BASE_DIR, file_name), 'rb') as file:
            data = pickle.load(file)
        # print(data.shape)
    data = data.T[2000:]
    return data


def load_hk_data(load_climate=False):
    """
    load the admission data of HK hospitals
    :param load_climate: whether to load the data about climate message
    :return:[time, 14+]
    """
    file = h5py.File(os.path.join(DATA_BASE_DIR, 'hk_data/hk_data_v1.mat'))
    datas = list()
    datas.append(np.array(file['data']))
    if load_climate:
        for k, v in file.items():
            if k == 'data':
                continue
            else:
                datas.append(np.array(v))
    data = np.concatenate(datas, axis=0).T
    print(data.shape)
    return data


def load_wind_speed_data(window_size=1):
    """
    load the wind speed dataset
    :param window_size: the window size
    :return:
    """
    data = sio.loadmat(os.path.join(DATA_BASE_DIR, 'wind_speed/201606232219windspeed.mat'))
    windspeed = data['windspeed']
    total_time_len, dim = windspeed.shape
    # print(total_time_len, ' ', dim)

    drop_time_number = total_time_len % window_size
    # print(drop_time_number)

    if drop_time_number != 0:
        windspeed = windspeed[: -drop_time_number]
    windspeed = windspeed.reshape((-1, window_size, dim))
    # print(windspeed.shape)
    windspeed = np.mean(windspeed, axis=1)  
    print(windspeed.shape)

    return windspeed


def load_traffic_data(loc='metr-la', window_size=None, fill_zero_with_mean=False):
    """
    load traffic dataset
    :param loc:
    :param window_size: 
    :param fill_zero_with_mean: whether to use mean value to fill the 0s.
    :return:
    """
    assert loc in ['metr-la', 'pems-bay'], 'error file name'
    data = pd.read_hdf(os.path.join(DATA_BASE_DIR, 'traffic/{}.h5'.format(loc)))

    excel_path = os.path.join(DATA_BASE_DIR, 'traffic/{}.csv'.format(loc))
    if not os.path.exists(excel_path):
        print(excel_path)
        data.to_csv(excel_path)

    if fill_zero_with_mean:
        data[data == 0.0] = np.nan  # first convert 0 to np.nan
        data = data.fillna(data.mean())

    data = data.values

    if window_size is not None:
        windowed_data = []
        for i in range(data.shape[0] - window_size + 1):
            windowed_data.append(np.mean(data[i:i + window_size], axis=0))
        data = np.stack(windowed_data)

    print(data.shape)
    return data


def load_typhoon_data(time_range=(99, 176), normalization=True):
    """
    load typhoon data
    :param time_range:
    :param normalization: whether to normalize the image data
    :return:
    """
    data = np.loadtxt(os.path.join(DATA_BASE_DIR, 'typhoon/typhoon.txt'),
                      dtype=np.float32, delimiter=' ')
    data = data[time_range[0]:time_range[1]]

    y_idxs = [0, 1]  # the latitude and longitude idxs
    for i in range(2, data.shape[-1]):
        if i % 2 == 0:
            y_idxs.append(i)
    data = data[:, y_idxs]

    if normalization:
        img_data = data[:, 2:]
        data[:, 2:] = (img_data - np.mean(img_data))/(np.max(img_data) - np.min(img_data))

    # print(np.mean(data, axis=0))
    print(data.shape)
    # print(data)
    return data


def window(data, size, stride):
    """
    apply a window to the raw data
    :param data: [t, dim]
    :param size:
    :param stride:
    :return:
    """

    t_len = data.shape[0]  # length of raw data
    win_len = (t_len - size) // stride + 1  # length of data after applying a window

    print(t_len)
    print(win_len)

    win_data = []
    for i in range(win_len):
        win_data.append(np.mean(data[i*stride:i*stride + size], axis=0))
    win_data = np.stack(win_data)
    print(win_data.shape)
    return win_data


def load_solar_irradiance(win_size=6, stride=1):
    """
    load data for solar irradiance dataset
    """
    data = sio.loadmat(os.path.join(DATA_BASE_DIR, 'solar_irradiance/201606241058solar_irradiance.mat'))
    data = data['solar_irradiance']

    data = window(data, size=win_size, stride=stride)

    print(data.shape)
    data[data < 0.0] = 0.0  # set all values smaller than 0 as 0.
    return data

