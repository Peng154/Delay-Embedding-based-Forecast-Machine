from forecast import st_delay_model
from data import data_processing
from forecast.lorenz_time_invariant import lorenz_time_invariant_config
from scipy.stats import pearsonr
import numpy as np
import pickle
import tensorflow as tf
import re
import os
from utils import utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_pic(known_y, label_y, predict_y, loss, pcc, x_label=None, y_label=None, y_lim=None,
             title=None, path=None, figsize=None):
    plt.rcParams['figure.figsize'] = figsize 
    plt.rcParams['savefig.dpi'] = 200

    fontsize = 17
    plt.title(title + ", PCC:{:.2f}".format(pcc), fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.xlabel(x_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.ylabel(y_label, fontdict={'family': 'Times New Roman', 'size': fontsize})
    plt.yticks(fontproperties='Times New Roman', size=fontsize)
    plt.xticks(fontproperties='Times New Roman', size=fontsize)

    if y_lim is not None:
        plt.ylim(*y_lim)

    train_len = len(known_y)
    all_y = np.concatenate([known_y, label_y])
    x = np.arange(len(all_y))
    plt.plot(x, all_y, color='blue', marker='.')

    x = np.arange(train_len, len(all_y))

    if title == 'lorenz':
        plt.scatter(x, predict_y, color='none', edgecolors='red', marker='o',
                    label='trained_loss:{:.2f},pcc:{:.2f}'.format(loss, pcc),
                    zorder=10, linewidths=1.2, s=40)
    else:
        plt.plot(x, predict_y, color='red', marker='.', label='trained_loss:{:.2f},pcc:{:.2f}'.format(loss, pcc))

        connected_y = np.stack([known_y[-1], predict_y[0]]) 
        x = np.arange(train_len - 1, train_len + 1)
        plt.plot(x, connected_y, color='red')

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2, mode="expand", borderaxespad=0.)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()


def get_model_path(base_log_dir, config):
    """
    get the model saved path by config
    :param config:
    :return:
    """
    name_pattern = re.compile('.*\({}\)'.format(config.name))
    # name_pattern = re.compile('.*noise_{:.1f}\)'.format(config.DATA_NOISE_STRENGTH))
    # name_pattern = re.compile('.*\(.*rate_{:.2f}\)'.format(config.DATASET_RATE))
    # name_pattern = re.compile('.*\(lorenz_{}.*\)'.format(config.TRAIN_LEN))

    file_pattern = re.compile('weights_epoch:{:0>4d}.*'.format(config.EPOCHS))
    model_path = None

    for d in os.listdir(base_log_dir):
        print(d)
        if name_pattern.match(d):
            for f in os.listdir(os.path.join(base_log_dir, d)):
                if file_pattern.match(f):
                    model_path = os.path.join(base_log_dir, d, f)
                    print('load weights from: {}'.format(model_path))
                    return model_path

    return model_path


if __name__ == '__main__':
    is_solar = False
    config = lorenz_time_invariant_config.LorenzTimeInvariantConfig()
    config.BATCH_SIZE = 1  # set the batch size as 1

    tf.keras.backend.clear_session()  # clear the previous session

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # load the data
    time_invariant = True
    data = data_processing.load_lorenz_data(time_invariant=time_invariant)  # lorenz data

    train_idxs, val_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(rate=0.5) 


    mean_train_losses = []
    mean_train_pccs = []

    results = {}

    model = st_delay_model.STDelayModel(config, mode='evaluation', log_dir_suffix=config.name)

    noised_data, y = data_processing.add_noise(data, config)  # 根据config添加噪声

    # get the data generator
    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)

    # model_path = get_model_path(base_log_dir='./logs/lorenz_len_cmp_rate_0.5', config=config)  
    # restore the model form specific path
    model_path = '../../logs/lorenz_len_cmp_rate_0.5/2020_04_09-14_01_03(lorenz_40_19_Yidx_0)/' \
                 'weights_epoch:0085_loss:0.029_val_loss:0.314_predict_loss:0.499.h5'
    print(model_path)
    model.load_weights(model_path)

    sets = {'train': train_generator, 'val': val_generator}

    for set in sets:
        if sets[set] is None:
            continue
        print(set)
        generator = sets[set]
        idx = 0
        result_dir = '../../logs/results/{}/{}'.format(config.name, set)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        losses = []
        pccs = []

        predict_ys = []
        label_ys = []
        known_ys = []

        for item in generator.get_item():
            input_x, label_y, label_matrix = item[0][0], item[-1], item[1][0]
            predict_y_matrix = model.model.predict(input_x)[0]

            # print(model.model.predict(input_x)[-1])
            # print(label_y)
            # print(predict_y_matrix)

            predict_y = utils.get_y_from_matrix(config, predict_y_matrix.T, weighted=False)

            known_y = input_x[0, :, config.Y_IDX]  
            label_y = item[-1][0]  

            if is_solar:  
                predict_y[predict_y < 0] = 0
                label_y[label_y < 0] = 0
                known_y[known_y < 0] = 0

            loss = np.sqrt(np.mean(np.square(label_y - predict_y)))
            known_ys.append(known_y)
            predict_ys.append(predict_y)
            label_ys.append(label_y)
            losses.append(loss)


            pcc, p_value = pearsonr(label_y, predict_y)
            pccs.append(pcc)

            draw_pic(known_y, label_y, predict_y, loss, pcc=pcc, path=result_dir + '/{}.png'.format(idx),
                     figsize=(8, 6), y_lim=None, x_label='Time', y_label='Value', title='lorenz')  # lorenz

            idx += 1

            if idx % 100 == 0:
                print(idx)

        print(np.sum(np.array(losses) < 1.0))
        print('mean loss：', np.mean(losses))
        print('mean pcc:', np.mean(pccs))
        print(np.argsort(losses)[:100])
        print(np.argsort(pccs)[::-1][:100])

        if set == 'train':
            prediction_results = {}
            prediction_results['train_ys'] = known_ys
            prediction_results['label_ys'] = label_ys
            prediction_results['predict_ys'] = predict_ys
            prediction_results['rmses'] = losses
            prediction_results['pccs'] = pccs
            with open('../../logs/results/{}_prediction_results.pkl'.format(config.name), 'wb') as file:
                pickle.dump(prediction_results, file)

