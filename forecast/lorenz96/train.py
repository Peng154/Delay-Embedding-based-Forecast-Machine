from forecast import st_delay_model
from data import data_processing
from forecast.lorenz96 import lorenz96_config
import tensorflow as tf

if __name__ == '__main__':

    config = lorenz96_config.Lorenz96Config()


    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    data = data_processing.load_lorenz96_data(N=config.K, F=config.F, time_range=(0, 3000), dt=0.02)

    train_idxs, val_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(rate=1,
                                                                                   interval=60,
                                                                                   total_count=2099,
                                                                                   train_count=2000)

    print(len(train_idxs))

    data, y = data_processing.add_noise(data, config)

    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)

    model = st_delay_model.STDelayModel(config, mode='training',
                                        log_dir_suffix=config.name)
    model.compile()

    model.train(train_generator, val_generator)
