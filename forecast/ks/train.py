from forecast import st_delay_model
from data import data_processing
from forecast.ks import ks_config
import tensorflow as tf

if __name__ == '__main__':

    config = ks_config.KSConfig()


    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    data = data_processing.load_ks_data()

    train_idxs, val_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(rate=1,
                                                                                   interval=60,
                                                                                   total_count=1300,
                                                                                   train_count=1200) 


    print(len(train_idxs))

    data, y = data_processing.add_noise(data, config) 

    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)

    model = st_delay_model.STDelayModel(config, mode='training',
                                        log_dir_suffix=config.name)
    model.compile()

    model.train(train_generator, val_generator)
