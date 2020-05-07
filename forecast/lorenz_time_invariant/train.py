from forecast import st_delay_model
from data import data_processing
from forecast.lorenz_time_invariant import lorenz_time_invariant_config
import tensorflow as tf

if __name__ == '__main__':

    config = lorenz_time_invariant_config.LorenzTimeInvariantConfig()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # load data
    time_invariant = True
    data = data_processing.load_lorenz_data(time_invariant=time_invariant, n=30, time=3000) 

    train_idxs, val_idxs = data_processing.get_data_idxs_as_predict_idx_no_overlap(rate=0.5)

    print(len(train_idxs))

    data, y = data_processing.add_noise(data, config)  # add noise

    train_generator = st_delay_model.DataGeneratorForLengthCmp(data, y, train_idxs, config)
    val_generator = None if val_idxs is None else st_delay_model.DataGeneratorForLengthCmp(data, y, val_idxs, config)

    model = st_delay_model.STDelayModel(config, mode='training',
                                        log_dir_suffix=config.name)
    model.compile()

    # load weights and continue the training
    # model_path = './logs/2020_03_29-13_44_03(ws_200_101_Yidx_0)/' \
    #              'weights_epoch:0150_loss:0.245_val_loss:0.259_predict_loss:0.623.h5'
    # model.load_weights(model_path)

    model.train(train_generator, val_generator)
