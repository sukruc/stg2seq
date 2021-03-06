
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import h5py
from lib.load_dataset import load_dataset_multi_demand_external
from lib.batch_generator import batch_generator_multi_Y
from lib.generate_adj import generate_graph_with_data
# from RootPATH import base_dir
base_dir = os.path.dirname(os.path.realpath(__file__))
#config dataset and model

from G2S_multistep import Graph

import pandas as pd

#config
DATASET = 'NYC'
DEVICE = '0'
SAVE_SUMMARY = False
PLOT_DETAIL  = False

# if DATASET == 'SY':
#     from params_stg2seq import params_SY as params
#     didi_data = np.loadtxt(os.path.join(base_dir, 'data/Didi_SY/didi_order_hour_grid.csv'),
#                            delimiter=',', skiprows=1)
#     adj = generate_graph_with_data(didi_data, 144, threshold=params.threshold)
# elif DATASET == 'SY_IR':
#     from params_stgcn import params_SY_IR as params
#     didi_data = np.loadtxt(os.path.join(base_dir, 'data/Didi_SY/didi_order_hour.csv'),
#                            delimiter=',', skiprows=1)
#     adj = generate_graph_with_data(didi_data, 144, threshold=params.threshold)
# elif DATASET == 'NYC':
from params_stg2seq_hyperparameter_tuning import params_NYC as params
bike_data = h5py.File(os.path.join(base_dir, 'data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'), 'r')
bike_data = bike_data['data'].value
# reshape the data format to [sample_nums, region_nums, dims], 4392 = 24*183
bike_data = np.transpose(bike_data, (0, 2, 3, 1))
bike_data = np.reshape(bike_data, (4392, -1, 2))
adj = generate_graph_with_data(bike_data, params.test_days * 24, threshold=params.threshold)    
# elif DATASET == 'BJ':
#     from data.TaxiBJ.load_dataset_BJ import remove_incomplete_days, load_stdata
#     from params_stg2seq import params_BJ as params
#     path = os.path.join(base_dir, 'data/TaxiBJ/BJ15_M32x32_T30_InOut.h5')
#     # BJ dataset has some incomplete days
#     taxi_data, timestamps = load_stdata(path)
#     taxi_data, timestamps = remove_incomplete_days(taxi_data, timestamps, 48)
#     taxi_data = np.transpose(taxi_data, (0, 2, 3, 1))
#     taxi_data = np.reshape(taxi_data, (5520, -1, 2))
#     adj = generate_graph_with_data(taxi_data, params.test_days * 48, threshold=params.threshold)
# else:
#     exit(0)

# for lear_rate in params.lr:

batchs = params.batch_size
horzon = params.horizon
bt1 = params.beta1
bt2 = params.beta2
learn = params.lr
epsln = params.epsilon
nunt = params.num_units


outdf = pd.DataFrame()

for bs in batchs:
    for horz in horzon:
        for b1 in bt1:
            for lear in learn:
                    # for b2 in bt2:
            # for eps in epsln:
                        # for nu in nunt:

                params.batch_size = bs
                params.horizon = horz
                params.lr = lear
                params.beta1 = b1 
                # params.beta2 = b2 
                # params.epsilon = eps 
                # params.num_units = nu 

                print('=====================')
                print(params.batch_size)
                print(params.horizon)
                print(params.beta1)
                print(params.lr)
                # print(params.beta2)
                # print(params.epsilon)
                # print(params.num_units)
                print('=====================')

                X_train, Y_train, X_test, Y_test, scaler = load_dataset_multi_demand_external(
                                                                                base_dir,
                                                                                params.source, params.nb_flow,
                                                                                params.map_height, params.map_width,
                                                                                len_closeness=params.closeness_sequence_length,
                                                                                len_period=params.period_sequence_length,
                                                                                len_trend=params.trend_sequence_length,
                                                                                test_days=params.test_days,
                                                                                horizon=params.horizon,
                                                                                external_length=params.external_length)
                train_batch_generator = batch_generator_multi_Y(X_train, Y_train, params.batch_size)

                #with teacher forcing: true, false
                #without: false, false
                tf.reset_default_graph()
                with tf.name_scope('Train'):
                    with tf.variable_scope('model', reuse=False):
                        model_train = Graph(adj_mx=adj, params=params, is_training=False)
                with tf.name_scope('Test'):
                    with tf.variable_scope('model', reuse=True):
                        model_test = Graph(adj_mx=adj, params=params, is_training=False)
                for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                        print(var)

                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.Session(config=config) as sess:
                    saver = tf.train.Saver()
                    writer = tf.summary.FileWriter(logdir=params.model_path+"/current")

                    sess.run(tf.global_variables_initializer())
                    writer.add_graph(sess.graph)
                    best_rmse = 10000
                    try:
                        for epoch in range(params.num_epochs):
                            loss_train = 0
                            loss_val = 0
                            rmse_val = 0
                            print("Epoch: {}\t".format(epoch))
                            # training
                            num_batches = (X_train[0].shape[0] // bs) + 1
                            # num_batches = 10
                            '''
                            #for dynamic learning rate
                            if epoch%params.lr_decay_epoch == 0:
                                new_lr = params.lr * (0.1 ** (epoch/params.lr_decay_epoch))
                                if new_lr > params.min_lr:
                                    model_train.set_lr(sess, new_lr)
                                    print(sess.run(model_train.lr))
                            '''
                            for b in tqdm(range(num_batches)):
                                x_batch, y_batch = next(train_batch_generator)
                                # print(x_batch.shape, y_batch.shape)
                                x_closeness = x_batch[0]
                                x_external = x_batch[1]
                                y_demand = y_batch[0]
                                y_time = y_batch[1][:, :, 0: params.et_dim]
                                loss_tr, _ = sess.run([model_train.loss, model_train.optimizer], feed_dict={
                                    model_train.c_inp: x_closeness,
                                    model_train.et_inp:y_time,
                                    model_train.labels: y_demand})
                                loss_train = loss_tr  + loss_train
                            # testing
                            x_closeness = X_test[0]
                            x_external = X_test[1]
                            y_demand = Y_test[0]
                            y_time = Y_test[1][:, :, 0: params.et_dim]
                            loss_v, rmse_val = sess.run([model_test.loss, model_test.rmse],
                                                        feed_dict={model_test.c_inp: x_closeness,
                                                                model_test.et_inp:y_time,
                                                                model_test.labels: y_demand})

                            rmse_val = rmse_val[0]
                            print("loss: {:.6f}, rmse_val: {:.3f}".format(loss_train, rmse_val))
                            # save the model every epoch
                            saver.save(sess, params.model_path+"/current")
                            if rmse_val < best_rmse:
                                best_rmse = rmse_val
                                rmse, mae, mape = sess.run([model_test.rmse, model_test.mae, model_test.mape],
                                                                feed_dict={model_test.c_inp: x_closeness,
                                                                            model_test.et_inp: y_time,
                                                                            model_test.labels: y_demand})
                                #saver.save(sess, './final_model.ckpt')
                    except Exception as e:
                        # raise e
                        continue
                    finally:
                        print("Finish Learning! Best RMSE is", rmse, "Best MAE is", mae, 'MAPE: ', mape)
                        sess.close()

                    sss = pd.DataFrame({
                                'source': params.source,
                                "batch_size" :params.batch_size,
                                "horizon" : params.horizon,
                                "lr" : params.lr,
                                "num_epochs" : params.num_epochs,
                                "epsilon" : params.epsilon,
                                "beta1" : params.beta1,
                                "beta2" : params.beta2,
                                "best_rmse" : rmse,
                                "best_mae":mae,
                                "best_mape":mape,

                    })
                    outdf = outdf.append(sss)
                outdf.to_csv(f"outdf_eps_{bs}_{b1}_{lear}_{horz}.csv", index=False)
print('##########################')
print('##########################')
outdf.to_csv("outdf_eps.csv", index=False)