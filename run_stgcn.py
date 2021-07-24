
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

#config
DATASET = 'NYC'
DEVICE = '0'
SAVE_SUMMARY = False
PLOT_DETAIL  = False

from params_stg2seq import params_NYC as params

bike_data = h5py.File(os.path.join(base_dir, 'data/BikeNYC/NYC14_M16x8_T60_NewEnd.h5'), 'r')
bike_data = bike_data['data'].value
# reshape the data format to [sample_nums, region_nums, dims], 4392 = 24*183
bike_data = np.transpose(bike_data, (0, 2, 3, 1))
bike_data = np.reshape(bike_data, (4392, -1, 2))
adj = generate_graph_with_data(bike_data, params.test_days * 24, threshold=params.threshold)
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
with tf.name_scope('Train'):
    with tf.variable_scope('model', reuse=False):
        model_train = Graph(adj_mx=adj, params=params, is_training=False)
    with tf.variable_scope('model1', reuse=False):
        model_train_1 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=1)
    with tf.variable_scope('model2', reuse=False):
        model_train_2 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=2)
    with tf.variable_scope('model3', reuse=False):
        model_train_3 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=3)
    with tf.variable_scope('model4', reuse=False):
        model_train_4 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=4)
    with tf.variable_scope('model5', reuse=False):
        model_train_5 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=5)
with tf.name_scope('Test'):
    with tf.variable_scope('model', reuse=True):
        model_test = Graph(adj_mx=adj, params=params, is_training=False)
    with tf.variable_scope('model1', reuse=True):
        model_test_1 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=1)
    with tf.variable_scope('model2', reuse=True):
        model_test_2 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=2)
    with tf.variable_scope('model3', reuse=True):
        model_test_3 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=3)
    with tf.variable_scope('model4', reuse=True):
        model_test_4 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=4)
    with tf.variable_scope('model5', reuse=True):
        model_test_5 = Graph(adj_mx=adj, params=params, is_training=False, num_blocks_to_use=5)
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logdir=params.model_path+"/current")

    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    best_rmse = 10000
    best_rmse1 = 10000
    best_rmse2 = 10000
    best_rmse3 = 10000
    best_rmse4 = 10000
    best_rmse5 = 10000
    try:
        for epoch in range(params.num_epochs):
            loss_train = 0
            loss_train_1 = 0
            loss_train_2 = 0
            loss_train_3 = 0
            loss_train_4 = 0
            loss_train_5 = 0
            loss_val = 0
            rmse_val = 0
            rmse_val1 = 0
            rmse_val2 = 0
            rmse_val3 = 0
            rmse_val4 = 0
            rmse_val5 = 0
            print("Epoch: {}\t".format(epoch))
            # training
            num_batches = (X_train[0].shape[0] // params.batch_size) + 1
            # num_batches = 10  # TODO: Revert this back
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
                loss_tr1, _ = sess.run([model_train_1.loss, model_train_1.optimizer], feed_dict={
                    model_train_1.c_inp: x_closeness,
                    model_train_1.et_inp: y_time,
                    model_train_1.labels: y_demand})
                loss_tr2, _ = sess.run([model_train_2.loss, model_train_2.optimizer], feed_dict={
                    model_train_2.c_inp: x_closeness,
                    model_train_2.et_inp: y_time,
                    model_train_2.labels: y_demand})
                loss_tr3, _ = sess.run([model_train_3.loss, model_train_3.optimizer], feed_dict={
                    model_train_3.c_inp: x_closeness,
                    model_train_3.et_inp: y_time,
                    model_train_3.labels: y_demand})
                loss_tr4, _ = sess.run([model_train_4.loss, model_train_4.optimizer], feed_dict={
                    model_train_4.c_inp: x_closeness,
                    model_train_4.et_inp: y_time,
                    model_train_4.labels: y_demand})
                loss_tr5, _ = sess.run([model_train_5.loss, model_train_5.optimizer], feed_dict={
                    model_train_5.c_inp: x_closeness,
                    model_train_5.et_inp: y_time,
                    model_train_5.labels: y_demand})
                loss_train = loss_tr  + loss_train
                loss_train_1 = loss_tr1 + loss_train_1
                loss_train_2 = loss_tr2 + loss_train_2
                loss_train_3 = loss_tr3 + loss_train_3
                loss_train_4 = loss_tr4 + loss_train_4
                loss_train_5 = loss_tr5 + loss_train_5
            # testing
            x_closeness = X_test[0]
            x_external = X_test[1]
            y_demand = Y_test[0]
            y_time = Y_test[1][:, :, 0: params.et_dim]
            loss_v, rmse_val = sess.run([model_test.loss, model_test.rmse],
                                        feed_dict={model_test.c_inp: x_closeness,
                                                   model_test.et_inp:y_time,
                                                   model_test.labels: y_demand})
            loss_v1, rmse_val1 = sess.run([model_test_1.loss, model_test_1.rmse],
                                        feed_dict={model_test_1.c_inp: x_closeness,
                                                   model_test_1.et_inp:y_time,
                                                   model_test_1.labels: y_demand})
            loss_v2, rmse_val2 = sess.run([model_test_2.loss, model_test_2.rmse],
                                        feed_dict={model_test_2.c_inp: x_closeness,
                                                   model_test_2.et_inp:y_time,
                                                   model_test_2.labels: y_demand})
            loss_v3, rmse_val3 = sess.run([model_test_3.loss, model_test_3.rmse],
                                        feed_dict={model_test_3.c_inp: x_closeness,
                                                   model_test_3.et_inp:y_time,
                                                   model_test_3.labels: y_demand})
            loss_v4, rmse_val4 = sess.run([model_test_4.loss, model_test_4.rmse],
                                        feed_dict={model_test_4.c_inp: x_closeness,
                                                   model_test_4.et_inp:y_time,
                                                   model_test_4.labels: y_demand})
            loss_v5, rmse_val5 = sess.run([model_test_5.loss, model_test_5.rmse],
                                        feed_dict={model_test_5.c_inp: x_closeness,
                                                   model_test_5.et_inp:y_time,
                                                   model_test_5.labels: y_demand})

            rmse_val = rmse_val[0]
            rmse_val1 = rmse_val1[0]
            rmse_val2 = rmse_val2[0]
            rmse_val3 = rmse_val3[0]
            rmse_val4 = rmse_val4[0]
            rmse_val5 = rmse_val5[0]
            print("1 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train_1, rmse_val1))
            print("2 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train_2, rmse_val2))
            print("3 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train_3, rmse_val3))
            print("4 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train_4, rmse_val4))
            print("5 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train_5, rmse_val5))
            print("6 block loss: {:.6f}, rmse_val: {:.3f}".format(loss_train, rmse_val))

            # save the model every epoch
            # saver.save(sess, params.model_path+"/current")
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                rmse, mae, mape = sess.run([model_test.rmse, model_test.mae, model_test.mape],
                                                 feed_dict={model_test.c_inp: x_closeness,
                                                            model_test.et_inp: y_time,
                                                            model_test.labels: y_demand})
            if rmse_val1 < best_rmse1:
                best_rmse1 = rmse_val1
                rmse1, mae1, mape1 = sess.run([model_test_1.rmse, model_test_1.mae, model_test_1.mape],
                                           feed_dict={model_test_1.c_inp: x_closeness,
                                                      model_test_1.et_inp: y_time,
                                                      model_test_1.labels: y_demand})
            if rmse_val2 < best_rmse2:
                best_rmse2 = rmse_val2
                rmse2, mae2, mape2 = sess.run([model_test_2.rmse, model_test_2.mae, model_test_2.mape],
                                           feed_dict={model_test_2.c_inp: x_closeness,
                                                      model_test_2.et_inp: y_time,
                                                      model_test_2.labels: y_demand})
            if rmse_val3 < best_rmse3:
                best_rmse3 = rmse_val3
                rmse3, mae3, mape3 = sess.run([model_test_3.rmse, model_test_3.mae, model_test_3.mape],
                                           feed_dict={model_test_3.c_inp: x_closeness,
                                                      model_test_3.et_inp: y_time,
                                                      model_test_3.labels: y_demand})
            if rmse_val4 < best_rmse4:
                best_rmse4 = rmse_val4
                rmse4, mae4, mape4 = sess.run([model_test_4.rmse, model_test_4.mae, model_test_4.mape],
                                           feed_dict={model_test_4.c_inp: x_closeness,
                                                      model_test_4.et_inp: y_time,
                                                      model_test_4.labels: y_demand})
            if rmse_val5 < best_rmse5:
                best_rmse5 = rmse_val5
                rmse5, mae5, mape5 = sess.run([model_test_5.rmse, model_test_5.mae, model_test_5.mape],
                                           feed_dict={model_test_5.c_inp: x_closeness,
                                                      model_test_5.et_inp: y_time,
                                                      model_test_5.labels: y_demand})
                #saver.save(sess, './final_model.ckpt')
    except Exception as e:
        raise e
    finally:
        print("1 block Finish Learning! Best RMSE is", rmse1, "Best MAE is", mae1, 'MAPE: ', mape1)
        print("2 blocks Finish Learning! Best RMSE is", rmse2, "Best MAE is", mae2, 'MAPE: ', mape2)
        print("3 blocks Finish Learning! Best RMSE is", rmse3, "Best MAE is", mae3, 'MAPE: ', mape3)
        print("4 blocks Finish Learning! Best RMSE is", rmse4, "Best MAE is", mae4, 'MAPE: ', mape4)
        print("5 blocks Finish Learning! Best RMSE is", rmse5, "Best MAE is", mae5, 'MAPE: ', mape5)
        print("6 blocks Finish Learning! Best RMSE is", rmse, "Best MAE is", mae, 'MAPE: ', mape)
        sess.close()
