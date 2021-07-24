# i = 0
# Q = tf.expand_dims(query, 1) @ Wq[0]
# values_T = tf.transpose(values_in, [0, 2, 1])
#
# K = values_T @ Wk[i]

B = 64
N = 128
F = 32
Et = 31

import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("deneme", reuse=False):
    values = tf.get_variable(shape=[B, F, N], name='value')
    values_T = tf.transpose(values, [0, 2, 1])
    query = tf.get_variable(shape=[B, Et], name='query')
    query = tf.expand_dims(query, axis=1)
    Wv = tf.get_variable(shape=[F, F], name='Wv')
    Wk = tf.get_variable(shape=[F, F], name='Wk')
    Wq1 = tf.get_variable(shape=[Et, N], name='Wq1')
    Wq2 = tf.get_variable(shape=[1, F], name='Wq2')

    Q = tf.transpose(query @ Wq1, [0, 2, 1]) @ Wq2
    K = values_T @ Wk
    #
    # Q = query @ Wq
    V = values_T @ Wv
    # print('Q.shape=',Q.shape)
    # print('V.shape=',V.shape)
    # print('K.shape=',K.shape)
V
tf.transpose(tf.reduce_sum(tf.nn.softmax((Q @ tf.transpose(K, [0,2,1])) / F**0.5, axis=2) @ V, axis=2, keepdims=True), [0,2,1])

tf.transpose(tf.reduce_sum((tf.transpose(query @ Wq1, [0, 2, 1]) @ Wq2) @ V, axis=2, keepdims=True), [0,2,1])
query @ Wq1
V
Q
values
K

tf.reduce_sum()


Q
K


Q @ tf.transpose(K, [0,2,1])
