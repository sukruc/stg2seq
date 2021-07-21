i = 0
Q = tf.expand_dims(query, 1) @ Wq[0]
values_T = tf.transpose(values_in, [0, 2, 1])

K = values_T @ Wk[i]


import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("deneme", reuse=False):
    values = tf.get_variable(shape=[64, 32, 128], name='value')
    values_T = tf.transpose(values, [0, 2, 1])
    query = tf.get_variable(shape=[64, 31], name='query')
    query = tf.expand_dims(query, axis=1)
    Wv = tf.get_variable(shape=[128, 32], name='Wv')
    Wk = tf.get_variable(shape=[32, 32], name='Wk')
    Wq = tf.get_variable(shape=[31, 32], name='Wq')

    Q = query @ Wq
    V = values @ Wv
    K = values_T @ Wk
    print('Q.shape=',Q.shape)
    print('V.shape=',V.shape)
    print('K.shape=',K.shape)
    
