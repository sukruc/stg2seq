from gcn_layer import *
from gcn_layer import graph_conv
import tensorflow as tf
from lib.metrics import MAE, RMSE, MAPE, MARE, R2
import numpy as np


def Conv_ST(inputs, supports, kt, dim_in, dim_out, activation):
    '''
    :param inputs: a tensor of shape [B, T, N, C]
    :param supports:
    :param kt: temporal convolution length
    :param dim_in:
    :param dim_out:
    :return:
    '''
    T = inputs.get_shape().as_list()[1]
    num_nodes = inputs.get_shape().as_list()[2]
    assert inputs.get_shape().as_list()[3] == dim_in
    if (dim_in > dim_out):
        w_input = tf.get_variable(
            'wt_input', shape=[1, 1, dim_in, dim_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
        res_input = tf.nn.conv2d(inputs, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif (dim_in < dim_out):
        res_input = tf.concat(
            [inputs, tf.zeros([tf.shape(inputs)[0], T, num_nodes, dim_out - dim_in])], axis=3)
    else:
        res_input = inputs
    # padding zero
    padding = tf.zeros([tf.shape(inputs)[0], kt - 1, num_nodes, dim_in])
    # extract spatial-temporal relationships at the same time
    inputs = tf.concat([padding, inputs], axis=1)
    x_input = tf.stack([inputs[:, i:i + kt, :, :] for i in range(0, T)], axis=1)    #[B*T, kt, N, C]
    x_input = tf.reshape(x_input, [-1, kt, num_nodes, dim_in])
    x_input = tf.transpose(x_input, [0, 2, 1, 3])

    if (activation == 'GLU'):
        conv_out = graph_conv(tf.reshape(x_input, [-1, num_nodes, kt * dim_in]),
                              supports, kt * dim_in, 2 * dim_out)
        conv_out = tf.reshape(conv_out, [-1, T, num_nodes, 2 * dim_out])
        # EQUATION: 5 - Gated Linear Unit
        out = (conv_out[:, :, :, 0:dim_out] + res_input) * \
              tf.nn.sigmoid(conv_out[:, :, :, dim_out:2 * dim_out])
    if (activation == 'sigmoid'):
        # EQUATION: 4 - Non-gated version
        # NOTE: GCN: This condition has nothing to do with sigmoid
        conv_out = graph_conv(tf.reshape(x_input, [-1, num_nodes, kt * dim_in]),
                              supports, kt * dim_in, dim_out)
        out = tf.reshape(conv_out, [-1, T, num_nodes, dim_out])
    # out = tf.nn.relu(conv_out + res_input)
    return out


def LN(y0, scope):
    # batch norm
    size_list = y0.get_shape().as_list()
    T, N, C = size_list[1], size_list[2], size_list[3]
    mu, sigma = tf.nn.moments(y0, axes=[1, 2, 3], keep_dims=True)
    with tf.variable_scope(scope):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, T, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, T, N, C]))
        y0 = (y0 - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return y0


def attention_t_mul(query, values, scope, num_attention_heads=4):
    '''
    :param query: a tensor shaped [B, Et]
    :param values: a tensor shaped [B, T, H*W, F]
    :return:
    [B, 1, N, F]

    Notes:
    -----------------
    B: Batch size
    Et: External parameters (target hour, etc)
    '''
    Et = query.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    T = values.get_shape().as_list()[1]
    # values_in = tf.reshape(values, [-1, N, F])
    # values = tf.transpose(values_in, [2, 0, 1]) #[F,B,N]
    # import pdb; pdb.set_trace()
    query = tf.expand_dims(query, axis=1)
    # values = tf.reshape(values, [-1, T, N * F])
    values_T = tf.transpose(values, [0, 3, 2, 1])
    # values_T = tf.transpose(tf.squeeze(values, axis=1), [0, 2, 1])
    # values = tf.squeeze(values, axis=1)
    # values = tf.squeeze(values, 1)
    with tf.variable_scope(scope):

        Wk = [tf.get_variable(f'Wk{i}', shape=[T, T], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wv = [tf.get_variable(f'Wv{i}',shape=[T, T],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wq1 = [tf.get_variable(f'Wq1{i}',shape=[Et, N],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wq2 = [tf.get_variable(f'Wq2{i}',shape=[1, T],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]

        bias_v = [tf.get_variable(f'bias_v{i}', initializer=tf.zeros([T])) for i in range(num_attention_heads)]
        bias_k = [tf.get_variable(f'bias_k{i}', initializer=tf.zeros([T])) for i in range(num_attention_heads)]
        bias_q = [tf.get_variable(f'bias_q{i}', initializer=tf.zeros([T])) for i in range(num_attention_heads)]

        # Wv = tf.get_variable(shape=[N, F], name='Wv')
        # Wk = tf.get_variable(shape=[F, F], name='Wk')
        # Wq1 = tf.get_variable(shape=[Et, N], name='Wq1')
        # Wq2 = tf.get_variable(shape=[1, F], name='Wq2')

        mh_attention_weights = tf.get_variable("mh_head", shape=[T * num_attention_heads, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # Wk = tf.get_variable('Wk', shape=[Et, F], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # Wv = tf.get_variable('Wv',shape=[F, N, 1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        # Wq = tf.get_variable('Wq',shape=[Et, F],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

        # bias_v = tf.get_variable('bias_v', initializer=tf.zeros([F, 1, 1]))
        # bias_k = tf.get_variable('bias_k', initializer=tf.zeros([F]))
        # bias_q = tf.get_variable('bias_q', initializer=tf.zeros([F]))
    # i = 0
    # import pdb; pdb.set_trace()
    Q = [tf.expand_dims(tf.transpose(query @ Wq1[i], [0, 2, 1]), axis=1) @ Wq2[i] + bias_q[i] for i in range(num_attention_heads)]
    K = [values_T @ Wk[i] + bias_k[i] for i in range(num_attention_heads)]
    V = [values_T @ Wv[i] + bias_v[i] for i in range(num_attention_heads)]

    # Q = [tf.matmul(query, Wq[i]) + bias_q[i] for i in range(num_attention_heads)]
    # K = [tf.matmul(values, Wk[i]) + bias_k[i] for i in range(num_attention_heads)]
    # V = [tf.matmul(values, Wv[i]) + bias_v[i] for i in range(num_attention_heads)]
    att = [tf.nn.softmax((Q[i] @ tf.transpose(K[i], [0,1,3,2])) / T**0.5, axis=2) @ V[i] for i in range(num_attention_heads)]

    # att = tf.transpose(tf.reduce_sum((tf.transpose(query @ Wq1, [0, 2, 1]) @ Wq2) @ V, axis=2, keepdims=True), [0,2,1])

    # att = [tf.nn.softmax((Q[i] @ tf.transpose(K[i])) / (F)**0.5, axis=1) @ V[i] for i in range(num_attention_heads)]
    # att = [tf.transpose(att[i], [1, 0, 2]) for i in range(num_attention_heads)]
    # out = [tf.matmul(values_in, att[i]) for i in range(num_attention_heads)]
    # import pdb; pdb.set_trace()
    out = tf.concat(att, axis=3, name='mh_att_concat')
    # import pdb; pdb.set_trace()
    # out = tf.squeeze(out, axis=2)
    # out = tf.transpose(out, [0, 2, 1])
    out = out @ mh_attention_weights
    out = tf.transpose(out, [0, 3, 2, 1])
    # import pdb; pdb.set_trace()
    # out = tf.reshape(out, [-1, 1, N, F])
    # out = tf.expand_dims(out, axis=-1)
    # import pdb; pdb.set_trace()

    return out


def attention_t_add(query, values, scope, num_attention_heads=None):
    '''
        :param query: a tensor shaped [B, Et]
        :param values: a tensor shaped [B, T, H*W, F]
        :return:
    '''
    Et = query.get_shape().as_list()[1]
    T = values.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    values_in = tf.reshape(values, [-1, T, N*F])  #[B, T, N*F]
    values_in = tf.transpose(values_in, [0, 2, 1]) #[B, N*F, T]
    values = tf.transpose(values_in, [2, 0, 1])  # [T,B,N*F]
    with tf.variable_scope(scope):
        Wv = tf.get_variable('Wv', shape=[T, N*F,1], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        bias_v = tf.get_variable('bias_v', initializer=tf.zeros([T]))
        Wq = tf.get_variable('Wq', shape=[Et, T], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    value_linear = tf.reshape(tf.transpose(tf.matmul(values, Wv), [1,0,2]), [-1, T])
    # score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    # EQUATION: 6 - Temporal Attention
    score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    score = tf.nn.softmax(score, dim=1)  # shape is [B,T]
    # EQUATION: 7 - Transform joint representation by importance
    values = tf.matmul(values_in, tf.expand_dims(score, axis=-1))  # [B,N*F,1]
    values = tf.reshape(tf.transpose(values, [0, 2, 1]), [-1, 1, N, F])
    return values


def attention_c_mul(query, values, scope, num_attention_heads=4):
    '''
    :param query: a tensor shaped [B, Et]
    :param values: a tensor shaped [B, 1, H*W, F]
    :return:
    [B, N, 1]

    Notes:
    -----------------
    B: Batch size
    Et: External parameters (target hour, etc)
    '''
    Et = query.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    # values_in = tf.reshape(values, [-1, N, F])
    # values = tf.transpose(values_in, [2, 0, 1]) #[F,B,N]
    # import pdb; pdb.set_trace()
    query = tf.expand_dims(query, axis=1)
    values_T = tf.transpose(tf.squeeze(values, axis=1), [0, 2, 1])
    values = tf.squeeze(values, axis=1)
    # values = tf.squeeze(values, 1)
    with tf.variable_scope(scope):

        Wk = [tf.get_variable(f'Wk{i}', shape=[F, F], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wv = [tf.get_variable(f'Wv{i}',shape=[F, F],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wq1 = [tf.get_variable(f'Wq1{i}',shape=[Et, N],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]
        Wq2 = [tf.get_variable(f'Wq2{i}',shape=[1, F],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()) for i in range(num_attention_heads)]

        bias_v = [tf.get_variable(f'bias_v{i}', initializer=tf.zeros([F])) for i in range(num_attention_heads)]
        bias_k = [tf.get_variable(f'bias_k{i}', initializer=tf.zeros([F])) for i in range(num_attention_heads)]
        bias_q = [tf.get_variable(f'bias_q{i}', initializer=tf.zeros([F])) for i in range(num_attention_heads)]

        # Wv = tf.get_variable(shape=[N, F], name='Wv')
        # Wk = tf.get_variable(shape=[F, F], name='Wk')
        # Wq1 = tf.get_variable(shape=[Et, N], name='Wq1')
        # Wq2 = tf.get_variable(shape=[1, F], name='Wq2')

        mh_attention_weights = tf.get_variable("mh_head", shape=[num_attention_heads*F, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # Wk = tf.get_variable('Wk', shape=[Et, F], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        # Wv = tf.get_variable('Wv',shape=[F, N, 1],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        # Wq = tf.get_variable('Wq',shape=[Et, F],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

        # bias_v = tf.get_variable('bias_v', initializer=tf.zeros([F, 1, 1]))
        # bias_k = tf.get_variable('bias_k', initializer=tf.zeros([F]))
        # bias_q = tf.get_variable('bias_q', initializer=tf.zeros([F]))
    # i = 0
    # import pdb; pdb.set_trace()
    Q = [tf.transpose(query @ Wq1[i], [0, 2, 1]) @ Wq2[i] + bias_q[i] for i in range(num_attention_heads)]
    K = [values @ Wk[i] + bias_k[i] for i in range(num_attention_heads)]
    V = [values @ Wv[i] + bias_v[i] for i in range(num_attention_heads)]

    # Q = [tf.matmul(query, Wq[i]) + bias_q[i] for i in range(num_attention_heads)]
    # K = [tf.matmul(values, Wk[i]) + bias_k[i] for i in range(num_attention_heads)]
    # V = [tf.matmul(values, Wv[i]) + bias_v[i] for i in range(num_attention_heads)]

    att = [tf.transpose(tf.nn.softmax((Q[i] @ tf.transpose(K[i], [0,2,1])) / F**0.5, axis=2) @ V[i], [0,2,1]) for i in range(num_attention_heads)]

    # att = tf.transpose(tf.reduce_sum((tf.transpose(query @ Wq1, [0, 2, 1]) @ Wq2) @ V, axis=2, keepdims=True), [0,2,1])

    # att = [tf.nn.softmax((Q[i] @ tf.transpose(K[i])) / (F)**0.5, axis=1) @ V[i] for i in range(num_attention_heads)]
    # att = [tf.transpose(att[i], [1, 0, 2]) for i in range(num_attention_heads)]
    # out = [tf.matmul(values_in, att[i]) for i in range(num_attention_heads)]
    # import pdb; pdb.set_trace()
    out = tf.concat(att, axis=1, name='mh_att_concat')
    # import pdb; pdb.set_trace()
    # out = tf.squeeze(out, axis=2)
    out = tf.transpose(out, [0, 2, 1])
    out = out @ mh_attention_weights
    # out = tf.expand_dims(out, axis=-1)
    # import pdb; pdb.set_trace()

    return out


def attention_c_add(query, values, scope, num_attention_heads=None):
    '''
    :param query: a tensor shaped [B, Et]
    :param values: a tensor shaped [B, 1, H*W, F]
    :return:

    Notes:
    -----------------
    B: Batch size
    Et: External parameters (target hour, etc)
    '''
    Et = query.get_shape().as_list()[1]
    N = values.get_shape().as_list()[2]
    F = values.get_shape().as_list()[3]
    values_in = tf.reshape(values, [-1, N, F])
    values = tf.transpose(values_in, [2, 0, 1]) #[F,B,N]
    with tf.variable_scope(scope):
        Wv = tf.get_variable('Wv', shape=[F, N,1], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())  #[F,N,1]
        bias_v = tf.get_variable('bias_v', initializer=tf.zeros([F]))
        Wq = tf.get_variable('Wq', shape=[Et, F], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
    # EQUATION: 8 - Channel attention
    value_linear = tf.reshape(tf.transpose(tf.matmul(values, Wv), [1, 0,2]), (-1, F))
    score = tf.nn.tanh((value_linear + bias_v) + tf.matmul(query, Wq))
    score = tf.nn.softmax(score, dim=1) #shape is [B,F]
    # EQUATION: 9 - Apply attention to scores
    values = tf.matmul(values_in, tf.expand_dims(score,axis=-1)) #[B,N,1]
    # import pdb; pdb.set_trace()
    # EQUATION: 9 - values = prediction
    return values


def attention_c(query, values, scope, num_attention_heads, attention_type):
    if attention_type == 'multiplicative':
        return attention_c_mul(query, values, scope, num_attention_heads=num_attention_heads)
    elif attention_type == 'additive':
        return attention_c_add(query, values, scope)
    else:
        raise ValueError("Unknown attention type: %s" % attention_type)


def attention_t(query, values, scope, num_attention_heads, attention_type):
    if attention_type == 'multiplicative':
        return attention_t_mul(query, values, scope, num_attention_heads=num_attention_heads)
    elif attention_type == 'additive':
        return attention_t_add(query, values, scope)
    else:
        raise ValueError("Unknown attention type: %s" % attention_type)


class Graph(object):
    def __init__(self, adj_mx, params, is_training, num_blocks_to_use=6, opt="Adam"):

        # self.adj_mx = adj_mx
        self.supports = np.float32(Cheb_Poly(Scaled_Laplacian(adj_mx), 2))
        self.params = params
        C, O = params.closeness_sequence_length, params.nb_flow
        H, W, = params.map_height, params.map_width
        Et, Em = params.et_dim, params.em_dim
        Horizon = params.horizon
        self.c_inp = tf.placeholder(tf.float32, [None, C, H, W, O], name='c_inp')
        inputs = tf.reshape(self.c_inp, [-1, C, H * W, O])  # [batch, seq_len, num_nodes, dim]
        self.et_inp = tf.placeholder(tf.float32, (None, Horizon, Et), name='et_inp')
        self.labels = tf.placeholder(tf.float32, shape=[None, Horizon, H, W, O], name='label')
        labels = tf.reshape(self.labels, (-1, Horizon, H * W, O))
        self.opt = opt
        input_conv = inputs


        for block in range(1, num_blocks_to_use + 1):
            block_name = 'block' + str(block)
            dim_in = 32
            if block == 1:
                dim_in = O
            with tf.variable_scope(block_name):
                l_inputs = Conv_ST(input_conv, self.supports, kt=3, dim_in=dim_in, dim_out=32, activation='GLU')
                l_inputs = LN(l_inputs, 'ln' + str(block))
                input_conv = l_inputs
        # #long term encoder, encoding 1 to 12
        # with tf.variable_scope('block1'):
        #     l_inputs = Conv_ST(inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation ='GLU')
        #     l_inputs = LN(l_inputs, 'ln1')
        # with tf.variable_scope('block2'):
        #     l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
        #     l_inputs = LN(l_inputs, 'ln2')
        # with tf.variable_scope('block3'):
        #     l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
        #     l_inputs = LN(l_inputs, 'ln3')
        # with tf.variable_scope('block4'):
        #     l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
        #     l_inputs = LN(l_inputs, 'ln4')
        # with tf.variable_scope('block5'):
        #     l_inputs = Conv_ST(l_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
        #     l_inputs = LN(l_inputs, 'ln5')
        # with tf.variable_scope('block6'):
        #     l_inputs = Conv_ST(l_inputs, self.supports, kt=2, dim_in=32, dim_out=32, activation='GLU')
        #     l_inputs = LN(l_inputs, 'ln6')

        #short term encoder, working differently for training and testing
        preds = []
        window = 3
        if is_training == True:
            label_padding = inputs[:, -window:, :, :]
            padded_labels = tf.concat((label_padding, labels), axis=1)
            print(padded_labels.shape)
            padded_labels = tf.stack([padded_labels[:, i:i + window, :, :] for i in range(0, Horizon)], axis=1)
            print('shape of padded labels:', padded_labels.shape)  # [B, Horizon, window, H*W, O]
            for i in range(0, Horizon):
                s_inputs = padded_labels[:, i, :, :, :]  #[B, window, N, O]
                et_inp = self.et_inp[:, i, :]
                with tf.variable_scope('horizon'+str(i)):
                    with tf.variable_scope('block7'):
                        gs_inputs = Conv_ST(s_inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln7')
                    # with tf.variable_scope('block8'):
                    #     gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                    #     gs_inputs = LN(gs_inputs, 'ln8')
                    with tf.variable_scope('block9'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln9')
                    ls_inputs = tf.concat((gs_inputs, l_inputs), axis=1)
                    print(ls_inputs.shape)
                    ls_inputs = attention_t(et_inp, ls_inputs, 'attn_t', self.params.num_attention_heads, self.params.temporal_attention_type)
                    if params.nb_flow == 1:
                        pred = attention_c(et_inp, ls_inputs, 'dim1', self.params.num_attention_heads, self.params.channel_attention_type)
                    if params.nb_flow == 2:
                        pred = tf.concat((attention_c(et_inp, ls_inputs, 'dim1', self.params.num_attention_heads, self.params.channel_attention_type),
                                          attention_c(et_inp, ls_inputs, 'dim2', self.params.num_attention_heads, self.params.channel_attention_type)), axis=-1)
                preds.append(pred)
        else:
            label_padding = inputs[:, -window:, :, :]
            for i in range(0, Horizon):
                s_inputs = label_padding
                et_inp = self.et_inp[:, i, :]
                with tf.variable_scope('horizon' + str(i)):
                    with tf.variable_scope('block7'):
                        gs_inputs = Conv_ST(s_inputs, self.supports, kt=3, dim_in=O, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln7')
                    '''
                    with tf.variable_scope('block8'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln8')
                    '''
                    with tf.variable_scope('block9'):
                        gs_inputs = Conv_ST(gs_inputs, self.supports, kt=3, dim_in=32, dim_out=32, activation='GLU')
                        gs_inputs = LN(gs_inputs, 'ln9')
                    ls_inputs = tf.concat((gs_inputs, l_inputs), axis=1)
                    print(ls_inputs.shape)
                    ls_inputs = attention_t(et_inp, ls_inputs, 'attn_t', self.params.num_attention_heads, self.params.temporal_attention_type)
                    if params.nb_flow == 1:
                        pred = attention_c(et_inp, ls_inputs, 'dim1', self.params.num_attention_heads, self.params.channel_attention_type)
                    if params.nb_flow == 2:
                        pred = tf.concat((attention_c(et_inp, ls_inputs, 'dim1', self.params.num_attention_heads, self.params.channel_attention_type),
                                          attention_c(et_inp, ls_inputs, 'dim2', self.params.num_attention_heads, self.params.channel_attention_type)), axis=-1)
                label_padding = tf.concat((label_padding[:, 1:,:,:], tf.expand_dims(pred, 1)), axis=1)
                preds.append(pred)

        self.preds = tf.stack(preds, axis=1)

        first_pred = preds[0]
        first_label = labels[:, 0, :, :]
        first_loss = tf.nn.l2_loss(first_pred - first_label)

        self.loss = tf.nn.l2_loss(self.preds - labels)
        #self.loss = tf.nn.l2_loss(self.preds - labels) + first_loss
        if(self.opt == "Adam"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=params.lr, beta1=params.beta1, beta2=params.beta2,
                                                    epsilon=params.epsilon).minimize(self.loss)
        elif(self.opt == "GradientDescent"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.lr).minimize(self.loss)
        elif(self.opt == "AdaGrad"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=params.lr).minimize(self.loss)
        elif(self.opt == "RMSProp"):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=params.lr, epsilon=params.epsilon).minimize(self.loss)
            
        self.mean_rmse = RMSE(self.preds, labels) * params.scaler

        self.mae = []
        self.rmse = []
        self.mape = []
        self.r2 = []
        trues = tf.unstack(labels, axis=1)
        for _, (i, j) in enumerate(zip(preds, trues)):
            mae = MAE(i, j) * params.scaler
            self.mae.append(mae)
            rmse = RMSE(i, j) * params.scaler
            self.rmse.append(rmse)
            mape = MAPE(i, j, params.scaler, mask_value=10)
            self.mape.append(mape)
            r2 = R2(i, j)
            self.r2.append(r2)
