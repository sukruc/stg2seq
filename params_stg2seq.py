class ParamMeta(type):
    channel_attention_type = 'additive'
    temporal_attention_type = 'additive'
    num_attention_heads = 4
    lr = 0.0007
    @property
    def model_path(self):
        return f"{self.channel_attention_type}-{self.temporal_attention_type}-{self.num_attention_heads}-{self.lr}"


class ModelParam(metaclass=ParamMeta):
    threshold = 0.05
    em_dim = None
    # model_path = 'newmodel'
    num_attention_heads = 2
    num_blocks_to_use = 3
    num_self_attention_blocks = 8
    num_epochs = 50
    temporal_attention_type = 'additive'  # {'additive', 'multiplicative'}
    channel_attention_type = 'additive'
    batch_size = 32
    optimizer = 'Adam'


class params_SY(ModelParam):
    #commen params
    source = 'SY'
    scaler = 81
    batch_size = 32
    map_height = 6
    map_width = 6
    closeness_sequence_length = 8
    period_sequence_length = 0
    trend_sequence_length = 0
    nb_flow = 1
    external_length = 8
    et_dim = 33
    ew_dim = 7
    horizon = 3
    lr = 0.0001
    num_epochs = 400
    logdir = "train"
    test_days = 6
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    #specific params
    filter_type = 'random_walk'
    num_layers = 2
    num_units = 64
    num_nodes = map_height*map_width
    keep_prob = 1
    exlstm_layers = 5

class params_SY_IR(ModelParam):
    #when DidiSY dataset is partitioned by roadnetworks
    source = 'SY_IR'
    scaler = 70
    batch_size = 32
    map_height = 35
    map_width = 1
    closeness_sequence_length = 8
    period_sequence_length = 0
    trend_sequence_length = 0
    nb_flow = 1
    external_length = 8
    et_dim = 33
    ew_dim = 7
    horizon = 3
    lr = 0.0005
    num_epochs = 1200
    logdir = "train"
    test_days = 6
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    #specific params
    filter_type = 'random_walk'
    num_layers = 2
    num_units = 64
    num_nodes = map_height*map_width
    keep_prob = 1
    exlstm_layers = 5

class params_NYC(ModelParam):
    source = 'NYC'
    scaler = 267
    map_height = 16
    map_width = 8
    closeness_sequence_length = 5
    period_sequence_length = 0
    trend_sequence_length = 0
    nb_flow = 2
    external_length = closeness_sequence_length
    et_dim = 31
    ew_dim = 0
    horizon = 3
    logdir = "train"
    test_days = 10
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    filter_type = 'random_walk'

    num_layers = 2
    num_units = 64
    num_nodes = map_height * map_width
    keep_prob = 1
    exlstm_layers = 3

class params_BJ(ModelParam):
    source = 'BJ'
    scaler = 1274
    batch_size = 32
    map_height = 32
    map_width = 32
    closeness_sequence_length = 8
    period_sequence_length = 0
    trend_sequence_length = 0
    nb_flow = 2
    external_length = 8
    et_dim = 57
    ew_dim = 19
    horizon = 3
    lr = 0.001
    num_epochs = 300
    logdir = "train"
    test_days = 10
    delta = 0.5
    epsilon = 1e-7
    beta1 = 0.8
    beta2 = 0.999
    filter_type = 'random_walk'

    num_layers = 2
    num_units = 64
    num_nodes = map_height * map_width
    keep_prob = 1
    exlstm_layers = 2
