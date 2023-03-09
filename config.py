Model_type = 'conv_trans'#'conv_trans'       ######################
app = 'sh_1h1l_allf'#'sh_1h1l_allf'# nosh_allf
epoch = 30
input_length = 320 * 128

def get_parameter():
    params = dict(
        n_fft = 320,
        learning_rate = 1e-4,
        input_length = input_length,
        model_type = Model_type,
        app = app,
        epoch = epoch,
        model_path = Model_type + str(input_length) + '_' + app,
        batch_size = 8,
        gamma = 1.0,
        train_data_root = './data/train_datas', 
        _lambda = 1e-3,
        _M = 15
    )
    return params
