Model_type = 'conv_trans'#'conv_trans'       ######################
app = 'nosh_1h1l_allf'#'sh_1h1l_allf'# nosh_allf
epoch = 30
input_length = 320 * 128

def get_parameter():
    params = dict(
        n_fft = 320,
        eta2 = [0.1], #, 1e6], #1, 10],
        learning_rate = 1e-4,
        input_length = input_length,
        model_type = Model_type,
        app = app,
        epoch = epoch,
        model_path = Model_type + str(input_length) + '_' + app,
        test_eta2 = 0.1,
        batch_size = 8,
        gamma = 1.0,
        train_data_root = '../data/train_datas_ori', 
        setF = 175,
        _lambda = 1e-3,
        _M = 15
    )
    return params

    # 441 220.5 147 110.25 88.2 73.5 63 55.125 49 44.1
