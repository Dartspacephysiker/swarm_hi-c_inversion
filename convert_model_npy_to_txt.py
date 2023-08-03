from utils import make_model_coeff_txt_file_analyticpot_zero
datadir = 'datadir/'
prefix_model_fn = datadir+'model_v3.1FULL_analyticpotzero_at_47,-47deg_v3.1_iteration_'
NT, MT = 65, 3
i = 61
coeff_fn = prefix_model_fn + str(i) + '.npy'
make_model_coeff_txt_file_analyticpot_zero(coeff_fn,
                                           NT=NT,MT=MT,
                                           TRANSPOSEEM=False,
                                           PRINTOUTPUT=False)
