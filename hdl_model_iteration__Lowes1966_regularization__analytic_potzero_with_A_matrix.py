"""
0) NOTE: If wanting to use ALL measurements and derive FULL model, use doFULL = True under "Select which type of model"
1) load data file and set up G0
2) load model vector and model values from iteration i
3) calculate huber weights
4) make system of equations (GTG, GTd)
5) solve - with the least possible regularization - and save model vector (i + 1)
6) calculate model values and save (i + 1)
7) go back to (2), unless ||model i + 1|| is close to ||model i||
"""

import numpy as np
import dask.array as da
import gc
import time
import h5py
import sys
from scipy.linalg import cholesky, cho_solve
from dask.diagnostics import ProgressBar
# from utils import nterms, SHkeys, getG_torapex_dask, make_model_coeff_txt_file
from utils import nterms_analytic_pot_zero, SHkeys, getG_torapex_dask_analytic_pot_zero, make_model_coeff_txt_file_analyticpot_zero
from gtg_array_utils import weighted_GTd_GTG_array, expand_GTG_and_GTd
from functools import reduce
# from hdl_model_iteration_helpers import itersolve, iterhuber

t0 = time.time()

# Get access to data_preparation/directories.py
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
if not dir_path+'data_preparation/' in sys.path:
    sys.path.append(dir_path+'data_preparation/')
from directories import masterhdfdir

DATAVERSION = 'v1'
DATAVERSION = 'v2'                                       # 2021/11/19
DATAVERSION = 'v3'                                       # 2023/06/01
DATAVERSION = 'v3.1'                                     # 2023/06/07

zero_lats = np.array([47.,-47.])
# zero_lats = np.array([-47.,47.])

## Select which type of model
MODELSUFF = '_analyticpotzero_at_47deg'
MODELSUFF = '_analyticpotzero_at_'+",".join([f"{this:.0f}" for this in zero_lats])+'deg_'+DATAVERSION

datafile       = masterhdfdir+f'modeldata_{DATAVERSION}_update.hdf5' # where the data are stored (see data_preparation/07_make_model_dataset.py)

MODELVERSION = DATAVERSION+'FINAL'

print("******************************")
print(f"MODEL VERSION: {MODELVERSION}")
print(f"Data file    : {datafile}")
print("******************************")
print("")

prefix_GTd_GTG_fn    = masterhdfdir+'matrices/model_'+MODELVERSION+'GTG_GTd_array_iteration_'
prefix_model_fn      = masterhdfdir+'matrices/model_'+MODELVERSION+'_iteration_'
prefix_model_value   = masterhdfdir+'matrices/model_'+MODELVERSION+'_values_iteration_'
prefix_huber_weights = masterhdfdir+'matrices/model_'+MODELVERSION+'_huber_iteration_'


""" MODEL/CALCULATION PARAMETERS """
i = -1 # number for previous iteration

NT, MT = 65, 3
# NT, MT = 60, 5
NEQ = nterms_analytic_pot_zero(NT, MT)

coeffsuff = ''
if (NT != 65) or (MT != 3):
    coeffsuff = f'_NT,MT={NT},{MT}'

MODELVERSION += coeffsuff

MODELVERSION = MODELVERSION + MODELSUFF

NWEIGHTS = 19
CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
    
    
print(f"NWEIGHTS, CHUNKSIZE: {NWEIGHTS}, {CHUNKSIZE}")
K = 5 # how many chunks shall be calculated at once

N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd (derived quantity - do not change)


""" HELPER FUNCTIONS """
def iterhuber(array, meanlim = 0.5, k = 1.5):
    """ compute mean and std with huber weights iteratively - repeat until |updated mean - old mean| < meanlim
    """


    m, s = huber(array, k = k)
    while True:
        newm, s = huber(array, k = k, inmean = m, instd = s)
        if np.abs(newm - m) < meanlim:
            m = newm
            break
        else:
            m = newm

    return m, s

def huber(array, k = 1.5, inmean = None, instd = None):
    """ compute huber mean of the array, using the Huber coefficient k. 
        adopted from matlab code by Nils Olsen and Egil Herland
    """

    if inmean is None and instd is None:
        mean_bare = np.mean(array)
        std_bare  = np.std(array)
        norm_res  = (array - mean_bare)/std_bare
    else:
        norm_res = (array - inmean)/instd

    # Huber weights
    w = k/np.abs(norm_res)
    w[w > 1.] = 1.

    # Huber mean and std:
    hmean = np.sum(w*array)/np.sum(w)
    hstd = np.sum(w*np.abs(array - hmean))/np.sum(w)

    return hmean, hstd

def itersolve(filename):

    # make regularization matrix:

    # lambda_V = 0
    lambda_T = 1.e5
    #lambda_T = 1.e4

    #lambda_T = 1.e2
    
    while True:
        # print( 'solving... with lambda_T = %s, lambda_V = %s' % (lambda_T, lambda_V))
        print( 'solving... with lambda_T = %s' % (lambda_T))
        try:
            # n_cos_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(0).n
            # n_sin_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(1).n
            n_cos_T = SHkeys(NT, MT).setNmin(1).MleN().Mge(0).Shaveoff_first_k_nterms_for_m_gt(2).n
            n_sin_T = SHkeys(NT, MT).setNmin(1).MleN().Mge(1).Shaveoff_first_k_nterms_for_m_gt(2).n
            
            GTd_GTG_num = np.load(filename)
            GTd, GTG = expand_GTG_and_GTd(GTd_GTG_num, NWEIGHTS, NEQ)
            
            # nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.), 
            #                 lambda_V * n_cos_V  * (n_cos_V  + 1.)                 , lambda_V * n_sin_V  * (n_sin_V  + 1.)                 )).flatten()
            # nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.))).flatten()
            
            # Regularization based on total power in the electric field (Lowes, 1966)
            nn = np.hstack((lambda_T * (n_cos_T  + 1.), lambda_T  * (n_sin_T  + 1.))).flatten()

            nn = np.tile(nn, NWEIGHTS)
                     
            R = np.diag(nn)
            
            c = cholesky(GTG + R, overwrite_a = True, check_finite = False)
            model_vector = cho_solve((c, 0), GTd)
            break # success!
        except:
            lambda_T *= 5 # increase regularization parameter by a factor of five
            gc.collect()
            continue

    return model_vector


""" HELPER FUNCTIONS DONE """


##################################
# (1) Load data file and set up G0
##################################
f = h5py.File(datafile, 'r')['/data']

# make a 2D array, one row for each item, and make a dictionary to map between the row and its name
names = [item[0] for item in f.items()]
datamap = dict(zip(names, range(len(names))))

# breakpoint()

data = da.vstack((da.from_array(f[name], chunks = CHUNKSIZE) for name in names))
ND = data.size/len(datamap) # number of datapoints
print( '%s - loaded data - %s points across %s arrays (dt = %.1f sec)' % (time.ctime(), ND, len(datamap), time.time() - t0))

# import warnings
# warnings.warn("2021/09/01 You have modified getG_torapex_dask so that coeffs have units mV/m (I hope!)")

G0 = getG_torapex_dask_analytic_pot_zero(NT, MT, 
                                         data[datamap['mlat'           ]].reshape((data.shape[1], 1)),
                                         15* data[datamap['mlt'        ]].reshape((data.shape[1], 1)),
                                         data[datamap['Be3_in_Tesla'   ]].reshape((data.shape[1], 1)),
                                         data[datamap['lperptoB_dot_e1']].reshape((data.shape[1], 1)),
                                         data[datamap['lperptoB_dot_e2']].reshape((data.shape[1], 1)),
                                         zero_lats=zero_lats)

G0 = G0.rechunk((G0.chunks[0], G0.shape[1]))
print( '%s - done computing G0 matrix graph. G0 shape is %s (dt = %.1f sec)' % (time.ctime(), G0.shape, time.time() - t0))

# data vector:
# d = da.hstack(tuple(data[datamap[key]] for key in ['Be', 'Bn', 'Bu']))
d = da.hstack(tuple(data[datamap[key]] for key in ['lperptoB_dot_ViyperptoB']))
d = d.reshape((d.size, 1)) # to column
print( '%s - made d vector (dt = %.1f sec)' % (time.ctime(), time.time() - t0))

# prepare static weights (0.5 for side-by side satellites)
s_weight = data[datamap['s_weight']]
# s_weight = da.hstack((s_weight, s_weight, s_weight)) # stack three times - one for each component
# s_weight = da.hstack((s_weight)) # stack once - one for each component

# prepare matrix of weights (the external parameters)

if NWEIGHTS > 1:
    # OLD (multiple weights)
    weights = da.vstack( tuple([data[datamap['w' + str(jj + 1).zfill(2)]] for jj in range(NWEIGHTS-1)]))
    weights = weights.astype(np.float32)
    weights = da.vstack((da.ones(weights.shape[1], chunks = weights[0].chunks), weights)) # add a 1 weight on top (c0)

elif NWEIGHTS == 1:
    # NEW (only constant weight term)
    weights = da.ones(data[datamap['w01']].shape[0],chunks = data[0].chunks)
    weights = weights.reshape((1,weights.shape[0]))

else:
    assert 2<0

# Reshape 'em
# weights = da.hstack((weights, weights, weights)).T # tile them and transpose, shape is (Nmeas*3, NWEIGHTS)
weights = weights.T  # tile them and transpose, shape is (Nmeas, NWEIGHTS)

# weights = da.hstack((weights)).T # tile them and transpose
weights = weights.rechunk((G0.chunks[0], NWEIGHTS))

# breakpoint()
print("Entering loop ...")
while True: # enter loop
    #########################################################################
    # (2) Load model vector, model values, and huber weights from iteration i
    #########################################################################
    
    if i != -1:
        model_i       = np.load(prefix_model_fn + str(i) + '.npy')     # model vector
        dm_i          = np.load(prefix_model_value + str(i) + '.npy')  # model values
        if i != 0:
            huber_weights = np.load(prefix_huber_weights + str(i) + '.npy').flatten()
        else:
            huber_weights = np.ones_like(dm_i) # huber weights in first iteration are just ones

        i += 1
    
        ############################
        # 3) calculate huber weights
        ############################
    
        residuals  = (dm_i - d.flatten().compute())   # weighted by satellite weights
    
        rms_misfit = np.sqrt(np.average(residuals**2, weights = s_weight.flatten() * huber_weights))
        print('%s - misfit in iteration %s was %.2f \n' % (time.ctime(), i, rms_misfit))
        huber_mean_residual, huber_std = iterhuber(residuals**2)
        sigma = np.sqrt(huber_mean_residual) # <- rms residual - huber mean
        huber_weights = 1.5 * sigma / np.abs(residuals)
        huber_weights[huber_weights > 1] = 1
        huber_weights = da.from_array(huber_weights[:, np.newaxis], chunks = d.chunks)
        np.save(prefix_huber_weights + str(i) + '.npy', huber_weights)
        print( 'mean huber weight: %.2f' % huber_weights.mean().compute())
        gc.collect()

    else:
        huber_weights = 1.
        model_i = np.array(0.)
        i += 1
    
    
    ########################################
    # 4) make system of equations (GTG, GTd)
    ########################################
    GTd_GTG = da.map_blocks(weighted_GTd_GTG_array, G0 * huber_weights * s_weight[:, np.newaxis],
                                                     d * huber_weights * s_weight[:, np.newaxis], 
                                                     weights, chunks = (1, N_NUM))
    ( '%s - made GTd_GTG matrix graph (dt = %.1f sec)\n' % (time.ctime(), time.time() - t0))
    
    t0 = time.time()
    print( '%s - ready to compute (dt = 0 min)' % time.ctime())
    
    GTd_GTG_num = np.zeros(N_NUM, dtype = np.float64) # initialize the array
    
    print( '%s - calculating in %s steps ' % (time.ctime(), GTd_GTG.numblocks[0]//K + 1))
    for jj in range(GTd_GTG.numblocks[0]//K + 1):
        t_start = time.time()
        gc.collect()
        if jj*K >= GTd_GTG.shape[0]:
            break
        with ProgressBar():
            GTd_GTG_num = GTd_GTG_num + GTd_GTG[jj*K:(jj+1)*K].sum(axis = 0).compute()
        print( '\r%s - %s/%s steps - approximately %.1f minutes left (dt = %.1f min)' % (time.ctime(), jj + 1, GTd_GTG.numblocks[0]/K, ((GTd_GTG.numblocks[0] - jj*K)/K) * (time.time() - t_start)/60, (time.time() - t0)/60))
        if jj % 10 == 0:
            np.save(prefix_GTd_GTG_fn + str(i) + '.npy', GTd_GTG_num)
            print( 'saved GTd_GTG_num in %s at i = %s' % (prefix_GTd_GTG_fn + str(i) + '.npy', jj))
    
    print ('\n%s - done computing - saving %s' % (time.ctime(), prefix_GTd_GTG_fn + str(i) + '.npy'))
    np.save(prefix_GTd_GTG_fn + str(i) + '.npy', GTd_GTG_num)
    
    ###################################################################################
    # 5) solve - with the least possible regularization - and save model vector (i + 1)
    ###################################################################################
    model_new = itersolve(prefix_GTd_GTG_fn + str(i) + '.npy')
    np.save(prefix_model_fn + str(i) + '.npy', model_new)
    print( 'saved new model in %s' % (prefix_model_fn + str(i) + '.npy'))
    
    ############################################
    # 6) calculate model values and save (i + 1)
    ############################################
    model_vectors = np.split(model_new, NWEIGHTS)
    model_vector  = da.from_array(model_new, chunks = (G0.shape[1]))
    
    dm = np.array([])
    for jj in range(len(G0.chunks[0])):
        gc.collect()
        gg = G0[jj*CHUNKSIZE:(jj+1)*CHUNKSIZE].compute()
        ww = weights[jj*CHUNKSIZE:(jj+1)*CHUNKSIZE].compute()
    
        dm_ = reduce(lambda x, y: x + y, ((gg * w[:, np.newaxis]).dot(m) for m, w in zip(model_vectors, ww.T)))
        dm = np.hstack((dm, dm_))
        print( '%s/%s, dt = %s' % (jj, len(G0.chunks[0]), time.time() - t0))
    
    np.save(prefix_model_value + str(i) + '.npy', dm)
    print( 'saved new model values in %s ' % (prefix_model_value + str(i) + '.npy'))
    
    ###################################################################
    # 7) go back to (2), unless ||model i + 1|| is close to ||model i||
    ###################################################################
    # difference = np.linalg.norm(model_new.flatten() - model_i.flatten())
    # if i > 0:
    #     if difference < 0.01 * np.linalg.norm(np.load(prefix_model_fn + '0.npy')):
    #         break
    # print( 'starting next iteration')

    difference = np.linalg.norm(model_new.flatten() - model_i.flatten())
    fracdiff = 0.
    if i > 0:
        fracdiff = difference/np.linalg.norm(np.load(prefix_model_fn + '0.npy'))
        if fracdiff < 0.01:
            print(f'difference for iteration {i}: {difference:.3f} (fracdiff rel to i=0: {fracdiff:.3f}). STOPPING')
            break
    
    print(f'difference for iteration {i}: {difference:.3f} (fracdiff rel to i=0: {fracdiff:.3f}). starting next iteration')
    

coeff_fn = prefix_model_fn + str(i) + '.npy'
make_model_coeff_txt_file_analyticpot_zero(coeff_fn,
                                           NT=NT,MT=MT,
                                           TRANSPOSEEM=False,
                                           PRINTOUTPUT=False)
print( 'done. DONE!!!')

