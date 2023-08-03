""" script to read HDF store, calculate all the weights, and store in weights, coordinates, and measurements 
    in  a format which can be streamed (using dask)     
"""

import dask.array as da
import numpy as np
import pandas as pd
from sh import nterms
from geodesy import geoc2geod

from directories import masterhdfdir
from directories import filepath as datapath

NT, MT, NV, MV = 65, 3, 45, 3
NEQ = nterms(NT, MT, NV, MV)

maxf107 = 300
minh = 410
mintstamp = "2014-05-01 00:00:00"

# output = 'modeldata_v1_update.hdf5' # where the data will be stored
# output = 'modeldata_v2_update.hdf5' # where the data will be stored, version created 20211119 with CT2Hz data through 2021/08
# output = 'modeldata_v3_update.hdf5' # version created 20230601 with CT2Hz data through 2023/04
output = 'modeldata_v3.01_update.hdf5' # version created 20230607 with CT2Hz data through 2023/04, dropping data below min height and above max F10.7
output = 'modeldata_v3.1_update.hdf5' # version created 20230607 with CT2Hz data through 2023/04, dropping data below min height, above max F10.7, and before May 2014 to avoid commissioning period when lots of small maneuvers were undertaken

# satellites = ['SwarmA', 'SwarmB']
# satmap = {'CHAMP':1, 'SwarmA':2, 'SwarmB':3, 'SwarmC':4}
sats = ['Sat_A','Sat_B']
satmap = {'Sat_A':1, 'Sat_B':2}

VERSION = '0302'

hdfsuff = '_5sres_44qdlat'
# hdfsuff = '_5sres'
# hdfsuff = '_5sres_allmlat'

##############################
# Quality flag enforcement
##############################
# Do you want to only include measurements with a certain quality flag?
# For data version 0302, the second bit ('0100') being set means that v_(i,y) is calibrated
# See section 3.4.1.1 in "EFI TII Cross-Track Flow Data Release Notes" (Doc. no: SW-RN-UOC-GS-004, Rev: 7)
enforce_quality_flag = True     
quality_flag = '0100'

# For example, as of 20211119 if you do the following with
#store = pd.HDFStore('Sat_A_ct2hz_v0302_5sres.h5','r'),
#you get '9047089' back:
#np.sum((store['/Quality_flags'].values & int(quality_flag,2)) > 0)


# Here's the proof that we can use lperptoB_dot_ViyperptoB as the LHS of Eq (7) in 'SHEIC DERIVATION'
# lperptoB_dot_ViyperptoB = store['/lperptoB_E']*store['/ViyperptoB_E']+store['/lperptoB_N']*store['/ViyperptoB_N']+store['/lperptoB_U']*store['/ViyperptoB_U']
# ViyperptoB = np.sqrt(store['/ViyperptoB_E']**2+store['/ViyperptoB_N']**2++store['/ViyperptoB_U']**2)
# print((np.abs(lperptoB_dot_ViyperptoB)-ViyperptoB).max())
# Out[37]: 7.275957614183426e-12
# print((np.abs(lperptoB_dot_ViyperptoB)-ViyperptoB).min())
# Out[38]: -2.7284841053187847e-11


columns = ['mlat', 'mlt','lperptoB_dot_e1','lperptoB_dot_e2','Latitude','Radius']
# columns_for_derived = ['d10','d11','d12',
#                        'd20','d21','d22',
#                        'lperptoB_E','lperptoB_N','lperptoB_U',
#                        'ViyperptoB_E','ViyperptoB_N','ViyperptoB_U']

choosef107 = 'f107obs'
print("Should you use 'f107obs' or 'f107adj'??? Right now you use "+choosef107)
ext_columns = ['vx', 'Bz', 'By', choosef107, 'tilt']

# Derived columns: ["lperptoB_dot_ViyperptoB","Be3_in_Tesla","D"]

# put the satellite data in a dict of dataframes:
subsets = {}
external = {}
for sat in sats:
    # print(sat)

    inputhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(inputhdf)

    with pd.HDFStore(masterhdfdir+inputhdf, 'r') as store:

        print ('reading %s' % sat)
        tmpdf = pd.DataFrame()
        tmpextdf = pd.DataFrame()

        # Make df with getcols
        print("Getting main columns ... ",end='')
        for wantcol in columns:
            tmpdf[wantcol] = store['/'+wantcol]

        print("Getting ext columns ...",end='')
        tmpextdf = store.select('/external', columns = ext_columns)
        # for wantcol in ext_columns:
            # tmpextdf[wantcol] = store['/'+wantcol]

        print("Getting derived quantities ...")
        print("D",end=', ')
        D = np.sqrt( (store['d11']*store['d22']-store['d12']*store['d21'])**2 + \
                     (store['d12']*store['d20']-store['d10']*store['d22'])**2 + \
                     (store['d10']*store['d21']-store['d11']*store['d20'])**2)
        print("Be3_in_Tesla",end=', ')
        tmpdf['Be3_in_Tesla'] = store['B0IGRF']/D/1e9
        print("lperptoB_dot_ViyperptoB",end=', ')
        tmpdf['lperptoB_dot_ViyperptoB'] = store['/lperptoB_E']*store['/ViyperptoB_E']+\
            store['/lperptoB_N']*store['/ViyperptoB_N']+\
            store['/lperptoB_U']*store['/ViyperptoB_U']
        print("OK!")
        # subsets[sat]  = store.select('/apex_data', columns = columns)
        # external[sat] = store.select(sat + '/external', columns = ext_columns)

        if enforce_quality_flag:

            print(f"Dropping records that do not have Quality_flag == {quality_flag} ...")

            nNow = len(tmpdf)
            tmpdf = tmpdf[(store['/Quality_flags'].values & int(quality_flag,2)) > 0]
            nLater = len(tmpdf)
            
            print(f"Dropped {nNow-nLater} of {nNow} records ({(nNow-nLater)/nNow*100:.3f}%)")


        subsets[sat]  = tmpdf
        external[sat] = tmpextdf

# add alpha charlie-weight to main df:
print ('adding satellite weights ...',end='')
for sat in sats:
    # if sat in ['SwarmA', 'SwarmC']:
    #     subsets[sat]['s_weight'] = 1.0
    # else:
    subsets[sat]['s_weight'] = 1.0

# add external parameters to main df - and drop nans:
print ('adding external to main dataframe')
for sat in sats:
    subsets[sat][external[sat].columns] = external[sat]
    length = len(subsets[sat])
    subsets[sat] = subsets[sat].dropna()
    print ('dropped %s out of %s datapoints because of nans' % (length - len(subsets[sat]), length))

print ('merging the subsets')
full = pd.concat(subsets)

# full['time'] = full.index
sat  = [k[0] for k in full.index]
time = [k[1] for k in full.index]
full['time'] = time
full['sat'] = sat

full.index = range(len(full))

gdlat, height, _, _ = geoc2geod(full.Latitude.values, full.Radius.values/1000,
                                np.zeros_like(full.Radius.values), np.zeros_like(full.Radius.values))


keepi = (full[choosef107] <= maxf107) & (height >= minh) & (full['time'] >= pd.Timestamp(mintstamp))
print(f"Dropping {(~keepi).sum()} observations for which either F10.7 > {maxf107} or height < {minh} km or time < {mintstamp}, or some combo")

full = full[keepi]
full.index = range(len(full))

# Drop radius and latitude columns
print("Dropping Radius and Latitude columns, as no longer needed")
full = full.drop(labels=['Radius','Latitude'], axis=1)

# calculate weights:
print ('calculating weights')
B = np.sqrt(full['By']**2 + full['Bz']**2)
ca = np.arctan2(full['By'], full['Bz'])
epsilon = full['vx'].abs()**(4/3.) * B**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 # mV/m # Newell coupling
tau     = full['vx'].abs()**(4/3.) * B**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # mV/m # "anti"(?)-Newell coupling
f107    = full[choosef107]
tilt = full['tilt']
full['w01' ] = 1              * np.sin(ca)
full['w02' ] = 1              * np.cos(ca)
full['w03' ] = epsilon
full['w04' ] = epsilon        * np.sin(ca)
full['w05' ] = epsilon        * np.cos(ca)
full['w06' ] = tilt
full['w07' ] = tilt           * np.sin(ca)
full['w08' ] = tilt           * np.cos(ca)
full['w09' ] = tilt * epsilon
full['w10' ] = tilt * epsilon * np.sin(ca)
full['w11' ] = tilt * epsilon * np.cos(ca)
full['w12' ] = tau
full['w13' ] = tau            * np.sin(ca)
full['w14' ] = tau            * np.cos(ca)
full['w15' ] = tilt * tau    
full['w16' ] = tilt * tau     * np.sin(ca)
full['w17' ] = tilt * tau     * np.cos(ca)
full['w18' ] = f107.copy()


columns = ['mlat', 'mlt',
           'Be3_in_Tesla', 'lperptoB_dot_ViyperptoB',
           'lperptoB_dot_e1','lperptoB_dot_e2',
           's_weight', 
           'w01', 'w02', 'w03', 'w04', 'w05', 'w06',
           'w07', 'w08', 'w09', 'w10', 'w11', 'w12',
           'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 
           'time', 'sat_identifier']

full['sat_identifier'] = full['sat'].map(satmap)
full['time'] = np.float64(full['time'].values)

print (full.shape)
chunksize = NEQ

newstore = da.to_hdf5(masterhdfdir+output, dict(zip(['data/' + k for k in columns], [da.from_array(full[k].values, chunksize) for k in columns])))
print ('put model data in %s' % output)
