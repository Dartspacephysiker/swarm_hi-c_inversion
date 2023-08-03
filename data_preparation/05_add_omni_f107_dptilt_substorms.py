##############################
import pandas as pd
import numpy as np
# from dipole import dipole_tilt
from dipole import Dipole
from datetime import datetime

from directories import filepath as datapath
from directories import masterhdfdir

PERIOD = '20Min'

VERSION = '0302'

hdfsuff = '_5sres_44qdlat'
# hdfsuff = '_5sres'
# hdfsuff = '_5sres_allmlat'
# hdfsuff = '_1sres'

####################
# sats = ['Sat_A','Sat_B','Sat_C']
# sats = ['Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']

print("The plan: Add OMNI, F10.7, and dipole tilt to HDF files")

##############################
## Get year range for selected DB type
if hdfsuff == '_5sres_44qdlat':
    y1 = '2013'
    y2 = '2024'
elif hdfsuff == '_5sres_allmlat':
    y1 = '2013'
    y2 = '2022'
elif hdfsuff == '_5sres':
    y1 = '2013'
    y2 = '2022'
elif hdfsuff == '_1sres':
    y1 = '2013'
    y2 = '2021'


def getIndices(seq, vals):
    return np.array([(np.abs(seq-val)).argmin() for val in vals])

##############################
# OMNI

print("Load OMNI first ... ",end='')
omnicols = dict(bz = 'BZ_GSM',
                by = 'BY_GSM',
                vx = 'Vx',
                nsw='proton_density')
with pd.HDFStore(datapath + 'omni_1min.h5', 'r') as omni:
    # ORIG
    # Bz = omni['/omni'][omnicols['bz']][y1:y2].rolling(PERIOD).mean()
    # By = omni['/omni'][omnicols['by']][y1:y2].rolling(PERIOD).mean()
    # vx = omni['/omni'][omnicols['vx']][y1:y2].rolling(PERIOD).mean()

    # NEW
    external = omni['/omni']
    external = external[~external.index.duplicated(keep='first')]
    Bz = external[omnicols['bz']][y1:y2].rolling(PERIOD).mean()
    By = external[omnicols['by']][y1:y2].rolling(PERIOD).mean()
    vx = external[omnicols['vx']][y1:y2].rolling(PERIOD).mean()
    nsw = external[omnicols['nsw']][y1:y2].rolling(PERIOD).mean()

external = pd.DataFrame([Bz, By, vx, nsw]).T
external.columns = ['Bz', 'By', 'vx', 'nsw']
external = external.dropna()

##############################
# F10.7

print("og så  F10.7 ... ",end='')
f107cols = dict(f107obs='observed_flux (solar flux unit (SFU))',
                f107adj='adjusted_flux (solar flux unit (SFU))')

f107 = pd.read_csv(datapath + 'penticton_radio_flux.csv', sep = ',', parse_dates= True, index_col = 0)  
f107[f107 == 0] = np.nan
# Convert Julian to datetime 
time = np.array(f107.index)
epoch = pd.to_datetime(0, unit = 's').to_julian_date()
time = pd.to_datetime(time-epoch, unit = 'D')
#set datetime as index
f107 = f107.reset_index()
f107.set_index(time, inplace=True)
f107 = f107[~f107.index.duplicated(keep='first')]
f107 = f107.sort_index()
f107 = f107[f107.index >= pd.Timestamp(y1+'-01-01')]

# interpolate f107: 
print("(interpolating F10.7 to match OMNI ...) ",end='')
f107[f107cols['f107obs']][f107[f107cols['f107obs']] < 0] = np.nan
f107[f107cols['f107adj']][f107[f107cols['f107adj']] < 0] = np.nan
# there is a huge data gap last 8 months of 2018 - I just inteprolate over this
f107 = f107.reindex(f107.index.union(external.index)).interpolate(method = 'linear', limit = 24*60*8*31)
for key,val in f107cols.items():
    external[key] = f107[val]

##############################
# Dipole tilt

print("then dipole tilt and B0_IGRF ... ",end='')
external['tilt'] = np.nan
for year in np.unique(external.index.year):
    print('calculating tilt for %s' % year)
    external.loc[str(year), 'tilt'] = Dipole(year+0.5).tilt(external[str(year)].index)
    # external.loc[str(year), 'tilt'] = dipole_tilt(external[str(year)].index, year)

##############################
# Add data to master hdfs

for sat in sats:
    # print(sat)

    masterhdf = f'{sat}_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    # Check if sorted, and sort if not
    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        print("HDF is monotonic: ",store['/mlat'].index.is_monotonic)
        if not store['/mlat'].index.is_monotonic:
            print("Sorting index of each key")
            keys = store.keys()
            for key in keys:
                print(key[1:]+' -> '+key,end=',')
                store.append(key[1:],store[key].sort_index(),format='t',append=False)
            print("")

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

    # align external data with satellite measurements
    sat_external = external.reindex(times, method = 'nearest', tolerance = '2Min')

    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        store.append('/external', sat_external, data_columns = True, append = False)
        print('added %s' % (sat + '/external'))
