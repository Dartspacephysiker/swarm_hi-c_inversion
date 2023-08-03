#!/usr/bin/env python
# coding: utf-8

# 01_make_swarm_CT2Hz_hdfs.py          : Here we open up all the .cdf.zip files and put their contents, along with some stuff in Modified Apex-110 coordinates, into an HDF file.
# 02_f107_download_and_filter.py       : Add F10.7
# 03_omni_download_1min_data.py        : Download OMNI data (IMF components, solar wind speed and density, SYM-H, + others?
# 04_omni_process_1min_data.py         : Process OMNI data (calculate IMF clock angle mean,variance, average over 30-min window, etc.)
# 05_add_omni_f107_dptilt_substorms.py : Sort index of each column in HDF file, add F10.7, OMNI, dipole tilt, B0_IGRF, og resten til HDF-filen
# 06_add_crosstrack_vector_info.py     : Calculate cross-track convection in MA-110 coordinates, add these to HDF:
#                                        ['Viy_d1','Viy_d2',
#                                         'Viy_f1','Viy_f2',
#                                         'yhat_d1', 'yhat_d2',
#                                         'yhat_f1', 'yhat_f2',
#                                         'gdlat', 'alt']
# 07_make_model_dataset.py             : read HDF store files, calculate all the weights,
#                                        (optionally) retain only measurements with a particular quality flag,
#                                        and then store weights, coordinates, and measurements in  a format that can be streamed (using dask)

# From swarmProcHelper.processSwarm2HzCrossTrackZip you get the following:
# 'Bx','By','Bz',
# 'Ehx','Ehy','Ehz',
# 'Evx','Evy','Evz',
# 'Latitude','Longitude','Radius',
# 'MLT','QDLatitude',
# 'Quality_flags',
# 'Time',
# 'Vicrx','Vicry','Vicrz',
# 'Vixh','Vixh_error',
# 'Vixv','Vixv_error',
# 'Viy','Viy_error',
# 'Viz','Viz_error',
# 'VsatC','VsatE','VsatN'

# From hCoord.geodetic2apex you get:
# MA-110 mlat, mlon, and mlt, as well as MA-110 e, d, and f base vectors


# Script for downloading all Swarm CT 2-Hz files from Swarm site
#
# Spencer Mark Hatch
# Birkeland Centre for Space Science
# 2021-01-18

########################################
# Imports

import cdflib
from datetime import datetime 
import numpy as np
import pandas as pd
import os
from prep_helpers import getCT2HzFileDateRange, processSwarm2HzCrossTrackZip, geoclatR2geodlatheight, geodetic2apex

from glob import glob

########################################
# function/other definitions

def store_Swarm_CT2Hz_DF(fn,df):

    store = pd.HDFStore(fn)
    for column in df.columns:
        store.append(column, df[column], format='t')

    store.close()

apex__geodetic2apexOpts=dict(min_time_resolution__sec=0.5,
                             apexRefHeight_km=110,
                             max_N_months_twixt_apexRefTime_and_obs=3,
                             return_apex_d_basevecs=True,
                             return_apex_e_basevecs=True,
                             return_apex_f_basevecs=True)  # Need these for getting flow in Apex coordinates

########################################
# Some directories

from directories import DownloadDir,masterhdfdir

wantext = '.ZIP'
VERSION = '0302'

########################################
# Define which satellites we'll look for during which years 

# mode = 'fulldb_1sres'
# mode = 'fulldb_5sres'

mode = 'fulldb_5sres_44qdlat'

# mode = 'fulldb_5sres_allmlat'   # This is OK (i.e., you won't get NaNs back) because of what's in data_preparation/journal__20210907__why_its_OK_to_use_all_CT2Hz_latitudes_with_modified_apex_coords.py
                                  # HOWEVER: Keep in mind that QDLatitude will comfortably go all the way to 0 at the Equator, whereas Modified Apex latitude (which is what you use) doesn't go below ~13°, and I'm not sure it's meaningful.
                                  #(Just because it's not a NaN doesn't mean it's not garbage, right?)

VALIDMODES = ['fulldb_5sres_44qdlat',
              'fulldb_5sres_allmlat','fulldb_5sres','fulldb_1sres']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']

assert mode in VALIDMODES,"Must choose one of " + ",".join(VALIDMODES)+"!"

# which coordinate system to use for applying mlatlowlim?
mlat_coordsys = 'ma'                # modified apex

if mode == 'fulldb_5sres_44qdlat':
    # hdfsuff = '_5sres_44qdlat'
    hdfsuff = '_5sres_44qdlat_try2'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 44
    mlat_coordsys = 'qd'        # use 44°MLat in quasi-dipole coords, since that's what UCalgary uses in their calibration of TII data

    date0 = '2013-12-01 00:00:00'
    date1 = '2023-04-16 00:00:00'
    

elif mode == 'fulldb_5sres_allmlat':
    hdfsuff = '_5sres_allmlat'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 0

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

if mode == 'fulldb_5sres_allmlat':
    hdfsuff = '_5sres_allmlat'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 0

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

elif mode == 'fulldb_5sres':
    hdfsuff = '_5sres'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 45

    # date0 = '2013-12-01 00:00:00'
    # date1 = '2022-01-01 00:00:00'

    # Just update
    date0 = '2020-12-31 00:00:00'
    date1 = '2022-01-01 00:00:00'

elif mode == 'fulldb_1sres':
    hdfsuff = '_1sres'

    decimationfactor = 2           # so 1-s resolution
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

########################################
# Define which satellites we'll look for during which years 

date0 = datetime.strptime(date0,"%Y-%m-%d %H:%M:%S")
date1 = datetime.strptime(date1,"%Y-%m-%d %H:%M:%S")

# dates = pd.date_range(start=date0,end=date1,freq='1D')
# dates = [dato.strftime("%Y%m%d") for dato in dates]

if VERSION == '0101':
    fullext = VERSION + '.CDF.ZIP'
else:
    fullext = VERSION + '.ZIP'

########################################
# Process
for sat in sats:

    # if sat == sats[0]:
    #     print("Already have Swarm A. Continue!")
    #     continue

    if sat == sats[1]:
        print("Working on Swarm B. Continue!")
        continue

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    print(masterhdf)

    # Get times
    havemastertimes = False
    if os.path.exists(masterhdfdir+masterhdf):
        mastertimes = pd.read_hdf(masterhdfdir+masterhdf,'/mlat').copy().index
        havemastertimes = True

    # for year in years:
    curIterStr = f"{sat}-{VERSION}"

    opts_hurtigLast = dict(FP__doCorrectTimestamps=False,
                           FP__doResample=False,
                           dont_touch_data=False,
                           dontInterp__justMag=False,
                           doDebug=False,
                           overwrite_existing=False,
                           use_existing=True,
                           removeCDF=True,
                           resampleString='500ms',
                           customSaveSuff='',
                           make_pickles=False)


    fileza = glob(localdir+'*.ZIP')
    fileza.sort()

    timeranges = [getCT2HzFileDateRange(os.path.basename(fila)) for fila in fileza]

    for fila,tidrange in zip(fileza,timeranges):
        dirrie = os.path.dirname(fila)+'/'

        if (tidrange[0] < date0) or (tidrange[0] > date1):
            print("SKIPPING {fila} because outside time range")
            continue

        if havemastertimes:

            if np.sum((mastertimes >= tidrange[0]) & (mastertimes <= tidrange[1])):
                print(f"Already have {os.path.basename(fila)}! Continue ...")
                continue
            else:
                print(f"Adding {os.path.basename(fila)} to the mix ...")

        df = processSwarm2HzCrossTrackZip(dirrie, os.path.basename(fila), localdir,
                                          doResample=False,
                                          resampleString="62500000ns",
                                          skipEphem=True,
                                          quiet=False,
                                          removeCDF=True,
                                          rmCDF_noPrompt=True,
                                          dont_touch_data=False,
                                          include_explicit_calibration_flags=False)

        if df.isna().any()[['Latitude','Longitude','Radius','Quality_flags','Viy']].any():
            # assert 2<0
            print("Gotta drop stuff!")
            df = df.dropna(subset=['Latitude','Longitude','Radius','QDLatitude','Quality_flags','Viy'])

        # if len(df) != 0:
        if df is not None:
            df.sort_index(inplace=True)
            
            # Decimation 
            if decimationfactor > 1:
                df = df.iloc[::decimationfactor]


            # print("Adding Apex coords")
            
            gdlat, gdalt_km = geoclatR2geodlatheight(
                df["Latitude"].values, df["Radius"].values/1000.)

            apexDict2 = geodetic2apex(gdlat, df["Longitude"].values,
                                      gdalt_km,
                                      df.index.to_pydatetime(),
                                      **apex__geodetic2apexOpts)

            df = df.assign(**apexDict2)

            dfN = df[df['Latitude'] > 0]
            dfS = df[df['Latitude'] < 0]
            dfNm = df[df['mlat'] > 0]
            dfSm = df[df['mlat'] < 0]
            latlimsN = (dfN['Latitude'].min(), dfN['Latitude'].max())
            latlimsS = (dfS['Latitude'].min(), dfS['Latitude'].max())
            mlatlimsN = (dfNm['mlat'].min(), dfNm['mlat'].max())
            mlatlimsS = (dfSm['mlat'].min(), dfSm['mlat'].max())
            print(f"min/max latitude before application of mlatlowlim screening: N - ({latlimsN[0]:6.2f},{latlimsN[1]:6.2f}), S - ({latlimsS[0]:6.2f},{latlimsS[1]:6.2f}) ")
            print(f"min/max mlat     before application of mlatlowlim screening: N - ({mlatlimsN[0]:6.2f},{mlatlimsN[1]:6.2f}), S - ({mlatlimsS[0]:6.2f},{mlatlimsS[1]:6.2f}) ")

            if mlat_coordsys == 'ma':
                df = df[np.abs(df['mlat']) >= mlatlowlim]
            elif mlat_coordsys =='qd':
                df = df[np.abs(df['QDLatitude']) >= mlatlowlim]

            dfN = df[df['Latitude'] > 0]
            dfS = df[df['Latitude'] < 0]
            dfNm = df[df['mlat'] > 0]
            dfSm = df[df['mlat'] < 0]
            latlimsN = (dfN['Latitude'].min(), dfN['Latitude'].max())
            latlimsS = (dfS['Latitude'].min(), dfS['Latitude'].max())
            mlatlimsN = (dfNm['mlat'].min(), dfNm['mlat'].max())
            mlatlimsS = (dfSm['mlat'].min(), dfSm['mlat'].max())
            print(f"min/max latitude after application of mlatlowlim screening: N - ({latlimsN[0]:6.2f},{latlimsN[1]:6.2f}), S - ({latlimsS[0]:6.2f},{latlimsS[1]:6.2f}) ")
            print(f"min/max mlat     after application of mlatlowlim screening: N - ({mlatlimsN[0]:6.2f},{mlatlimsN[1]:6.2f}), S - ({mlatlimsS[0]:6.2f},{mlatlimsS[1]:6.2f}) ")

            # Add this DataFrame to master .h5 file
            store_Swarm_CT2Hz_DF(masterhdfdir+masterhdf,df)

    #     # Delete .cdf file
    #     # os.remove(localdir+fil)

    # print("")

    # break


# TEST READING HDF
# with pd.HDFStore(localdir+masterhdf, mode = 'r') as store:
#     indata = store[satellite + '/raw_data']

# print("Reading HDF file ...")
# with pd.HDFStore(localdir+masterhdf, mode='r') as hdf:
#     # This prints a list of all group names:
#     keys = hdf.keys()

# indata = pd.DataFrame({key.replace('/', ''): pd.read_hdf(localdir+masterhdf, key=key).values for key in keys},
#                       index=pd.read_hdf(localdir+masterhdf, key=keys[0]).index)

