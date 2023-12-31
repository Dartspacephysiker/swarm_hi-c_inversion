#!/usr/bin/env python
# coding: utf-8

# Script for downloading all Swarm CT 2-Hz files from Swarm site
#
# Spencer Mark Hatch
# Birkeland Centre for Space Science
# 2021-01-18

from directories import DownloadDir
wantext = '.ZIP'
VERSION = '0302'

import wget
from glob import glob
filelistdir = '../filelists/'


########################################
# Imports

from bs4 import BeautifulSoup
from datetime import datetime 
import numpy as np
import pandas as pd
import os
import requests
from prep_helpers import getCT2HzFTP,getCT2HzFileDateRange

########################################
# Define which satellites we'll look for during which years 

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']
y0,y1 = 2013,2021
years = [str(val) for val in np.arange(y0,y1)]

date0 = datetime(y0,1,1,0)
date0 = datetime(2013,12,1,0)
date1 = datetime(y1,1,1,0)
dates = pd.date_range(start=date0,end=date1,freq='1D')
dates = [dato.strftime("%Y%m%d") for dato in dates]

########################################
# function definitions

def get_remote_swarm_ct2hz_dir(*args):
    """
    remoteDir = get_remote_swarm_ct2hz_dir(sat,year)
    """
    BaseAddr = 'swarm-diss.eo.esa.int/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/'
    sat = args[0]
    version = args[1]
    print(BaseAddr)
    assert version in ['0101','0201','0301','0302'], "PWHAT?"
        
    if version == '0302':
        paf = 'New_baseline'
    else:
        paf = 'Old_baseline'
        assert 2<1,"Can't do old calibrations! FTP hasn't worked for some time now (20210224), and current implementation only downloads new files ..."
    # year = args[1]
    return BaseAddr + '/'.join([paf,sat,''])


def get_url_paths(url, ext='', params={}, debug=False):
    """
    Does nothing more than find out what filenames exist at a particular url
    """
    if debug:
        print("DEBUG: url = {:s}".format(url))
    response = requests.get(url, params=params, timeout=10)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()

    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [node.get('href') for node in soup.find_all('a') if node.get('href') is not None]
    if ext != '':
        parent = [thing for thing in parent if thing.endswith(ext)]


    return parent


def store_Swarm_CT2Hz_DF(fn,df):

    store = pd.HDFStore(fn)
    for column in df.columns:
        store.append(column, df[column], format='t')

    store.close()


def get_files_we_dont_have(fn,checkfiles):

    # Open master hdf
    # fn = localdir+masterhdf

    # Check if HDF exists!
    if not os.path.exists(fn):
        print(f"{fn} does not exist!")
        return checkfiles

    with pd.HDFStore(fn, mode='r') as hdf:
        keys = hdf.keys()


    indices = pd.DataFrame({key.replace('/', ''): pd.read_hdf(fn, key=key).values for key in [keys[0]]},
                           index=pd.read_hdf(fn, key=keys[0]).index).index

    # Dates in this HDF file
    # havedates = np.unique(indices.date)

    # Loop over provided filenames to see if they are already here
    wantfiles = []
    wantfiledict = dict()
    for fil in checkfiles:

        dtrange = getCT2HzFileDateRange(fil)
        havefile = ((indices >= dtrange[0]) & (indices <= dtrange[1])).sum() > 0
        # datestr = fil.split('_')[-2]
        # dt = datetime.strptime(datestr,"%Y%m%d").date()

        wantfiledict[fil] = havefile
        if not havefile:
            wantfiles.append(fil)

    print(f"Already had {len(checkfiles)-len(wantfiles)} of {len(checkfiles)} total files")

    # return wantfiles
    return wantfiles,wantfiledict

if VERSION == '0101':
    fullext = VERSION + '.CDF.ZIP'
else:
    fullext = VERSION + '.ZIP'

########################################
# Download!
for sat in sats:

    print(sat)

    # masterhdf = sat+f'_ct2hz_v{VERSION}.h5'

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    # print(masterhdf)

    # for year in years:
    curIterStr = f"{sat}-{VERSION}"

    ########################################
    ########################################
    # SINCE FTP DOESN'T WORK, TRY THIS

    # Read in manually snatched list of files

    print("20210224 Even my new wget-based method doesn't work. The FTP site seems to hate me now, and I don't know why.")
    print("20210802 But now wget works again?")

    swarmFTPAddr = "swarm-diss.eo.esa.int"
    subfolder = 'New_baseline' 

    sattie = sat.replace('Sat_','')
    flist = glob(filelistdir+f'{sattie}*.txt')

    assert len(flist) == 1

    with open(flist[0]) as f:
        flist = f.read().splitlines()

    todownload = []
    for f in flist:
        # print(f,end=' ')
        if os.path.exists(localdir+f):

            if os.path.getsize(localdir+f) < 900: # in bytes
                print(f"{f} is teensy-weensy! Removing and downloading again ...")
                os.remove(localdir+f)    
            else:
                # print("Skipping!")
                continue

        # print(f"Adding {f}")
        todownload.append(f)
    
    if len(todownload) == 0:
        print(f"Already have all files for {sat}. Continuing ...")
        continue

    cont = False
    while not cont:
        response = input(f"Going to download {len(todownload)} files. Sound OK? [y/n/(s)how me]")
        if len(response) == 0:
            continue
        response = response.lower()[0]
        cont = response in ['y','n']
        
        if not cont:
            if response == 's':
                [print(f) for f in todownload]
            else:
                print("Invalid. Say 'yes' or 'no'")

    if response == 'n':
        print("OK, skipping ...")
        continue

    print("Downloading!")

    subDir = f'/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/{subfolder:s}/Sat_{sattie:s}/'

    for ftpFile in todownload:
        print(f"Downloading {ftpFile}")
        # breakpoint()
        wget.download('ftp://'+swarmFTPAddr+subDir+ftpFile,localdir+ftpFile)
        # print(f"Would run this: wget.download('{'ftp://'+swarmFTPAddr+subDir+ftpFile,localdir+ftpFile}')")
    
