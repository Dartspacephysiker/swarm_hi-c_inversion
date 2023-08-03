#!/usr/bin/env python
# coding: utf-8

# Script for checking validity of Swarm CT 2-Hz zipfiles from Swarm FTP site
#
# Spencer Mark Hatch
# University of Bergen
# 2023-06-01

########################################
# Imports

import os
import zipfile
from glob import glob
filelistdir = '../filelists/'

swarmFTPAddr = "swarm-diss.eo.esa.int"
subfolder = 'New_baseline' 
subfolder = '' 

from directories import DownloadDir
wantext = '.ZIP'
VERSION = '0302'

########################################
# Define which satellites we'll look for during which years 

sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']
# y0,y1 = 2013,2021
# years = [str(val) for val in np.arange(y0,y1)]

########################################
# Download!
for sat in sats:

    print(sat)

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    ########################################
    ########################################
    # SINCE FTPLIB DOESN'T WORK, TRY THIS

    # Read in manually snatched list of files

    sattie = sat.replace('Sat_','')
    flist = glob(filelistdir+f'{sattie}*.txt')

    assert len(flist) == 1

    with open(flist[0]) as f:
        flist = f.read().splitlines()

    todownload = []
    have = []
    for f in flist:
        # print(f,end=' ')
        if os.path.exists(localdir+f):

            if os.path.getsize(localdir+f) < 900: # in bytes
                print(f"{f} is teensy-weensy! Removing and downloading again ...")
                os.remove(localdir+f)    
            else:
                # print("Skipping!")
                have.append(f)
                continue

        # print(f"Adding {f}")
        todownload.append(f)
    
    if len(todownload) != 0:
        print(f"Missing {len(todownload)} files for {sat}! You should download them ...")

    cont = False
    while not cont:
        response = input(f"Going to check {len(have)} zipfiles for validity. Sound OK? [y/n/(s)how me]")
        if len(response) == 0:
            continue
        response = response.lower()[0]
        cont = response in ['y','n']
        
        if not cont:
            if response == 's':
                [print(f) for f in have]
            else:
                print("Invalid. Say 'yes' or 'no' (or 's') [It rhymes!]")

    if response == 'n':
        print("OK, skipping ...")
        continue

    print("Checking!")

    for zipf in have:
        print(f"Checking {zipf}")
        try:
            the_zip_file = zipfile.ZipFile(localdir+zipf)
            ret = the_zip_file.testzip()
            if ret is not None:
                print("First bad file in zip: {:s}".format(ret))
        except Exception as ex:
            print("Exception:", ex)
    
