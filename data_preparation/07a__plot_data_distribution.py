import numpy as np
import pandas as pd
from pysymmetry.geodesy import geoc2geod

import matplotlib as mpl
# import tkinter
mplBckend = 'QtAgg'
mpl.use(mplBckend)
import sys

import matplotlib.pyplot as plt
plt.ion()


from directories import masterhdfdir
huberdir = masterhdfdir+'matrices/'
plotdir = '/home/plots/'

# data = pd.read_pickle('external_data_and_coordinates.pd')
# data = pd.concat((data,data,data))
DATAVERSION = 'v3'
DATAVERSION = 'v3.1'

# output = 'modeldata_v3.1_update.hdf5' # version created 20230607 with CT2Hz data through 2023/04, dropping data below min height and above max F10.7


MODELSUFF = 'analyticpotzero_at_47,-47deg'

prefix = 'model_v3FINAL_analyticpotzero_at_47,-47deg_v3'
iternum = 5

# kingpin, the one I think we will use
prefix = 'model_v3.1FULL_analyticpotzero_at_47,-47deg_v3.1'
iternum = 23

huberf = huberdir+f'{prefix}_huber_iteration_{iternum}.npy'
huber = np.load(huberf).flatten()

# satellites = ['SwarmA', 'SwarmB']
# satmap = {'CHAMP':1, 'SwarmA':2, 'SwarmB':3, 'SwarmC':4}
sats = ['Sat_A','Sat_B']
satmap = {'Sat_A':1, 'Sat_B':2}

VERSION = '0302'

hdfsuff = '_5sres_44qdlat'
# hdfsuff = '_5sres'
# hdfsuff = '_5sres_allmlat'

plotname = f'data_{DATAVERSION}_distribution__model={MODELSUFF}_i={iternum}.png'

OK = False

while not OK:
    svar = input(f"Gonna make\n'{plotname}'\nusing Huber weight file\n'{huberf}'.\nSound OK? (y/n)")

    if svar.lower().startswith('n'):
        sys.exit()
    elif svar.lower().startswith('y'):
        print("Grease fire!")
        OK = True
    else:
        print("Try again, Ricky!")

##############################
# Quality flag enforcement
##############################
# Do you want to only include measurements with a certain quality flag?
# For data version 0302, the second bit ('0100') being set means that v_(i,y) is calibrated
# See section 3.4.1.1 in "EFI TII Cross-Track Flow Data Release Notes" (Doc. no: SW-RN-UOC-GS-004, Rev: 7)
enforce_quality_flag = True     
quality_flag = '0100'

columns = ['mlat', 'mlt','QDLatitude','Radius','Latitude']
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

        if enforce_quality_flag:

            print(f"Dropping records that do not have Quality_flag == {quality_flag} ...")

            nNow = len(tmpdf)
            tmpdf = tmpdf[(store['/Quality_flags'].values & int(quality_flag,2)) > 0]
            nLater = len(tmpdf)
            
            print(f"Dropped {nNow-nLater} of {nNow} records ({(nNow-nLater)/nNow*100:.3f}%)")


        subsets[sat]  = tmpdf
        external[sat] = tmpextdf

# add external parameters to main df - and drop nans:
print ('adding external to main dataframe')
for sat in sats:
    subsets[sat][external[sat].columns] = external[sat]
    length = len(subsets[sat])
    subsets[sat] = subsets[sat].dropna()
    print ('dropped %s out of %s datapoints because of nans' % (length - len(subsets[sat]), length))

print ('merging the subsets')
full = pd.concat(subsets)
sat  = [k[0] for k in full.index]
time = [k[1] for k in full.index]
full['time'] = time
# full['time'] = full.index

full.index = range(len(full))

gdlat, height, _, _ = geoc2geod(full.Latitude.values, full.Radius.values/1000,
                                np.zeros_like(full.Radius.values), np.zeros_like(full.Radius.values))

full['h'] = height

if DATAVERSION.startswith('v3'):
    maxf107 = 300
    minh = 410

    keepi = (full[choosef107] <= maxf107) & (height >= minh)

    if DATAVERSION == 'v3.1':
        mintstamp = "2014-05-01 00:00:00"
        keepi = keepi & (full['time'] >= pd.Timestamp("2014-05-01 00:00:00"))
        print(f"Dropping {(~keepi).sum()} observations for which either F10.7 > {maxf107} or height < {minh} km or time < {mintstamp}, or some combo")
    else:
        print(f"Dropping {(~keepi).sum()} observations for which either F10.7 > {maxf107} or height < {minh} km, or both")

if DATAVERSION in ['v3','v3.1']:
    full = full[keepi]

    if huber.size == keepi.size:
        huber = huber[keepi]
    else:
        assert huber.size == full.shape[0],"Something is off! Your huber weights are the wrong size!"

full.index = range(len(full))

# assert 2<0

# sat_weights = data.s_weight.values
sat_weights = np.ones_like(full.Bz.values)

fig = plt.figure(figsize = (13, 13))

# set up the axes
#ax_cbar = plt.subplot2grid((9,  45), (0, 0), rowspan = 3)
ax_vB   = plt.subplot2grid((9,  15), (1, 0), rowspan = 2, colspan = 4)
ax_v    = plt.subplot2grid((9,  15), (0, 0), rowspan = 1, colspan = 4)
ax_B    = plt.subplot2grid((9,  15), (1, 4), rowspan = 2, colspan = 2)

ax_Byz  = plt.subplot2grid((9,  15), (1, 9), rowspan = 2, colspan = 4)
ax_By   = plt.subplot2grid((9,  15), (0, 9), rowspan = 1, colspan = 4)
ax_Bz   = plt.subplot2grid((9,  15), (1, 13), rowspan = 2, colspan = 2)

ax_f107 = plt.subplot2grid((9,  15), (4, 0 ), rowspan = 1, colspan = 6)
ax_tilt = plt.subplot2grid((9,  15), (4, 9), rowspan = 1, colspan = 6)

ax_mlat = plt.subplot2grid((9,  15), (6, 0), rowspan = 1, colspan = 6)
ax_mlt  = plt.subplot2grid((9,  15), (8, 0), rowspan = 1, colspan = 6)

ax_h    = plt.subplot2grid((9,  15), (6, 9), rowspan = 3, colspan = 6)

ax_vB.hist2d(full.vx.values, np.sqrt(full.Bz.values**2 + full.By.values**2), bins = (240, 240), range = ((-800, -200), (0, 15)), weights = None, density = True, cmap = plt.cm.Blues)
ax_vB.spines['right'].set_visible(False)
ax_vB.spines['top'].set_visible(False)
ax_vB.yaxis.set_ticks_position('left')
ax_vB.xaxis.set_ticks_position('bottom')
ax_vB.set_xlabel('$v_x$ [km/s]', size = 14)
ax_vB.set_ylabel('$\sqrt{B_y^2 + B_z^2}$ [nT]', size = 14)
ax_vB.yaxis.set_ticks([0, 3, 6, 9, 12, 15])


ax_v.hist(full.vx.values, bins = 240, range = (-800, -200), density = True, weights = sat_weights)
ax_v.hist(full.vx.values, bins = 240, range = (-800, -200), density = True, weights = sat_weights * huber, histtype = 'step', color = 'black')
ax_v.spines['right'].set_visible(False)
ax_v.spines['top'].set_visible(False)
ax_v.yaxis.set_ticks_position('left')
ax_v.xaxis.set_ticks_position('bottom')
ax_v.set_xlim(-800, -200)
ax_v.set_axis_off()

ax_B.hist(np.sqrt(full.Bz.values**2 + full.By.values**2), bins = 240, range = (0, 15), density = True, orientation = 'horizontal', weights = sat_weights)
ax_B.hist(np.sqrt(full.Bz.values**2 + full.By.values**2), bins = 240, range = (0, 15), density = True, orientation = 'horizontal', weights = sat_weights * huber, histtype = 'step', color = 'black')
ax_B.spines['right'].set_visible(False)
ax_B.spines['top'].set_visible(False)
ax_B.yaxis.set_ticks_position('left')
ax_B.xaxis.set_ticks_position('bottom')
ax_B.set_ylim(0, 15)
ax_B.set_axis_off()


ax_Byz.hist2d(full.By.values, full.Bz.values, bins = (240, 240), range = ((-10, 10), (-10, 10)), density = True, cmap = plt.cm.Blues, weights = sat_weights)
ax_Byz.spines['right'].set_visible(False)
ax_Byz.spines['top'].set_visible(False)
ax_Byz.yaxis.set_ticks_position('left')
ax_Byz.xaxis.set_ticks_position('bottom')
ax_Byz.set_xlabel('$B_y$ [nT]', size = 14)
ax_Byz.set_ylabel('$B_z$ [nT]', size = 14)
ax_Byz.xaxis.set_ticks([-10, -5, 0, 5, 10])
ax_Byz.yaxis.set_ticks([-10, -5, 0, 5, 10])


ax_By.hist(full.By.values, bins = 240, range = (-10, 10), density = True, weights = sat_weights)
ax_By.hist(full.By.values, bins = 240, range = (-10, 10), density = True, weights = sat_weights * huber, histtype = 'step', color = 'black')
ax_By.spines['right'].set_visible(False)
ax_By.spines['top'].set_visible(False)
ax_By.yaxis.set_ticks_position('left')
ax_By.xaxis.set_ticks_position('bottom')
ax_By.set_xlim(-10, 10)
ax_By.set_axis_off()

ax_Bz.hist(full.Bz.values, bins = 240, range = (-10, 10), density = True, orientation = 'horizontal', weights = sat_weights)
ax_Bz.hist(full.Bz.values, bins = 240, range = (-10, 10), density = True, orientation = 'horizontal', weights = sat_weights * huber, histtype = 'step', color = 'black')
ax_Bz.spines['right'].set_visible(False)
ax_Bz.spines['top'].set_visible(False)
ax_Bz.yaxis.set_ticks_position('left')
ax_Bz.xaxis.set_ticks_position('bottom')
ax_Bz.set_ylim(-10, 10)
ax_Bz.set_axis_off()


showf107i = full.f107obs.values <= 300
showf107 = full.f107obs.values[showf107i]
ax_f107.hist(showf107, bins = 240, density = True, weights = sat_weights[showf107i])
ax_f107.hist(showf107, bins = 240, density = True, weights = (sat_weights * huber)[showf107i], histtype = 'step', color = 'black')
ax_f107.spines['top'].set_visible(False)
ax_f107.spines['right'].set_visible(False)
ax_f107.spines['left'].set_visible(False)
ax_f107.axes.get_yaxis().set_visible(False)
ax_f107.xaxis.set_ticks_position('bottom')
ax_f107.set_xlabel('F10.7 [sfu]', size = 14)

ax_tilt.hist(full.tilt.values, bins = 240, density = True, weights = sat_weights)
ax_tilt.hist(full.tilt.values, bins = 240, density = True, weights = sat_weights * huber, histtype = 'step', color = 'black')
ax_tilt.spines['top'].set_visible(False)
ax_tilt.spines['right'].set_visible(False)
ax_tilt.spines['left'].set_visible(False)
ax_tilt.axes.get_yaxis().set_visible(False)
ax_tilt.xaxis.set_ticks_position('bottom')
ax_tilt.set_xlabel('Dipole tilt angle [deg]', size = 14)
ax_tilt.set_xlim(-35, 35)


ax_mlat.hist(full.QDLatitude.values, bins = 240, density = True, label = 'QD', weights = sat_weights)
ax_mlat.hist(full.mlat.values, bins = 240, density = True, alpha = 0.5, label = 'MA', weights = sat_weights)
ax_mlat.spines['top'].set_visible(False)
ax_mlat.spines['right'].set_visible(False)
ax_mlat.spines['left'].set_visible(False)
ax_mlat.axes.get_yaxis().set_visible(False)
ax_mlat.xaxis.set_ticks_position('bottom')
ax_mlat.set_xlabel('Magnetic latitude [deg]', size = 14)
ax_mlat.set_xlim(-90, 90)
ax_mlat.xaxis.set_ticks([-90, -45, 0, 45, 90])
ax_mlat.legend(frameon = False, loc = 1, bbox_to_anchor=(0.9,1))

ax_mlt.hist(full.mlt.values, bins = 240, density = True, weights = sat_weights)
ax_mlt.spines['top'].set_visible(False)
ax_mlt.spines['right'].set_visible(False)
ax_mlt.spines['left'].set_visible(False)
ax_mlt.axes.get_yaxis().set_visible(False)
ax_mlt.xaxis.set_ticks_position('bottom')
ax_mlt.set_xlabel('Magnetic local time [h]', size = 14)
ax_mlt.set_xlim(0, 24)
ax_mlt.xaxis.set_ticks([0, 6, 12, 18, 24])

showhi = full.h.values >= 410
ax_h.hist(full.h.values[showhi], orientation = 'horizontal', bins = 200, density = True, weights = sat_weights[showhi])
ax_h.spines['right'].set_visible(False)
ax_h.yaxis.set_ticks_position('left')
ax_h.spines['top'].set_visible(False)
ax_h.spines['bottom'].set_visible(False)
ax_h.axes.get_xaxis().set_visible(False)
ax_h.set_ylabel('Height [km]', size = 14)

plt.subplots_adjust(hspace = 0, wspace = 0, top = .99, right = .99, bottom = .06, left = .05)


plt.savefig(plotdir+plotname, dpi = 300)
# plt.savefig('data_distribution.svg')
# plt.savefig('data_distribution.pdf')

