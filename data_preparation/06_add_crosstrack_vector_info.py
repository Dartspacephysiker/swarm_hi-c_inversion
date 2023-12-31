import geodesy
import pandas as pd
import numpy as np
from Swarm_TII import calculate_satellite_frame_vectors_in_NEC_coordinates,calculate_crosstrack_flow_in_NEC_coordinates
import ppigrf
from dateutil.relativedelta import relativedelta
from datetime import timedelta


maxtdiff_between_wanttime_and_omni = pd.Timedelta('2 min')

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']
# sats = ['Sat_B','Sat_C']
VERSION = '0302'
hdfsuff = '_5sres_44qdlat'
# hdfsuff = '_5sres'
# hdfsuff = '_5sres_allmlat'
# hdfsuff = '_1sres'
# hdfsuff = '_Anna'
# hdfsuff = '_Anna2'
# hdfsuff = '_2014'

from directories import masterhdfdir
from directories import filepath as datapath

OMNIPERIOD = '20Min'
y1OMNI = '2013'
y2OMNI = '2024'

getcols = ['Bx','By','Bz',
           'Quality_flags',
           'Latitude','Longitude','Radius',
           'Viy',
           'mlat','mlon','mlt',
           'VsatN','VsatE','VsatC',
           'd10','d11','d12',
           'd20','d21','d22',
           'e10','e11','e12',
           'e20','e21','e22',
           'f10','f11',
           'f20','f21']


########################################    
# Loop over satellites

for sat in sats:
    # print(sat)

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    # Make df with getcols
    df = pd.DataFrame()

    # Check if sorted
    # with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
    #     print("HDF is monotonic: ",store['/mlat'].index.is_monotonic)
    #     if not store['/mlat'].index.is_monotonic:
    #         print("Sorting index of each key")
    #         keys = store.keys()
    #         for key in keys:
    #             print(key[1:]+' -> '+key,end=',')
    #             store.append(key[1:],store[key].sort_index(),format='t',append=False)
    #         print("")

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

        for wantcol in getcols:
            df[wantcol] = store['/'+wantcol]

    ##############################
    # Get convection in geodetic and Apex coordinates
    # Bonus: get altitude and geodetic latitude
    print("Getting crosstrack vector in NEC coordinates ...")
    Viy_NEC = calculate_crosstrack_flow_in_NEC_coordinates(df)
    
    df['Viy_N'] = Viy_NEC[0,:]
    df['Viy_E'] = Viy_NEC[1,:]
    df['Viy_C'] = Viy_NEC[2,:]
    
    # Convert 'Viy_N' (and 'Viy_C') from geocentric to geodetic coordinates
    print("Getting convection meas in geodetic coordinates ...")
    gdlat, alt, Viy_NGeod, Viy_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
                                                         df["Radius"].values/1000.,
                                                         -df['Viy_N'],df['Viy_C'])
    

    # get satellite yhat vector in ENU coords
    xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])

    print("Converting yhat to geodetic coordinates ...")
    # print("You've been using -yhat[1,:] as input for theta component in geodesy.geoc2geod, but that seems wrong! Check it!!")
    # gdlat, alt, yhat_NGeod, yhat_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
    #                                                      df["Radius"].values/1000.,
    #                                                        -yhat[1,:],yhat[2,:])

    gdlat, alt, yhat_NGeod, yhat_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
                                                         df["Radius"].values/1000.,
                                                           -yhat[0,:],yhat[2,:])  # Think this is right
    df['gdlat'] = gdlat
    df['alt'] = alt
    
    # print("Check magnitude of yhat_ENU")
    # print(yhat[1,:]**2+yhat_NGeod**2+((-1)*yhat_DGeod)**2)

    print("Getting B_IGRF ...")
    date0 = df.index[0].floor('1 d').to_pydatetime()
    date1 = df.index[-1].ceil('1 d').to_pydatetime()
    reldelt = relativedelta(months=3)

    curdate = date0
    totinds = 0
    df['B0IGRF'] = np.nan
    df['BeIGRF'] = np.nan
    df['BnIGRF'] = np.nan
    df['BuIGRF'] = np.nan
    while curdate <= date1:
        # inds = slice(curdate,curdate+reldelt)#df['2013-12-10':'2014-03-10']
        tmpdate = curdate+reldelt-relativedelta(days=1)
        inds = slice("{:04d}-{:02d}-{:02d}".format(curdate.year,curdate.month,curdate.day),
                     "{:04d}-{:02d}-{:02d}".format(tmpdate.year,tmpdate.month,tmpdate.day))
        print(inds)
        # Br, Btheta, Bphi = ppigrf.igrf_gc(df[inds]['Radius']/1000,
        #                                   90-df[inds]['Latitude'],
        #                                   df[inds]['Longitude'],curdate+reldelt/2)

        # df.loc[inds,'B0IGRF'] = np.sqrt(Br**2+Btheta**2+Bphi**2)

        Be, Bn, Bu = ppigrf.igrf(df[inds]['Longitude'],
                                 df[inds]['gdlat'],
                                 df[inds]['alt'],
                                 curdate+reldelt/2)
        df.loc[inds,'B0IGRF'] = np.sqrt(Be**2+Bn**2+Bu**2).ravel()
        df.loc[inds,'BeIGRF'] = Be.ravel()
        df.loc[inds,'BnIGRF'] = Bn.ravel()
        df.loc[inds,'BuIGRF'] = Bu.ravel()

        totinds += len(df[inds])
        print("totinds: ",totinds)
        curdate += reldelt
    
    assert totinds == len(df),"Something's screwy, probably have to make index monotonic back in script 01_make_swarm_CT2Hz_hdfs.py ..."

    print("Calculating line-of-sight (y-hat) vector with along-B component removed")
    # Now get line-of-sight (y hat) vector with along-B component removed
    Behat = (df['BeIGRF']/df['B0IGRF']).values
    Bnhat = (df['BnIGRF']/df['B0IGRF']).values
    Buhat = (df['BuIGRF']/df['B0IGRF']).values
    bhat = np.vstack([Behat,Bnhat,Buhat])  # bhat in ENU coords
    
    # line-of-sight vector and los vector with component along B0_IGRF removed
    l = np.vstack([yhat[1,:],yhat_NGeod,(-1)*yhat_DGeod])  # ENU coords
    lperptoB = l - np.sum(l*bhat,axis=0)*bhat
    lperptoB = lperptoB/np.sqrt(np.sum(lperptoB**2,axis=0))

    # y hat with no along-B0 component
    df['lperptoB_E'] = lperptoB[0,:]
    df['lperptoB_N'] = lperptoB[1,:]
    df['lperptoB_U'] = lperptoB[2,:]

    print("Calculating line-of-sight convection vector with along-B component removed")
    vENU = np.vstack([df['Viy_E'].values,Viy_NGeod,(-1)*Viy_DGeod])  # ENU coords
    vEperptoB = vENU - np.sum(vENU*bhat,axis=0)*bhat

    # fraction of measured vector that is parallel to B
    # assert 2<0,"DAD!"
    # vnorm = vENU/np.linalg.norm(vENU,axis=0)
    # vfracpartob = np.abs(np.sum(vENU*bhat,axis=0))/np.linalg.norm(vENU,axis=0)

    df['ViyperptoB_E'] = vEperptoB[0,:]
    df['ViyperptoB_N'] = vEperptoB[1,:]
    df['ViyperptoB_U'] = vEperptoB[2,:]

    # Now Apex coords
    # REMEMBER (from Richmond, 1995):
    # • d1 base vector points "more or less in the magnetic eastward direction"
    # • d2 base vector points "generally downward and/or equatorward (i.e., southward in NH [and in SH?])" 
    print("Calculating Viy_d1, Viy_d2, yhat_d1, yhat_d2 ...")
    df['Viy_d1'] = df['d10']*df['Viy_E'] + df['d11']*Viy_NGeod + df['d12']*Viy_DGeod
    df['Viy_d2'] = df['d20']*df['Viy_E'] + df['d21']*Viy_NGeod + df['d22']*Viy_DGeod
    
    print("Calculating Viy_d1, Viy_d2, yhat_d1, yhat_d2 ...")
    df['Viy_f1'] = df['f10']*df['Viy_E'] + df['f11']*Viy_NGeod
    df['Viy_f2'] = df['f20']*df['Viy_E'] + df['f21']*Viy_NGeod
    
    # Also get satellite yhat vector in Apex coordinates
    print("Calculating l.e1 and l.e2 ('l' stands for 'los')")
    # 'l' stands for 'los'
    l_dot_e1 = df['e10']*l[0,:] + df['e11']*l[1,:] + df['e12']*l[2,:]
    l_dot_e2 = df['e20']*l[0,:] + df['e21']*l[1,:] + df['e22']*l[2,:]

    lperptoB_dot_e1 = df['e10']*lperptoB[0,:] + df['e11']*lperptoB[1,:] + df['e12']*lperptoB[2,:]
    lperptoB_dot_e2 = df['e20']*lperptoB[0,:] + df['e21']*lperptoB[1,:] + df['e22']*lperptoB[2,:]

    # A little diagnostic to see how different these dot products are
    # weird = (np.abs(l_dot_e1-lperptoB_dot_e1)/((l_dot_e1+lperptoB_dot_e1)/2)*100) > 10

    # d vectors
    # yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat[0,:] + df['d12']*yhat[2,:]
    # yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat[0,:] + df['d22']*yhat[2,:]
    yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat_NGeod + df['d12']*(-1)*yhat_DGeod
    yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat_NGeod + df['d22']*(-1)*yhat_DGeod
    
    # f vectors
    # yhat_f1 = df['f10']*yhat[1,:] + df['f11']*yhat[0,:]
    # yhat_f2 = df['f20']*yhat[1,:] + df['f21']*yhat[0,:]
    yhat_f1 = df['f10']*yhat[1,:] + df['f11']*yhat_NGeod
    yhat_f2 = df['f20']*yhat[1,:] + df['f21']*yhat_NGeod
    yhatfmag = np.sqrt(yhat_f1**2+yhat_f2**2)
    yhat_f1 /= yhatfmag
    yhat_f2 /= yhatfmag

    # 'sign 'em all
    df['l_dot_e1'] = l_dot_e1
    df['l_dot_e2'] = l_dot_e2

    df['lperptoB_dot_e1'] = lperptoB_dot_e1
    df['lperptoB_dot_e2'] = lperptoB_dot_e2

    df['yhat_d1'] = yhat_d1
    df['yhat_d2'] = yhat_d2
    
    df['yhat_f1'] = yhat_f1
    df['yhat_f2'] = yhat_f2
    
    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        # storecols = ['Viy_d1','Viy_d2', 'yhat_d1', 'yhat_d2', 'gdlat', 'alt']
        storecols = ['Viy_d1','Viy_d2',
                     'Viy_f1','Viy_f2',
                     'yhat_d1', 'yhat_d2',
                     'yhat_f1', 'yhat_f2',
                     'gdlat', 'alt',
                     'BeIGRF','BnIGRF','BuIGRF',
                     'B0IGRF',
                     'l_dot_e1','l_dot_e2',
                     'lperptoB_dot_e1','lperptoB_dot_e2',
                     'ViyperptoB_E',
                     'ViyperptoB_N',
                     'ViyperptoB_U',
                     'lperptoB_E',
                     'lperptoB_N',
                     'lperptoB_U']

        # print(f"Storing {', '.join(storecols)} for {sat} ...")
        print(f"Storing ",end='')
        for column in storecols:
            print(column,end=', ')
            store.append(column, df[column], format='t', append=False)
