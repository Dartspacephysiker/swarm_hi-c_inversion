import numpy as np
import pandas as pd
import ftplib
import fnmatch
import os.path
from datetime import datetime
import time
import zipfile

def isSorted(x, key=lambda x: x):
    """
    Just another way to do it
    """
    return all([key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1)])

def toYearFraction(date):
    """
    From https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years
    """
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def getCT2HzFileDateRange(fName):
    """
    Should begin with 'SW_EXPT_EFIA_TIICT_'
    """
    yr0, mo0, day0 = int(fName[19:23]), int(fName[23:25]), int(fName[25:27])
    hr0, min0, sec0 = int(fName[28:30]), int(fName[30:32]), int(fName[32:34])
    yr1, mo1, day1 = int(fName[35:39]), int(fName[39:41]), int(fName[41:43])
    hr1, min1, sec1 = int(fName[44:46]), int(fName[46:48]), int(fName[48:50])
    return [datetime(yr0, mo0, day0, hr0, min0, sec0),
            datetime(yr1, mo1, day1, hr1, min1, sec1)]


def getCT2HzFTP(sat='A',
                dates=None,
                localSaveDir='/media/spencerh/data/Swarm/',
                calversion='0302',
                only_list=False,
                check_for_existing_func=None,
                append_dir=False):
    """
    Get a cross-track 2-Hz file
    """

    VALID_CAL_VERSIONS = ['0302','0301','0201','0101']

    fsuffdict = {'0101':'_0101.CDF.ZIP',
                 '0201':'_0201.ZIP',
                 '0301':'_0301.ZIP',
                 '0302':'_0302.ZIP'}

    assert calversion in VALID_CAL_VERSIONS
    if calversion == '0302':
        subfolder = 'New_baseline' 
        # elif calversion == '0101':
    else:
        subfolder = 'Old_baseline' 

    fsuff = fsuffdict[calversion]

    localSaveDir += '2Hz_TII_Cross-track/Swarm_'+sat+'/'

    swarmFTPAddr = "swarm-diss.eo.esa.int"

    subDir = f'/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/{subfolder:s}/Sat_{sat:s}/'


    # EXAMPLE: SW_EXPT_EFIA_TIICT_20151101T155814_20151101T235004_0101.CDF.ZIP
    # ftpFilePref = 'SW_EXPT_EFI'+sat+'_TIICT_'

    breakpoint()

    gotFiles = _getFTP_dateGlob(dates, localSaveDir, subDir,
                                only_list=only_list)

    gotFiles = [gotFile for gotFile in gotFiles if fsuff in gotFile]

    if append_dir:
        if gotFiles is not None:
            gotFiles = [localSaveDir+gotFile for gotFile in gotFiles]

    return gotFiles


def _getFTP_dateGlob(dates, localSaveDir, subDir,
                     only_list=False):
    """
    Get a Swarm FTP file, genericizliaed
    """

    # WGET VERSION
    import wget

    swarmFTPAddr = "swarm-diss.eo.esa.int"

    ftp = ftplib.FTP(swarmFTPAddr)
    ftp.login()                 # Anonymous
    ftp.cwd(subDir)

    filz = ftp.nlst(subDir)
    ftp.close()                 # TRY THIS

    ftpFiles = []

    if isinstance(dates, str):
        dates = [dates]
    elif isinstance(dates, list):

        for date in dates:
            if not isinstance(date, str):
                print("Must provide date strings 'YYYYMMDD'!")
                return None

    else:

        if only_list:
            ftp.close()
            return filz
        else:
            assert isinstance(
                dates, list), "Must provide list of date strings or a single date string (YYYYMMDD format)! (Or set kw only_list == True)"

    # Pick up all the files that match provided dates
    for date in dates:

        for f in filz:
            if fnmatch.fnmatch(f, '*'+date+'*'):
                # print(f)
                ftpFiles.append(f)
                # break

    if only_list:
        ftp.close()
        return ftpFiles

    # If no files found, exit
    if len(ftpFiles) == 0:
        print("Found no file! Exiting ...")
        ftp.close()
        return None

    # Junk the already-havers
    # ftpNotHavers = [ftpFile in ftpFiles if not os.path.isfile(localSaveDir + ftpFile)]
    ftpNotHavers = []
    for ftpFile in ftpFiles:
        fileExists = os.path.isfile(localSaveDir + ftpFile)

        if fileExists:
            if os.stat(localSaveDir+ftpFile).st_size == 0:
                print("{:s}:  File size is zero! Trying to get på nytt".
                      format(ftpFile))
                os.remove(localSaveDir+ftpFile)
        elif not fileExists:
            ftpNotHavers.append(ftpFile)


    if len(ftpNotHavers) == 0:
        print("Already have all {:d} files for {:d} date(s) provided! Exiting ...".format(
            len(ftpFiles), len(dates)))
        ftp.close()
        return ftpFiles

    print("Found {:d} files for the {:d} date(s) provided ({:d} are already downloaded)".format(
        len(ftpFiles), len(dates), len(ftpFiles)-len(ftpNotHavers)))

    # Get all matching files
    for ftpFile in ftpNotHavers:

        # Make sure we don't already have file
        if not os.path.isfile(localSaveDir + ftpFile):

            if not os.path.isdir(localSaveDir):
                os.mkdir(localSaveDir)

            print("Trying to get " + ftpFile + ' ...',end='')
            try:
                wget.download('ftp://'+swarmFTPAddr+subDir+ftpFile,localSaveDir+ftpFile)
                print("Done!")
            except:
                print("Couldn't get "+ftpFile+"!")

            # with open(localSaveDir+ftpFile, "wb") as getFile:
            #     print("Trying to get " + ftpFile + ' ...')
            #     try:
            #         ftp.retrbinary("RETR " + ftpFile, getFile.write)
            #         print("Done!")
            #     except:
            #         print("Couldn't get "+ftpFile+"!")

        else:
            print("Already have " + ftpFile + '!')

    ftp.close()

    return ftpFiles


def processSwarm2HzCrossTrackZip(zipDir, CTZip, dataDir,
                                 doResample=False,
                                 resampleString="62500000ns",
                                 skipEphem=True,
                                 quiet=False,
                                 removeCDF=False,
                                 rmCDF_noPrompt=False,
                                 dont_touch_data=False,
                                 include_explicit_calibration_flags=False):

    from spacepy import pycdf

    dataInd = 0
    CTFile = funker(zipDir, CTZip, dataDir, dataInd,
                    quiet=quiet)
    if CTFile is None:
        return None

    if not quiet:
        print(CTFile)

    cdf = pycdf.CDF(dataDir+CTFile)

    if removeCDF:
        junk = funker(zipDir, CTZip, dataDir, dataInd,
                      remove=True,
                      rm_noprompt=rmCDF_noPrompt,
                      quiet=quiet)

    # return cdf

    if not quiet:
        print("Getting tStamps ...", end='')

    calversion = CTZip[-8:-4]
    if calversion == '0101':
        tStamps = cdf["timestamp"][:]
    elif calversion == '0201':
        tStamps = cdf["Time"][:]
    elif calversion in ['0301','0302']:
        tStamps = cdf["Timestamp"][:]
    else:
        print(f"Not equipped to handle calversion == {calversion}! Exiting")
        return None

    if isinstance(tStamps[0], np.float64):
        try:
            OFFSET = datetime(2000, 1, 1, 0) - \
                datetime(1970, 1, 1)
            tStamps = np.array([datetime.utcfromtimestamp(
                tStamp)+OFFSET for tStamp in tStamps])
        except:
            print("No to timestamps!")
            return None

    tStamps = pd.Series(list(map(pd.Timestamp, tStamps)))

    if include_explicit_calibration_flags and (calversion != '0301'):
        print("Have only implemented explicit calibration flags for cross-track data v0301!")

    try:
        if calversion == '0101':
            df = pd.DataFrame(data={'latitude': cdf['latitude'][:],
                                    'longitude': cdf['longitude'][:],
                                    'radius': cdf['radius'][:],
                                    'qdlat': cdf['qdlat'][:],
                                    'mlt': cdf['mlt'][:],
                                    'viy': cdf['viy'][:],
                                    'viz': cdf['viz'][:],
                                    'vsatnorth': cdf['vsatnorth'][:],
                                    'vsateast': cdf['vsateast'][:],
                                    'vsatcentre': cdf['vsatcentre'][:],
                                    'vcorotation': cdf['vcorotation'][:],
                                    'angleH': cdf['angleH'][:],
                                    'angleV': cdf['angleV'][:],
                                    'offset1h': cdf['offset1h'][:],
                                    'offset1v': cdf['offset1v'][:],
                                    'offset2h': cdf['offset2h'][:],
                                    'offset2v': cdf['offset2v'][:],
                                    'ex': cdf['ex'][:],
                                    'ey': cdf['ey'][:],
                                    'ez': cdf['ez'][:],
                                    'bx': cdf['bx'][:],
                                    'by': cdf['by'][:],
                                    'bz': cdf['bz'][:],
                                    'qy': cdf['qy'][:],
                                    'qz': cdf['qz'][:],
                                    'qe': cdf['qe'][:]},
                              index=tStamps)

        elif calversion == '0201':
            df = pd.DataFrame(data={'Bx':cdf['Bx'][:],
                                    'By':cdf['By'][:],
                                    'Bz':cdf['Bz'][:],
                                    'Ehx':cdf['Ehx'][:],
                                    'Ehy':cdf['Ehy'][:],
                                    'Ehz':cdf['Ehz'][:],
                                    'Evx':cdf['Evx'][:],
                                    'Evy':cdf['Evy'][:],
                                    'Evz':cdf['Evz'][:],
                                    'Latitude':cdf['Latitude'][:],
                                    'Longitude':cdf['Longitude'][:],
                                    'MLT':cdf['MLT'][:],
                                    'QDLatitude':cdf['QDLatitude'][:],
                                    'Radius':cdf['Radius'][:],
                                    # 'Time':cdf['Time'][:],
                                    'Vicrx':cdf['Vicrx'][:],
                                    'Vicry':cdf['Vicry'][:],
                                    'Vicrz':cdf['Vicrz'][:],
                                    'Vixh_experimental':cdf['Vixh_experimental'][:],
                                    'Vixv_experimental':cdf['Vixv_experimental'][:],
                                    'Viy':cdf['Viy'][:],
                                    'Viz_experimental':cdf['Viz_experimental'][:],
                                    'VsatC':cdf['VsatC'][:],
                                    'VsatE':cdf['VsatE'][:],
                                    'VsatN':cdf['VsatN'][:],
                                    'Vsatx':cdf['Vsatx'][:],
                                    'Vsaty':cdf['Vsaty'][:],
                                    'Vsatz':cdf['Vsatz'][:],
                                    'flags':cdf['flags'][:]},
                              index=tStamps)

        elif calversion in ['0301','0302']:
            # REF 1 --- Doc. no: SW-RN-UOC-GS-004, Rev: 6
            df = pd.DataFrame(data={'Bx':cdf['Bx'][:],
                                    'By':cdf['By'][:],
                                    'Bz':cdf['Bz'][:],
                                    'Ehx':cdf['Ehx'][:],
                                    'Ehy':cdf['Ehy'][:],
                                    'Ehz':cdf['Ehz'][:],
                                    'Evx':cdf['Evx'][:],
                                    'Evy':cdf['Evy'][:],
                                    'Evz':cdf['Evz'][:],
                                    'Latitude':cdf['Latitude'][:],
                                    'Longitude':cdf['Longitude'][:],
                                    'MLT':cdf['MLT'][:],
                                    'QDLatitude':cdf['QDLatitude'][:],
                                    'Radius':cdf['Radius'][:],
                                    'Quality_flags':cdf['Quality_flags'][:],
                                    # 'Time':cdf['Time'][:],
                                    'Vicrx':cdf['Vicrx'][:],
                                    'Vicry':cdf['Vicry'][:],
                                    'Vicrz':cdf['Vicrz'][:],
                                    'Vixh':cdf['Vixh'][:],
                                    'Vixh_error':cdf['Vixh_error'][:],
                                    'Vixv':cdf['Vixv'][:],
                                    'Vixv_error':cdf['Vixv_error'][:],
                                    'Viy':cdf['Viy'][:],
                                    'Viy_error':cdf['Viy_error'][:],
                                    'Viz':cdf['Viz'][:],
                                    'Viz_error':cdf['Viz_error'][:],
                                    'VsatC':cdf['VsatC'][:],
                                    'VsatE':cdf['VsatE'][:],
                                    'VsatN':cdf['VsatN'][:]},
                              index=tStamps)

            ##############################
            # Convert "no error estimate" rows to NaNs  (REF 1 p. 17)
            # 
            # For 'Vixh_error', 'Vixv_error', 'Viy_error', 'Viz_error': "For Negative value indicates no estimate available."
            
            # As of pandas v0.25.3, the nan-replacer line is safe even in the row matcher (i.e., df['Vixh_error'] < 0) matches no rows
            errcols = ['Vixh_error', 'Vixv_error', 'Viy_error', 'Viz_error']
            for errcol in errcols:
                df.loc[df[errcol] < 0,errcol] = np.nan


            if include_explicit_calibration_flags:

                if not quiet:
                    print("Including explicit calibration flags ...")

                ##############################
                # Interpret quality flags (REF 1 p. 19)
                
                # "Bitwise flag for each velocity component, where a value of 1 for a particular component signifies that calibration was successful,
                # and that the baseline 1-sigma noise level is less than or equal to 100 m/s at 2 Hz.
                # Electric field quality can be assessed from these flags according to -vxB."
                
                # Bit0 (least significant) = Vixh, bit1= Vixv, bit2 = Viy, bit3 = Viz
                df['Vixh_calibrated'] = df['Quality_flags'].values & int("0001",base=2)
                df['Vixv_calibrated'] = (df['Quality_flags'].values & int("0010",base=2)) >> 1
                df['Viy_calibrated'] = (df['Quality_flags'].values & int("0100",base=2)) >> 2
                df['Viz_calibrated'] = (df['Quality_flags'].values & int("1000",base=2)) >> 3

    except:
        print("Problems processing Swarm 2-Hz cross-track stuff!")
        return None

    return df


def funker(dir, file, outDir, filNummer=None, remove=False, quiet=False,
           rm_noprompt=False):

    if not os.path.isdir(dir):
        # if not os.path.isfile(dir+file):
        print("Not have: " + file + ' !! Try FTP...')
        print("But not implemented!")

    else:
        # print("Yeah: " + file)
        if not os.path.isfile(dir+file):
            print("Doesn't exist! Returning ...")
            return None
        try:
            zip_ref = zipfile.ZipFile(dir+file, 'r')
        except:
            print("Trouble with zip file! Headed out ...")
            return None

        if filNummer is None:
            print("Hvilken vil du ha?")
            print("=======")
            for dude in enumerate(zip_ref.namelist()):
                print("{0}: {1}".format(dude[0], dude[1]))

            filNummer = int(input("Enter a number..."))

        want = zip_ref.namelist()[filNummer]

        if remove:

            if rm_noprompt:
                svar = 'Y'
            else:
                svar = input("Remove " + outDir + want + '? (y/n)')

            if svar.upper()[0] == 'Y':
                os.remove(outDir+want)
                if not quiet:
                    print("Removed " + outDir + want + '!')
        else:
            if not quiet:
                print(want)

            if os.path.exists(outDir+want):
                print(outDir+want+' exists! Should junk ...')
                os.remove(outDir+want)
                if not quiet:
                    print("Removed " + outDir + want + '!')

            try:
                zip_ref.extract(want, outDir)
            except:
                print("Couldn't extract " + want + '! Returning ...')
                zip_ref.close()
                return None

        zip_ref.close()
        return want


def geoclatR2geodlatheight(glat, r_km):
    """
    glat : geocentric latitude (degrees)
    r_km : radius (km)
    Returns gdlat, gdalt_km
    """
    d2r = np.pi/180
    r2d = 180 / np.pi
    WGS84_e2 = 0.00669437999014
    WGS84_a = 6378.137

    a = WGS84_a
    b = a*np.sqrt(1 - WGS84_e2)

    E2 = 1.-(b/a)**2
    E4 = E2*E2
    E6 = E4*E2
    E8 = E4*E4
    A21 = (512.*E2 + 128.*E4 + 60.*E6 + 35.*E8)/1024.
    A22 = (E6 + E8) / 32.
    A23 = -3.*(4.*E6 + 3.*E8) / 256.
    A41 = -(64.*E4 + 48.*E6 + 35.*E8)/1024.
    A42 = (4.*E4 + 2.*E6 + E8) / 16.
    A43 = 15.*E8 / 256.
    A44 = -E8 / 16.
    A61 = 3.*(4.*E6 + 5.*E8)/1024.
    A62 = -3.*(E6 + E8) / 32.
    A63 = 35.*(4.*E6 + 3.*E8) / 768.
    A81 = -5.*E8 / 2048.
    A82 = 64.*E8 / 2048.
    A83 = -252.*E8 / 2048.
    A84 = 320.*E8 / 2048.

    SCL = np.sin(glat * d2r)

    RI = a/r_km
    A2 = RI*(A21 + RI * (A22 + RI * A23))
    A4 = RI*(A41 + RI * (A42 + RI*(A43+RI*A44)))
    A6 = RI*(A61 + RI * (A62 + RI * A63))
    A8 = RI*(A81 + RI * (A82 + RI*(A83+RI*A84)))

    CCL = np.sqrt(1-SCL**2)
    S2CL = 2.*SCL * CCL
    C2CL = 2.*CCL * CCL-1.
    S4CL = 2.*S2CL * C2CL
    C4CL = 2.*C2CL * C2CL-1.
    S8CL = 2.*S4CL * C4CL
    S6CL = S2CL * C4CL + C2CL * S4CL

    DLTCL = S2CL * A2 + S4CL * A4 + S6CL * A6 + S8CL * A8
    gdlat = DLTCL + glat * d2r
    gdalt_km = r_km * np.cos(DLTCL) - a * np.sqrt(1 - E2 * np.sin(gdlat) ** 2)

    gdlat = gdlat / d2r

    return gdlat, gdalt_km


def geodetic2apex(*args,
                  apexRefTime=datetime(2012, 1, 1),
                  apexRefHeight_km=110,
                  quiet=False,
                  nancheck=False,
                  max_N_months_twixt_apexRefTime_and_obs=3,
                  min_time_resolution__sec=1,
                  interpolateArgs={'method': 'time', 'limit': 21},
                  do_qdcoords=False,
                  returnPandas=True,
                  return_apex_d_basevecs=False,
                  return_apex_e_basevecs=False,
                  return_apex_f_basevecs=False,
                  return_apex_g_basevecs=False,
                  return_mapratio=False):
    """
    geodetic2apex(gdlat, gdlon, gdalt_km[, times])
    gdlat, gdlon in degrees
    times: datetime list object
    apexRefTime: datetime object
    Returns
    """

    import apexpy
    from pyamps.mlt_utils import mlon_to_mlt
    from dateutil.relativedelta import relativedelta

    debug = True
    canDoMLT = True
    multitime = False

    get_apex_basevecs = return_apex_d_basevecs or return_apex_e_basevecs or \
        return_apex_f_basevecs or return_apex_g_basevecs or return_mapratio

    if max_N_months_twixt_apexRefTime_and_obs is None:
        max_N_months_twixt_apexRefTime_and_obs = 0

    assert len(args) == 4, "geodetic2apex(gdlat, gdlon, gdalt_km,times)"

    # Find out --- should we even worry about time here?
    haveDTIndex = isinstance(args[3],pd.DatetimeIndex)
    if haveDTIndex:
        multitime = True
    else:
        if isinstance(args[3],list) or isinstance(args[3],np.ndarray):
            if len(args[3]) == len(args[2]):
                multitime = True
            # else:
            #     multitime = False

    # Set up apex object
    a = apexpy.Apex(apexRefTime, refh=apexRefHeight_km)

    # If not doing multiple times, just do the conversions and get out
    if not multitime:
        if do_qdcoords:
            mlat, mlon = a.geo2qd(
                args[0], args[1], args[2])
        else:
            mlat, mlon = a.geo2apex(
                args[0], args[1], args[2])
    
        returnList = [mlat, mlon]
        rListNames = ['mlat', 'mlon']
    
        if canDoMLT:
            mlt = mlon_to_mlt(mlon, [args[3]]*len(mlon), args[3].year)
    
            returnList.append(mlt)
            rListNames.append('mlt')

        returnDict = {key: val for key, val in zip(rListNames, returnList)}

        if returnPandas:
            dfOut = pd.DataFrame(data=np.vstack(returnList).T,columns=rListNames)

            return dfOut

        else:
            return returnDict

    df = pd.DataFrame(
        {'gdlat': args[0], 'gdlon': args[1], 'gdalt_km': args[2]}, index=args[3])

    # Interp over nans
    checkCols = ['gdlat', 'gdalt_km']
    interp_over_nans(df, checkCols,
                     max_Nsec_twixt_nans=1,
                     max_Nsec_tot=5,
                     interpolateArgs={'method': 'time', 'limit': 21})

    if df.isna().any().any() and not nancheck:
        print("HELP (or just set nancheck=True)!")
        breakpoint()

    ########################################
    # check if we need to downsample/consider subset

    period_df = ((df.iloc[1].name -
                  df.iloc[0].name)/pd.Timedelta('1s'))

    is_subsampled = False
    if period_df < min_time_resolution__sec:

        strider = int(min_time_resolution__sec/period_df)

        if debug:
            print("DEBUG   DOWNSAMPLE: Reducing number of conversions by a factor of {:d}".format(
                strider))
        assert 2<0,"Temporarily disabling subsampling because I don't want to screw up Swarm TII observational database for SWIPE model"
        dfSub = df.iloc[0::strider]
        is_subsampled = True

    else:
        dfSub = df

    # Return these to NaNs further down ...
    if nancheck:

        nanners = dfSub['gdlat'].isna(
        ) | dfSub['gdlon'].isna() | dfSub['gdalt_km'].isna()

        if nanners[nanners].size/nanners.size > 0.0:
            if not quiet:
                print("nannies!"+"{0}".format(nanners[nanners].size))
            dfSub.loc[nanners, 'gdlat'] = 0
            dfSub.loc[nanners, 'gdalt_km'] = 0


    # Now group by max_N_months_twixt_apexRefTime_and_obs
    times = dfSub.index.to_pydatetime()
    apexRefTime = times[0]

    relDelta = relativedelta(
        months=max_N_months_twixt_apexRefTime_and_obs)

    maxIterHere = 3000
    nIter = 0

    NOBS = dfSub['gdlat'].values.shape[0]
    mlat = np.zeros(NOBS)*np.nan
    mlon = np.zeros(NOBS)*np.nan

    if get_apex_basevecs:
        
        if return_apex_d_basevecs:
            d1,d2,d3 = np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan
        if return_apex_e_basevecs:
            e1,e2,e3 = np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan
        if return_apex_f_basevecs:
            f1,f2,f3 = np.zeros((2,NOBS))*np.nan,np.zeros((2,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan
        if return_apex_g_basevecs:
            g1,g2,g3 = np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan,np.zeros((3,NOBS))*np.nan

    while apexRefTime < times[-1]:

        # See if we have any here; if not, skip

        ind_timesHere = (times >= apexRefTime) & (
            times < (apexRefTime+relDelta))

        nIndsHere = np.where(ind_timesHere)[0].size

        if debug:
            print("DEBUG   {:s} to {:s} : Got {:d} inds for mlat/mlon conversion".format(
                apexRefTime.strftime("%Y%m%d"),
                (apexRefTime+relDelta).strftime("%Y%m%d"),
                nIndsHere))

        if nIndsHere == 0:
            # Increment apexRefTime by relDelta
            apexRefTime += relDelta
            nIter += 1
            continue

        a.set_epoch(toYearFraction(apexRefTime))

        if do_qdcoords:
            mlattmp, mlontmp = a.geo2qd(
                dfSub['gdlat'].values[ind_timesHere],
                dfSub['gdlon'].values[ind_timesHere],
                dfSub['gdalt_km'].values[ind_timesHere])
        else:
            mlattmp, mlontmp = a.geo2apex(
                dfSub['gdlat'].values[ind_timesHere],
                dfSub['gdlon'].values[ind_timesHere],
                dfSub['gdalt_km'].values[ind_timesHere])

        mlat[ind_timesHere] = mlattmp
        mlon[ind_timesHere] = mlontmp

        # Basevectors
        if get_apex_basevecs:

            # if not quiet:
            #     print("Getting Apex basevectors ...")
            # From Laundal and Richmond (2016):
            # e1 "points eastward along contours of constant λma,"
            # e2 "points equatorward along contours of constant φ ma (magnetic meridians)"
            # "t" stands for "temporary"
            #
            # From apexpy.apex.basevectors_apex:
            # "vector components are geodetic east, north, and up (only east and north for `f1` and `f2`)"
            f1t, f2t, f3t, g1t, g2t, g3t, d1t, d2t, d3t, e1t, e2t, e3t = a.basevectors_apex(
                dfSub['gdlat'].values[ind_timesHere],
                dfSub['gdlon'].values[ind_timesHere],
                dfSub['gdalt_km'].values[ind_timesHere], coords='geo')

            if return_apex_d_basevecs or return_mapratio:
                d1[:,ind_timesHere] = d1t
                d2[:,ind_timesHere] = d2t
                d3[:,ind_timesHere] = d3t
            if return_apex_e_basevecs:
                e1[:,ind_timesHere] = e1t
                e2[:,ind_timesHere] = e2t
                e3[:,ind_timesHere] = e3t
            if return_apex_f_basevecs:
                f1[:,ind_timesHere] = f1t
                f2[:,ind_timesHere] = f2t
                f3[:,ind_timesHere] = f3t
            if return_apex_g_basevecs:
                g1[:,ind_timesHere] = g1t
                g2[:,ind_timesHere] = g2t
                g3[:,ind_timesHere] = g3t

        # Increment apexRefTime by relDelta
        apexRefTime += relDelta

        nIter += 1
        if nIter >= maxIterHere:
            print("Too many iterations! Breaking ...")
            break


    if canDoMLT:

        if max_N_months_twixt_apexRefTime_and_obs == 0:

            mlt = mlon_to_mlt(mlon, times, times[0].year)

        else:

            print("Updating apexRefTime as we go ...")

            mlt = np.zeros(mlat.shape)*np.nan

            wasSorted = isSorted(times)
            if not wasSorted:
                print("Times not sorted! Sorting ...")
                sortinds = np.argsort(notsorted)
                unsortinds = np.argsort(sortinds)

                times = np.array(times)[sortinds]
                mlat = mlat[sortinds]
                mlon = mlon[sortinds]

            # Now group by max_N_months_twixt_apexRefTime_and_obs
            apexRefTime = times[0]

            maxIterHere = 3000
            nIter = 0
            while apexRefTime < times[-1]:

                # See if we have any here; if not, skip

                ind_timesHere = (times >= apexRefTime) & (
                    times < (apexRefTime+relDelta))

                nIndsHere = np.where(ind_timesHere)[0].size

                if debug:
                    print("DEBUG   {:s} to {:s} : Got {:d} inds for MLT conversion".format(
                        apexRefTime.strftime("%Y%m%d"),
                        (apexRefTime+relDelta).strftime("%Y%m%d"),
                        nIndsHere))

                if nIndsHere == 0:
                    # Increment apexRefTime by relDelta
                    apexRefTime += relDelta
                    nIter += 1
                    continue

                a.set_epoch(toYearFraction(apexRefTime))

                maxNIndsSamtidig = 500000
                if nIndsHere > maxNIndsSamtidig:
                    print("Break it up...")

                    indbatchCounter = 0
                    nIndsConverted = 0
                    while nIndsConverted < nIndsHere:

                        startIndInd = nIndsConverted
                        stopIndInd = np.min(
                            [startIndInd+maxNIndsSamtidig, nIndsHere])
                        nToConvert = stopIndInd-startIndInd

                        tmpUseInds = np.where(ind_timesHere)[
                            0][startIndInd:stopIndInd]
                        mlt[tmpUseInds] = mlon_to_mlt(mlon[tmpUseInds],
                                                      times[tmpUseInds],
                                                      times[tmpUseInds][0].year)


                        nIndsConverted += nToConvert

                else:
                    mlt[ind_timesHere] = mlon_to_mlt(mlon[ind_timesHere],
                                                     times[ind_timesHere],
                                                     times[ind_timesHere][0].year)

                # Increment apexRefTime by relDelta
                apexRefTime += relDelta

                nIter += 1
                if nIter >= maxIterHere:
                    print("Too many iterations! Breaking ...")
                    break

            if not wasSorted:
                print("Unsorting things again, you filthy animal")
                times = list(times[unsortinds])
                mlat = mlat[unsortinds]
                mlon = mlon[unsortinds]
                mlt = mlt[unsortinds]

    returnList = [mlat, mlon]
    rListNames = ['mlat', 'mlon']

    if canDoMLT:
        returnList.append(mlt)
        rListNames.append('mlt')

    if get_apex_basevecs:

        if return_mapratio:
            mapratio = 1. / np.linalg.norm(np.cross(d1.T, d2.T), axis=1)
            returnList = returnList + [mapratio]
            rListNames = rListNames + ['mapratio']

        if return_apex_d_basevecs:
            returnList = returnList + [d1[0, ], d1[1, ], d1[2, ],
                                       d2[0, ], d2[1, ], d2[2, ],
                                       d3[0, ], d3[1, ], d3[2, ]]
            rListNames = rListNames + ['d10', 'd11', 'd12',
                                       'd20', 'd21', 'd22',
                                       'd30', 'd31', 'd32']

        if return_apex_e_basevecs:
            returnList = returnList + [e1[0, ], e1[1, ], e1[2, ],
                                       e2[0, ], e2[1, ], e2[2, ],
                                       e3[0, ], e3[1, ], e3[2, ]]
            rListNames = rListNames + ['e10', 'e11', 'e12',
                                       'e20', 'e21', 'e22',
                                       'e30', 'e31', 'e32']
        if return_apex_f_basevecs:
            returnList = returnList + [f1[0, ], f1[1, ], #f1[2, ],
                                       f2[0, ], f2[1, ], #f2[2, ],
                                       f3[0, ], f3[1, ], f3[2, ]]
            rListNames = rListNames + ['f10', 'f11', #'f12',
                                       'f20', 'f21', #'f22',
                                       'f30', 'f31', 'f32']
        if return_apex_g_basevecs:
            returnList = returnList + [g1[0, ], g1[1, ], g1[2, ],
                                       g2[0, ], g2[1, ], g2[2, ],
                                       g3[0, ], g3[1, ], g3[2, ]]
            rListNames = rListNames + ['g10', 'g11', 'g12',
                                       'g20', 'g21', 'g22',
                                       'g30', 'g31', 'g32']

    if nancheck:
        if nanners[nanners].size/nanners.size > 0.0:
            dfSub.loc[nanners, 'gdlat'] = np.nan
            dfSub.loc[nanners, 'gdalt_km'] = np.nan

    ########################################
    # Make final outputdict
    returnDict = {key: val for key, val in zip(rListNames, returnList)}

    if returnPandas:

        if is_subsampled:
            intoApexIndex = df.index.intersection(dfSub.index)

            dfOut = pd.DataFrame(columns=rListNames, dtype=np.float64,
                                 index=df.index)

            shouldBeUnwrapped = ['mlon', 'mlt']
            for col in rListNames:

                if col in shouldBeUnwrapped:
                    if col == 'mlon':
                        dfOut.loc[intoApexIndex, col] = np.unwrap(
                            np.deg2rad(returnDict['mlon']))
                    elif col == 'mlt':
                        dfOut.loc[intoApexIndex, 'mlt'] = np.unwrap(
                            np.deg2rad(returnDict['mlt']*15.))
                    else:
                        print("What is this?")
                        breakpoint()

                else:
                    dfOut.loc[intoApexIndex, col] = returnDict[col]

                dfOut.loc[:, col] = dfOut[col].interpolate(**interpolateArgs)

                if col in shouldBeUnwrapped:
                    if col == 'mlon':
                        dfOut.loc[:, 'mlon'] = np.rad2deg(
                            (dfOut['mlon'].values + np.pi) % (2 * np.pi) - np.pi)
                    elif col == 'mlt':
                        dfOut.loc[:, 'mlt'] = np.rad2deg(
                            (dfOut['mlt'].values) % (2 * np.pi))/15.

        else:
            dfOut = pd.DataFrame(data=np.vstack(returnList).T,
                                 columns=rListNames, index=df.index)

        return dfOut

    else:
        return returnDict


def interp_over_nans(df, checkCols,
                     max_Nsec_twixt_nans=1,
                     max_Nsec_tot=5,
                     interpolateArgs={'method': 'time', 'limit': 21}):

    max_Nsec_dt = pd.Timedelta(str(max_Nsec_twixt_nans)+'s')
    max_Nsec_dtTot = pd.Timedelta(str(max_Nsec_tot)+'s')

    naninds = (df[checkCols[0]] != df[checkCols[0]]) & False
    for col in checkCols:
        naninds = naninds | df[col].isna()
    naninds = np.where(naninds)[0]

    if naninds.size == 0:
        print("No NaNs in this df (at least not in {:s})!".format(
            ", ".join(checkCols)))
        return

    if naninds.size > 0:
        # Check if good timeresolution before and after

        nanindgroups = group_consecutives(naninds)

        print("Interping over {:d} nanInd groups ({:d} total nanInds) ...".format(
            len(nanindgroups),
            naninds.size))

        for indgrp in nanindgroups:

            nInGrp = indgrp.size
            if indgrp[0] == 0:
                startInd = 0
                stopInd = 1
            else:
                startInd = indgrp[0] - 1

            if indgrp[-1] == df.shape[0]-1:
                stopInd = df.shape[0]-1
            else:
                stopInd = indgrp[-1] + 1

            # dtBef = df.iloc[ind].name-df.iloc[startInd].name
            # dtAft = df.iloc[stopInd].name-df.iloc[ind].name
            # if (dtBef <= pd.Timedelta('1s')) and (dtAft <= pd.Timedelta('1s')):

            dtTot = df.iloc[stopInd].name-df.iloc[startInd].name
            if (dtTot/nInGrp <= max_Nsec_dt) and (dtTot <= max_Nsec_dtTot):

                # print("Good, Rickie")

                df.loc[df.iloc[startInd:stopInd+1].index,
                       :] = df.iloc[startInd:stopInd+1].interpolate(**interpolateArgs)


def group_consecutives(vals, maxDiff=1,
                       min_streak=None,
                       do_absDiff=False,
                       print_summary=False,
                       print__maxVal=None):
    """Return list of consecutive lists of numbers from vals (number list).

    Based on https://stackoverflow.com/questions/7352684/
    how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    """

    # assert np.issubdtype(maxDiff,np.integer),"maxDiff must be of type integer!"
    if maxDiff <= 0:
        print("maxDiff must be >= 0!")
        return np.array([], dtype=np.int64)

    if do_absDiff:
        this = np.split(vals, np.where(np.abs(np.diff(vals)) > maxDiff)[0]+1)
    else:
        this = np.split(vals, np.where(np.diff(vals) > maxDiff)[0]+1)

    # if min_streak is None:
        # return this
    # else:
    if min_streak is not None:
        keep = []
        for liszt in this:
            # print(liszt.size)
            if liszt.size >= min_streak:
                keep.append(liszt)

        # return keep
        this = keep

    if print_summary:
        titleStr = "{:2s} {:{width}s}   {:{width}s} - {:{width}s} (/{:{width}s})"
        rowStr = titleStr.replace("s", "d")

        # rowStr = "{:2d} {:{width}d}   {:{width}d} - {:{width}d} (/{:{width}d})"

        nDigs = str(np.int64(np.log10(np.max(vals))))

        if print__maxVal is None:
            print__maxVal = np.max(vals)
        print(titleStr.format("i",
                              "N",
                              "strt",
                              "stop",
                              "tot",
                              width=nDigs))
        for i, dudG in enumerate(this):
            print(rowStr.format(i,
                                len(dudG),
                                dudG[0],
                                dudG[-1],
                                print__maxVal,
                                width=nDigs))

    return this
