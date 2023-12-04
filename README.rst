|DOI| 


Swarm Hi-C inversion code. Adapted (with heavy modifications) from Karl Laundal's AMPS inversion code.

This code was used to produce model coefficients for the Swipe model. The forward model is available here: https://zenodo.org/badge/latestdoi/685879333 (latest development version available at https://github.com/Dartspacephysiker/pyswipe)

There are two parts to this code. In the data_preparation folder, there are scripts that are used to produce a static datafile which is later used as input in the inversion code. This is in the data_preparation folder. The inversion code is included in the base directory. 

The data preparation files do the following:
# 01_make_swarm_CT2Hz_hdfs.py          : Open up all the .cdf.zip files and put their contents, along with some stuff in Modified Apex-110 coordinates, into an HDF file.
# 02_f107_download_and_filter.py       : Add F10.7
# 03_omni_download_1min_data.py        : Download OMNI data (IMF components, solar wind speed and density, SYM-H, + others?
# 04_omni_process_1min_data.py         : Process OMNI data (calculate IMF clock angle mean,variance, average over 30-min window, etc.)
# 05_add_omni_f107_dptilt_substorms.py : Sort index of each column in HDF file, add F10.7, OMNI, dipole tilt, B0_IGRF, og the rest to an HDF file
# 06_add_crosstrack_vector_info.py     : Calculate cross-track convection in MA-110 coordinates, add these to HDF:
#                                        ['Viy_d1','Viy_d2',
#                                         'Viy_f1','Viy_f2',
#                                         'yhat_d1', 'yhat_d2',
#                                         'yhat_f1', 'yhat_f2',
#                                         'gdlat', 'alt']
# 07_make_model_dataset.py             : Read HDF store files, calculate all the weights, (optionally) retain only measurements with a particular quality flag,
#                                        and then store weights, coordinates, and measurements in  a format that can be streamed using dask

Running the scripts in numerical order should produce a static datafile which is the only input to the inversion code. 

The FINAL inversion code is "hdl_model_iteration__Lowes1966_regularization__analytic_potzero_with_A_matrix.py"

Python dependencies
===================================
- apexpy 
- bs4
- cdflib
- dask
- dipole #https://github.com/klaundal/dipole
- ftplib
- fnmatch
- glob
- datetime
- dateutil
- h5py
- numpy
- os
- pandas
- ppigrf
- pyamps (for mlon_to_mlt conversion)
- requests
- scipy
- spacepy
- wget
- zipfile


Acknowledgments
---------------
The code is produced with support from the European Space Agency through the Swarm Data Innovation and Science Cluster (Swarm DISC), ESA Contract no. 4000109587/13/I-NB. 

For more information on the Swipe project, please visit https://earth.esa.int/eogateway/activities/swipe

For more information on Swarm DISC, please visit https://earth.esa.int/web/guest/missions/esa-eo-missions/swarm/disc


.. |DOI| image:: https://zenodo.org/badge/674153432.svg
        :target: https://zenodo.org/badge/latestdoi/674153432
