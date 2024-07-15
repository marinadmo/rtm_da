#
import numpy as np
import netCDF4
import xarray as xr
import sys, os
import datetime
import glob
from subprocess import call

# Function for running in terminal
def cmd(command):
    print(command)
    result = call(command, shell = True)
    if result != 0:
        print("Command failed: %d" % result)
    else:
        return result

'''
def class globalVars():
    channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
    Nens = 10 # Model ensemble size
    mask_file = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/2024/01/07/20240107_dm-metno-MODEL-topaz5-ARC-b20231229-fv02.0_mem005.nc' # File used for masking RTM output
    model = 'topaz' # Model used for the RTM simulation
    size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000
'''
