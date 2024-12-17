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
