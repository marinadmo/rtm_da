# Code to update model ensemble after data assimilation

import numpy as np
import netCDF4
from subprocess import call

# Function for running in terminal
def cmd(command):
    print(command)
    result = call(command, shell = True)
    if result != 0:
        print("Command failed: %d" % result)
    else:
        return result

import config
print(f"Config module loaded from: {config.__file__}")

def updated_categories(ens) :
    ensb_dir = f'{config.storage_dir}ensb/'
    postprocess_dir = f'{config.storage_dir}postprocess/'
    # Read total ice concentration (fice_a, fice_f)
    fice_f = f'{ensb_dir}/mem{ens + 1:03d}_{var}.nc'
    fice_a = f'{postprocess_dir}/mem{ens + 1:03d}_{var}.nc.analysis'
    # Read category files (aicen, vicen background and analysis)
    aicen_f = f'{ensb_dir}/mem{ens + 1:03d}_{var}.nc'
    #aicen_a
    vicen_f = f'{ensb_dir}/mem{ens + 1:03d}_{var}.nc'
    #vicen_a
    
    ncat = np.shape(aicen_f)[0]
    nj = np.shape(aicen_f)[1]
    ni = np.shape(aicen_f)[2]
    
    bkcat = [0, 0.6445072, 1.391433, 2.470179, 4.567288, 12.]
    
    # Compute total ice concentration
    fice_f = np.nansum(aicen_f, axis = 0)
    fice_a = np.nansum(aicen_a, axis = 0)
    
    # The values of aicen within [0 1], and  the sum of ice concentration not exceeds 1
    aicen_a = np.clip(aicen_a, 0.0, 1.0)    
    
    for j in range(0, nj) :
        for i in range(0, ni) :
            for k in range(0, ncat) :
                # Rescaling the concentrations according to the previous fraction distribution in the 5 categories.
                aicen_a[k, j, i] = aicen_f[k, j, i]*fice_a[j, i]/fice_f[j, i]
                # Adjust the ice volumes using the rescaled the concentrations so it means the ice thickness distribution will be same as before.
                if (aicen_f[k, j, i] > 0 and aicen_a[k, j, i] > 0) :
                    vicen_a[k, j, i] = vicen_a[k, j, i]*aicen_a[k, j, i]/aicen_f[k, j, i]
                # The adjusted ice volume should be in a range of each category
                vicen_a[k, j, i] = max(vicen_a[k, j, i], aicen_a[k, j, i]*bkcat[k])
                vicen_a[k, j, i] = min(vicen_a[k, j, i], aicen_a[k, j, i]*bkcat[k+1])
        
    return aicen_a, vicen_a

    
    
def up_ensemble():
    # Ice variables to be updated
    ice_vars = ('aice_d', 'hi_d', 'hs_d', 'alvl_d', 'vlvl_d', 'iage', 'FYarea') 
    #varnames = ('aice_d', 'hi_d', 'hs_d', 'Tsfc_d', 'sst_d', 'sss_d', 'alvl_d', 'vlvl_d', 'iage', 'FYarea')
    ocean_vars = ('sst_d', 'sss_d')
    ensa_dir = f'{config.storage_dir}ensa/'
    postprocess_dir = f'{config.storage_dir}postprocess/'
    
    for ens in range(0, config.Nens) :
        for var in ice_vars :
            #cmd(f'cp {ensa_dir}/*{var}.nc.analysis {postprocess_dir}')
            print('We are updating variable: ', var)
            ice_analysis_file = f'{ensa_dir}/mem{ens + 1:03d}_{var}.nc.analysis'
            ice_ini_file =  f'{postprocess_dir}/mem{ens + 1:03d}_{var}.nc.analysis'
            cmd(f'cp {ice_analysis_file} {postprocess_dir}')
            with netCDF4.Dataset(ice_analysis_file, mode = "r") as ice_ana, netCDF4.Dataset(ice_ini_file, mode = "r+") as ice_ini:
                temp = ice_ana.variables[var][0,:]
                if var == 'aice_d' :
                    temp[temp < 0.01] = 0
                    temp[temp > 1] = 1
                    aice = temp.copy()
                else :
                    temp[temp < 0.0] = 0
                    temp[aice == 0] = 0 
                    
                if var == 'FYarea' :
                    temp[temp > 1] = 1
                
                ice_ini.variables[var][0, :] = temp   
                print('write variable') 
                
        for var in ocean_vars :
            print('We are updating variable: ', var)
            ocean_analysis_file = f'{ensa_dir}/mem{ens + 1:03d}_{var}.nc.analysis'
            ocean_ini_file =  f'{postprocess_dir}/mem{ens + 1:03d}_{var}.nc.analysis'
            cmd(f'cp {ocean_analysis_file} {postprocess_dir}')
            with netCDF4.Dataset(ocean_analysis_file, mode = "r") as ocean_ana, netCDF4.Dataset(ocean_ini_file, mode = "r+") as ocean_ini:
                temp = ocean_ana.variables[var][0,:]
                if var == 'sst_d' :
                    temp[temp < -2] = -2
                elif var == 'sss_d' :
                    temp[temp < 0] = 0
                    
                ocean_ini.variables[var][0, :] = temp   
                print('write variable')          
