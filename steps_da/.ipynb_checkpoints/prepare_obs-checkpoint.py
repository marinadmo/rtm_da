# Preparation of observations for SYN and ASYN anlysis (L4 and L3 SIC obs)

import numpy as np
import xarray as xr
import glob
from datetime import datetime, timedelta
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


def read_mask() :
    
    model_file = config.mask_file
    model_var = 'model_mask'
    model_data = xr.open_dataset(model_file)
    mask = np.logical_not(model_data[model_var])
    
    return mask

def update_obs(infile, obsfile, passes = False, bounded = True) :

    # Model mask
    mask = read_mask()

    # Open observation dataset
    sat_data = xr.open_dataset(infile)
    print('In file:', infile)
    print('Variables:', sat_data.variables)

    if passes : fsat2 = infile.split('_')[2].split('.')[0]
    else : fsat2 = config.date
    if '2021' in config.date : # Problem with time data in new files!! 
        files_sat2 = glob.glob(f"{config.sat_data_dir2}/*{fsat2}*nc")[0]
        print(files_sat2)
        sat_data2 = xr.open_dataset(files_sat2)
        time_data = sat_data2.time.data
    else : time_data = sat_data.time.data
    print('TIME: ', time_data)

    if '2024' in config.date :
        lon_data = sat_data.lon.isel(yc = slice(None, None, -1)).data
        lat_data = sat_data.lat.isel(yc = slice(None, None, -1)).data
        ice_data = np.ma.masked_array(((sat_data.ice_conc[:]/100).clip(0, 1).isel(yc = slice(None, None, -1)).data), mask = mask)
        tb19v_data = np.ma.masked_array((sat_data.tb19v.isel(yc = slice(None, None, -1)).data), mask = mask)
        tb19h_data = np.ma.masked_array((sat_data.tb19h.isel(yc = slice(None, None, -1)).data), mask = mask)
        tb37v_data = np.ma.masked_array((sat_data.tb37v.isel(yc = slice(None, None, -1)).data), mask = mask)
        tb37h_data = np.ma.masked_array((sat_data.tb37h.isel(yc = slice(None, None, -1)).data), mask = mask)
        x_data = sat_data.xc.data[::-1]
        y_data = sat_data.yc.data[::-1]
        
    elif '2021' in config.date :
        lon_data = sat_data.lon.data
        lat_data = sat_data.lat.data
        if bounded :
            print(f'Bounded {bounded}')
            ice_data = np.ma.masked_array((sat_data.ice_conc[:]/100).clip(0, 1).data, mask = mask)
        else :
            print(f'Bounded {bounded}')
            ice_data = np.ma.masked_array((sat_data.ice_conc[:]/100).clip(-1.5, 1.5).data, mask = mask)    
        # Filter OW in SIC owf_SICCI3LF_corrSICCI3LF
        owf_sat = (sat_data.owf_SICCI3LF_corrSICCI3LF[:]); ice_data.data[owf_sat.data == 1] = 0.0
        # Climatology file
        clim_path = '/lustre/storeB/project/fou/fd/project/acciberg/marina/climatology/'
        data_clim = xr.open_dataset(f'{clim_path}LandOceanLakeMask_nh_ease2-250_resampled.nc')['climatology'][int(config.date[4:6]) - 1, :, :]; data_clim = np.expand_dims(data_clim, axis = 0) 
        print(np.shape(data_clim), np.shape(ice_data))
        #[int(config.date[4:6]) - 1, :, :]
        ice_data.data[data_clim.data == 0.0] = 0.0
        
        tb19v_data = np.ma.masked_array(sat_data.tb19v.data, mask = mask)
        tb19h_data = np.ma.masked_array((sat_data.tb19h.data), mask = mask)
        tb37v_data = np.ma.masked_array((sat_data.tb37v.data), mask = mask)
        tb37h_data = np.ma.masked_array((sat_data.tb37h.data), mask = mask)
        x_data = sat_data.xc.data
        y_data = sat_data.yc.data

    print('TIME: ', time_data)
    ds = xr.Dataset(

        {"longitude": (("y", "x"), lon_data),
        "latitude": (("y", "x"), lat_data),
        "ice_conc": (("time", "y", "x"), ice_data),
        "tb19v": (("time", "y", "x"), tb19v_data),
        "tb19h": (("time", "y", "x"), tb19h_data),
        "tb37v": (("time", "y", "x"), tb37v_data),
        "tb37h": (("time", "y", "x"), tb37h_data)
            },

        coords = {

            "x": sat_data.xc.data[::-1],

            "y": sat_data.yc.data[::-1],

            "time": time_data,

        },

    )

    ds.to_netcdf(obsfile)
    

def prep_topaz() :

    obs_dir = f"{config.exps_dir}/observations/means/"
    sat_dir = f"{config.sat_data_dir}" #main_data_dir}atlems/topaz_l3/"

    print(f"{sat_dir}/*{config.date}*nc")
    files_sat = glob.glob(f"{sat_dir}/*{config.date}*nc")[0] #{config.date[0:4]}
    print(files_sat)
    new_obs_file = f"{obs_dir}amsr2_topaz_obs_{config.date}.nc"
    cmd('rm ' + new_obs_file)
    tmpfile = obs_dir + '/tmp/tmp.nc'
    tmpfile2 = obs_dir + '/tmp/tmp2.nc'
    print('file sat:', files_sat)#owf_SICCI3LF_corrSICCI3LF
    cmd('ncks -v lat,lon,ct_SICCI3LF_corrSICCI3LF,owf_SICCI3LF_corrSICCI3LF,tb19v,tb19h,tb37v,tb37h ' + files_sat + ' ' + tmpfile)
    cmd('ncrename -v ct_SICCI3LF_corrSICCI3LF,ice_conc ' + tmpfile + ' ' + tmpfile2)
    cmd('rm ' + tmpfile)

    bounded = True
    if bounded : 
        print('SIC is BOUNDED!!')
    else : print('SIC is NOT BOUNDED!!')
    update_obs(tmpfile2, new_obs_file, passes = False, bounded = bounded)
    cmd('rm ' + tmpfile2)

    
def prep_topaz_passes() :
	
    obs_dir = f"{config.exps_dir}/observations/passes/"
    sat_dir = f"{config.sat_data_dir}" 
    
    # Previous day
    previous_day = (datetime.strptime(config.date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
    # Check list of passes
    list_passes = glob.glob(f"{sat_dir}/*{previous_day}*nc") 
    print(list_passes)

    for file_sat in list_passes :

        print(' ')
        print('PASS FILE NAME: ', file_sat)
        pass_name = file_sat.split('_')[4]
        print(pass_name)
        print(' ')

        new_obs_file = f"{obs_dir}amsr2_topaz_obs_{pass_name}.nc"
        cmd('rm ' + new_obs_file)
        tmpfile = obs_dir + '/tmp/tmp.nc'
        tmpfile2 = f'{obs_dir}/tmp/tmp2_{pass_name}.nc'
        
        print(' ')
        print('PASS FILE NAME again: ', file_sat)
        cmd('ncks -v lat,lon,ct_SICCI3LF_corrSICCI3LF,owf_SICCI3LF_corrSICCI3LF,tb19v,tb19h,tb37v,tb37h ' + file_sat + ' ' + tmpfile )
        cmd('ncrename -v ct_SICCI3LF_corrSICCI3LF,ice_conc ' + tmpfile + ' ' + tmpfile2 )
        print(' ')
        cmd('rm ' + tmpfile)

        bounded = True
        if bounded : 
            print('SIC is BOUNDED!!')
        else : print('SIC is NOT BOUNDED!!')
        update_obs(tmpfile2, new_obs_file, passes = True, bounded = bounded)
        cmd('rm ' + tmpfile2)