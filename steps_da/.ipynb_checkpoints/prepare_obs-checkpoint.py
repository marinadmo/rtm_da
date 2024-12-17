#Preparation of observations for SYN and ASYN anlysis (L4 and L3 SIC obs)

import sys, os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Code to build ensemble files from model output to feed enkf-c
from main_imports import *
# Now you can import the config module
import config


def read_mask2() :
    # Topaz5 data
    model_file = config.mask_file
    if '2024' in config.date : model_var = 'model_depth'
    elif '2021' in config.date : model_var = 'aice_d'

    print(model_file)
    model_data = xr.open_dataset(model_file)
    mask = model_data[model_var].to_masked_array().mask
    return mask 

def read_mask() :
    model_file = config.mask_file_plots
    model_var = 'model_mask'
    model_data = xr.open_dataset(model_file)
    mask = np.logical_not(model_data[model_var])
    return mask

def update_obs(infile, obsfile, passes = False) :

    # Model mask
    mask = read_mask()

    # Open observation dataset
    sat_data = xr.open_dataset(infile)

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
        ice_data = np.ma.masked_array((sat_data.ice_conc[:]/100).clip(0, 1).data, mask = mask)
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

    obs_dir = f"{config.exps_dir}/observations/"
    sat_dir = f"{config.sat_data_dir}" #main_data_dir}atlems/topaz_l3/"

    print(f"{sat_dir}/*{config.date}*nc")
    files_sat = glob.glob(f"{sat_dir}/*{config.date}*nc")[0] #{config.date[0:4]}
    print(files_sat)
    new_obs_file = f"{obs_dir}amsr2_topaz_obs_{config.date}.nc"
    cmd('rm ' + new_obs_file)
    tmpfile = obs_dir + '/tmp/tmp.nc'
    tmpfile2 = obs_dir + '/tmp/tmp2.nc'
    cmd('ncks -v lat,lon,ct_SICCI3LF_corrSICCI3LF,tb19v,tb19h,tb37v,tb37h ' + files_sat + ' ' + tmpfile )
    cmd('ncrename -v ct_SICCI3LF_corrSICCI3LF,ice_conc ' + tmpfile + ' ' + tmpfile2 )
    cmd('rm ' + tmpfile)

    update_obs(tmpfile2, new_obs_file)
    cmd('rm ' + tmpfile2)

    
def prep_topaz_passes() :
	
    obs_dir = f"{config.exps_dir}/observations/passes/"
    sat_dir = f"{config.sat_data_dir}" 
    
    # Check list of passes
    list_passes = glob.glob(f"{sat_dir}/*{config.date}*nc") 
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
        cmd('ncks -v lat,lon,ct_SICCI3LF_corrSICCI3LF,tb19v,tb19h,tb37v,tb37h ' + file_sat + ' ' + tmpfile )
        cmd('ncrename -v ct_SICCI3LF_corrSICCI3LF,ice_conc ' + tmpfile + ' ' + tmpfile2 )
        print(' ')
        cmd('rm ' + tmpfile)

        update_obs(tmpfile2, new_obs_file, passes = True)
        cmd('rm ' + tmpfile2)