#Preparation of observations for SYN and ASYN anlysis (L4 and L3 SIC obs)

# Import python functions for DAL computation
#from .dyn_algos_marina import read_dyn_algos
#import io_handler

# Import RTM python code
#import rtm_dal.rtm_amsr_fcts as rtm_amsr #rtm_dal.

# Use absolute imports
from rtm_dal.dyn_algos_marina import read_dyn_algos_v2
import io_handler
import rtm_dal.rtm_amsr_fcts as rtm_amsr

import numpy as np
import netCDF4
import xarray as xr
import datetime
import os, glob

from sklearn import linear_model
import pandas as pd
import csv

channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h')
#import config
#print(f"Config module loaded from: {config.__file__}")

def compute_dal(date, size_tpd_data) : #tpd_data_dir, tpa_data_dir, size_tpd_data) :    
    
    '''
    This function computes Distance Along the Line (DAL, Lavergne et al., 2019) from observed Tbs (19v, 37h, 37v)
    
    Input
        date: string format (ex: '20240101')
        tpd_dir: directory of TPD (Tie Point) files 
        json_dir: directory of JSON files
    
    Output
        DAL: DAL (normalized or owf); unity: Kelvin    
    '''  

    #size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000
    interval = 7 # DAL is computed for a period of 7 days    
    m_dal_norm, m_dal_owf = [np.zeros((interval, size_tpd_data)) for _ in range(2)]
    list_tpd_files = []  
    
    # Convert the string to a datetime object
    date_obj = datetime.datetime.strptime(date, '%Y%m%d')
    
    # Loop through the interval of 7 days
    for i in range(-(interval - 1), 1) :
         
        current_date = date_obj + datetime.timedelta(days = i); current_date_str = current_date.strftime('%Y%m%d')           
        year = f"{current_date.year:04}"
        month = f"{current_date.month:02}"
        day = f"{current_date.day:02}" 

        '''
        if date[0:4] in ('2024', '2023') :
            period = '7'; centre = False; 
            dyn_algos_return = read_dyn_algos(sat_id, f"{tpa_data_dir}{year}/{month}/", corr, period, current_date, centre, algorithms, ignore, yesterday)     
        elif '2021' in date[0:4] :
            period = '15'; centre = True; 
            dyn_algos_return = read_dyn_algos(sat_id, f"{tpa_data_dir}{year}/{month}/{day}/", corr, period, current_date, centre, algorithms, ignore, yesterday)    
        # TPD files
        #if date[0:4] in ('2023', '2024', '2025') :
        if int(date[0:4]) >= 2023:
            swath_filename = f"{tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
        else : #elif '2021' in date[0:4] :
            swath_filename = f"{tpd_data_dir}{year}/{month}/{day}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
        '''

        # TPA files (JSON files)
        sat_id = 'amsr_gw1'; corr = 'ucorr';
        algorithms = ['SICCI3LF']; ignore = False; 
        search_closest = False #yesterday = False
        
        if (int(year), int(month)) >= (2022, 12):
            print("The date is from December 2022 to the future.")            
            tpa_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpa/' # TPA files directory
            period = '7'; centre = False; 
            tpa_folder = f"{tpa_data_dir}{year}/{month}/"
            # 
            tpd_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpd/' # TPD files directory
            swath_filename = f"{tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
        elif (int(year), int(month)) >= (2021, 1):
            print("The date is from all of 2021 to November 2022.")            
            tpa_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpa/amsr_gw1/'
            period = '15'; centre = True; 
            tpa_folder = f"{tpa_data_dir}{year}/{month}/{day}/"
            #
            tpd_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpd/amsr_gw1/'
            swath_filename = f"{tpd_data_dir}{year}/{month}/{day}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
        else:
            print("The date is before 2021.")
            #/lustre/storeB/project/fou/fd/project/sicci/cci3y3/tpd/amsr_gw1
            #/lustre/storeB/project/fou/fd/project/sicci/cci3y3/tpa/amsr_gw1 

        # Missing TPD or TPA data
        if current_date.strftime('%Y%m%d') == '20231124' :
            swath_filename = '/lustre/storeB/users/atlems/marina/tpd/dyn_tp_nh_amsr_gw1_ucorr_20231124_oneday.nc'
        elif current_date.strftime('%Y%m%d') in ['20230520', '20230521']:
            swath_filename = f"{tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_20230519_oneday.nc"
            search_closest=True
        elif current_date.strftime('%Y%m%d') == '20241003':
            swath_filename = f"{tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_20241002_oneday.nc"
        elif current_date.strftime('%Y%m%d') == '20240718':
            swath_filename = f"{tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_20240717_oneday.nc"
    
        dyn_algos_return = read_dyn_algos_v2(sat_id, tpa_folder, corr, period, current_date, centre, algorithms, ignore, search_closest) #yesterday)    
                
        #f"{config.tpd_data_dir}dyn_tp_nh_amsr_gw1_ucorr_{current_date_str}_oneday.nc" 
        swath_file = io_handler.SwathFactory.get_swath(sat_id, swath_filename)
        list_tpd_files.append(swath_filename) # TPD file list to be used in compute_coeffs function
        
        all_chns = dict()
        for algos in list(dyn_algos_return.keys()):
            # query specifications of the algorithm in terms of ChannelTransforms and BlendFunction
            algo_nh = dyn_algos_return[algos]['nh']
            # extract a uniqued list of needed channels
            all_chns[algos] = algo_nh.bci.channels
        # Partition the swath in NH and SH FoV's                
        tb_hemis = dict()
        for ch in all_chns[algos]:
            #print(list(tb_hemis.keys()))
            for hm in ('n'):
                if ( not hm in list(tb_hemis.keys()) ):
                    tb_hemis[hm] = dict()
            tb_hemis[hm][ch] = (swath_file.read('cice_' + ch))
            
        # DAL computation
        _, _, dal_norm, _ = algo_nh.compute_sic(tb_hemis['n'], dal_type = 'normalized'); 
        _, _, dal_owf, _ = algo_nh.compute_sic(tb_hemis['n'], dal_type = 'owf'); 
        
        if dal_norm.shape[1] < m_dal_norm.shape[1] :
            # Fill with dal_norm values and NaN for the remaining columns if necessary
            m_dal_norm[i + (interval - 1), :dal_norm.shape[1]] = dal_norm[0, :]
            m_dal_norm[i + (interval - 1), dal_norm.shape[1]:] = np.nan  # Fill the rest with NaN
            #
            m_dal_owf[i + (interval - 1), :dal_owf.shape[1]] = dal_owf[0, :]
            m_dal_owf[i + (interval - 1), dal_owf.shape[1]:] = np.nan  # Fill the rest with NaN
        else :
            m_dal_norm[i + (interval - 1), :] = dal_norm[0, :]; # Normalized DAL
            m_dal_owf[i + (interval - 1), :] = dal_owf[0, :] # OWF DAL
        
        
    return m_dal_norm, m_dal_owf, list_tpd_files


def compute_coeffs(date, dal_norm, list_tpd_files, size_tpd_data, coeffs_filename) :
    
    '''
    This function computes the coefficients of a 2D-plane: Emissivity = 2Dplane(DAL, T2M)
    
    Input
        date: string format (ex: '20240101')
        list_tpd_files: list of TPD files
    
    Output
        a text file containing the 2D-plane coefficients is produced
    '''
    
    amsr2_eia = 55
    
    # Emissivity and T2M arrays to be filled with TPD data
    m_em = np.zeros((len(channels), len(list_tpd_files), size_tpd_data));
    m_t2m = np.zeros((len(list_tpd_files), size_tpd_data));
    
    # Loop that computes Effective Emissivity data to be saved in m_em and reads T2M data to be saved in m_t2m    
    for ifile, file_tpd in enumerate(list_tpd_files):                
        v, w, l, t, tb_obs = read_era5(file_tpd) 
        
        if t.shape[0] < m_t2m.shape[1] :
            print(t.shape[0], m_t2m.shape[1])
            m_t2m[ifile, :t.shape[0]] = t
            m_t2m[ifile, t.shape[0]:] = np.nan
            for ich, ch in enumerate(channels) : 
                m_em[ich, ifile, :t.shape[0]] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])
                m_em[ich, ifile, t.shape[0]:] = np.nan
        else :
            m_t2m[ifile, :] = t
            for ich, ch in enumerate(channels) : 
                m_em[ich, ifile, :] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])        
        
    # Computation of coefficients for 2D-plane corresponding to Emissivity = plane(DAL, T2M)    
    m_a1, m_a2, m_c = [np.zeros(len(channels)) for _ in range(3)]
    for ich in range(0, len(channels)) :
        x1_nan, y1_nan, z1_nan = m_t2m.flatten(), dal_norm.flatten(), m_em[ich, :].flatten() # Z = a1*X + a2*Y + c
        # Create a mask where none of the values are NaN
        mask = ~np.isnan(x1_nan) & ~np.isnan(y1_nan) & ~np.isnan(z1_nan)
        # Filter out the NaN values using the mask
        x1 = x1_nan[mask]
        y1 = y1_nan[mask]
        z1 = z1_nan[mask]
        
        X_data = np.array([x1, y1]).T; Y_data = z1
        reg = linear_model.LinearRegression().fit(X_data, Y_data) 
        #print("coefficients of equation of plane, (a1, a2): ", reg.coef_)
        #print("value of intercept, c:", reg.intercept_)
        m_a1[ich] = reg.coef_[0]; m_a2[ich] = reg.coef_[1]; m_c[ich] = reg.intercept_
        
    # Writing 2D plane coefficients to a CSV file
    with open(coeffs_filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'ch', 'a1', 'a2', 'c'])
        for ich, ch in enumerate(channels):
            row = [date if ich == 0 else '', ch, m_a1[ich], m_a2[ich], m_c[ich]]
            writer.writerow(row)  


def read_era5(filename, type_file = 'tpd', noise = False) :

    '''
    This function reads ERA5 and AMSR2 data projected in the model grid
    
    Input
        filename: netCDF file containing the data, type_file can be tpd or amsr2
    
    Output
        v: array, water vapor; unity: [mm]
        w: array, wind speed; unity: [m/s]        
        l: array, liquid water; unity: [mm]
        t: array, air temperature; unity: [K]        
        tb: dictionary, AMSR2 TBs; unity: [K]
        sic: array, AMSR2 sea ice concentration; unity: []
        dal: array, distance along the line; unity: [K]
        eia: AMSR2 inclination angle; unity: []
    '''

    era5_data = xr.open_dataset(filename)
    tb_sat = {}

    def replace_invalid_values(data, invalid_value = -1.e+10):
        data[data == invalid_value] = np.nan
        return data

    if type_file == 'tpd':
        variables = ("cice_tcwv", "cice_wind_speed", "cice_tclv", "cice_air_temp")
        v, w, l, t = [replace_invalid_values(era5_data[var][0, :]) for var in variables]

        for ch in channels:
            tb_sat[ch] = replace_invalid_values(era5_data[f"cice_{ch}"][0, :])

        return v, w, l, t, tb_sat

    elif type_file == 'amsr2':
        variables = ('tcwv@tb37', 'wind_speed@tb37', 'tclw@tb37', 'air_temp@tb37', 'eia')
        v, w, l, t, eia = [era5_data[var][0, :].isel(yc = slice(None, None, -1)) for var in variables]

        clim_path = '/lustre/storeB/project/fou/fd/project/acciberg/marina/climatology/'

        if date[0:4] in ('2023', '2024') :
            v, w, l, t, eia = [era5_data[var][0, :].isel(yc = slice(None, None, -1)) for var in variables]
            dal = (era5_data.dal_SICCI3LF_corrSICCI3LF[0, :] / 100).isel(yc = slice(None, None, -1))
            sic_sat = (era5_data.ct_SICCI3LF_corrSICCI3LF[0, :] / 100).clip(0, 1).isel(yc = slice(None, None, -1))
            
            # Filter OW in SIC
            owf_sat = (era5_data.owf_SICCI3LF_corrSICCI3LF[0, :].isel(yc = slice(None, None, -1)))
            sic_sat.data[owf_sat.data == 1] = 0.0
            
            # Climatology file
            clim_file = f'{clim_path}LandOceanLakeMask_nh_ease2-250_resampled_normal_grid.nc'
            data_clim = xr.open_dataset(f'{clim_file}', decode_cf = False)['climatology'][int(date[4:6]) - 1, :]
            sic_sat.data[data_clim.data == 0.0] = 0.0
            for ch in channels:
                tb_sat[ch] = era5_data[ch][0, :].isel(yc = slice(None, None, -1))

        elif '2021' in date[0:4] :
            v, w, l, t, eia = [era5_data[var][0, :] for var in variables]
            dal = (era5_data.dal_SICCI3LF_corrSICCI3LF[0, :] / 100)
            sic_sat = (era5_data.ct_SICCI3LF_corrSICCI3LF[0, :] / 100).clip(0, 1)
            
            # Filter OW in SIC
            owf_sat = (era5_data.owf_SICCI3LF_corrSICCI3LF[0, :]); sic_sat.data[owf_sat.data == 1] = 0.0
            
            # Climatology file
            clim_file = f'{clim_path}LandOceanLakeMask_nh_ease2-250_resampled.nc'
            data_clim = xr.open_dataset(f'{clim_file}', decode_cf = False)['climatology'][int(date[4:6]) - 1, :]
            sic_sat.data[data_clim.data == 0.0] = 0.0
            for ch in channels:
                tb_sat[ch] = era5_data[ch][0, :]

        if noise:
            w += np.random.normal(0, 1, np.shape(w))  # mean, std, size
            t += np.random.normal(0, 1.5, np.shape(t))  # mean, std, size

        return v, w, l, t, tb_sat, sic_sat, dal, eia


