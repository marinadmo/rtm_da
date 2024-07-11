import numpy as np
import netCDF4 as nc
import xarray as xr
import pandas as pd
import glob, os, sys, csv
from datetime import datetime, timedelta
from sklearn import linear_model

########## Import python functions for DAL computation
from dyn_algos_marina import read_dyn_algos
import io_handler
########## Import RTM python code
import rtm_amsr_fcts as rtm_amsr



class globalVars():
    channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
    Nens = 10 # Model ensemble size
    mask_file = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/2024/01/07/20240107_dm-metno-MODEL-topaz5-ARC-b20231229-fv02.0_mem005.nc' # File used for masking RTM output
    model = 'topaz' # Model used for the RTM simulation
    size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000

    
    
def compute_dal(date, tpd_dir, json_dir) :    
    
    '''
    This function computes Distance Along the Line (DAL, Lavergne et al., 2019) from observed TBs (19v, 37h, 37v)
    
    Input
        date: string format (ex: '20240101')
        tpd_dir: directory of TPD (Tie Point) files 
        json_dir: directory of JSON files
    
    Output
        DAL: DAL (normalized or owf); unity: Kelvin    
    '''  
    
    interval = 7 # DAL is computed for a period of 7 days    
    m_dal_norm, m_dal_owf = [np.zeros((interval, globalVars.size_tpd_data)) for _ in range(2)]
    list_tpd_files = []  
    
    # Convert the string to a datetime object
    date_obj = datetime.strptime(date, '%Y%m%d')
    
    # Loop through the interval of 7 days
    for i in range(-(interval - 1), 1):
         
        current_date = date_obj + timedelta(days = i); current_date_str = current_date.strftime('%Y%m%d')           
        
        # TPA files (JSON files)
        sat_id = 'amsr_gw1'; corr = 'ucorr'; period = '7'
        centre = False; algorithms = ['SICCI3LF']; ignore = False; yesterday = False
        tpa_file = f"{json_dir}{current_date.strftime('%Y')}/{current_date.strftime('%m')}/"
        dyn_algos_return = read_dyn_algos(sat_id, tpa_file, corr, period, current_date, centre, algorithms, ignore, yesterday)
        # TPD files
        swath_filename = f"{tpd_dir}dyn_tp_nh_amsr_gw1_ucorr_{current_date_str}_oneday.nc" 
        swath_file = io_handler.SwathFactory.get_swath(sat_id, swath_filename)
        list_tpd_files.append(swath_filename) # TPD file list to be used in compute_coeffs function        
        print('TPA file: ', tpa_file)
        print('TPD file: ', swath_filename)
        
        all_chns = dict()
        for algos in list(dyn_algos_return.keys()):
            # query specifications of the algorithm in terms of ChannelTransforms and BlendFunction
            algo_nh = dyn_algos_return[algos]['nh']
            # extract a uniqued list of needed channels
            all_chns[algos] = algo_nh.bci.channels
        # Partition the swath in NH and SH FoV's                
        tb_hemis = dict()
        for ch in all_chns[algos]:
            for hm in ('n'):
                if ( not hm in list(tb_hemis.keys()) ):
                    tb_hemis[hm] = dict()
            tb_hemis[hm][ch] = (swath_file.read('cice_' + ch))
            
        # DAL computation
        _, _, dal_norm, _ = algo_nh.compute_sic(tb_hemis['n'], dal_type = 'normalized'); m_dal_norm[i + (interval - 1), :] = dal_norm[0, :]; # Normalized DAL
        _, _, dal_owf, _ = algo_nh.compute_sic(tb_hemis['n'], dal_type = 'owf');  m_dal_owf[i + (interval - 1), :] = dal_owf[0, :] # OWF DAL
        
    return m_dal_norm, m_dal_owf, list_tpd_files



def compute_coeffs(date, dal_norm, list_tpd_files, coeffs_filename) :
    
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
    m_em = np.zeros((len(globalVars.channels), len(list_tpd_files), globalVars.size_tpd_data));
    m_t2m = np.zeros((len(list_tpd_files), globalVars.size_tpd_data));
    
    # Loop that computes Effective Emissivity data to be saved in m_em and reads T2M data to be saved in m_t2m    
    for ifile, file_tpd in enumerate(list_tpd_files):                
        v, w, l, t, tb_obs = read_era5(file_tpd) 
        m_t2m[ifile, :] = t
        for ich, ch in enumerate(globalVars.channels) : 
            m_em[ich, ifile, :] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])        
        
    # Computation of coefficients for 2D-plane corresponding to Emissivity = plane(DAL, T2M)    
    m_a1, m_a2, m_c = [np.zeros(len(globalVars.channels)) for _ in range(3)]
    for ich in range(0, len(globalVars.channels)) :
        x1, y1, z1 = m_t2m.flatten(), dal_norm.flatten(), m_em[ich, :].flatten() # Z = a1*X + a2*Y + c
        X_data = np.array([x1, y1]).T; Y_data = z1
        reg = linear_model.LinearRegression().fit(X_data, Y_data) 
        print('Channel :', globalVars.channels[ich])
        print("coefficients of equation of plane, (a1, a2): ", reg.coef_)
        print("value of intercept, c:", reg.intercept_)
        m_a1[ich] = reg.coef_[0]; m_a2[ich] = reg.coef_[1]; m_c[ich] = reg.intercept_
        
    # Writing 2D plane coefficients to a CSV file
    with open(coeffs_filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'ch', 'a1', 'a2', 'c'])
        for ich, ch in enumerate(globalVars.channels):
            row = [date if ich == 0 else '', ch[2::], m_a1[ich], m_a2[ich], m_c[ich]]
            writer.writerow(row)  
            
            

def read_csv_coefficients_plan(coeffs_filename, date) :
        
    '''
    This function reads the coefficients of a 2D-plane: Emissivity = plane(DAL, T2M)
    
    Input
        date: string format (ex: '20240101')
    
    Output
        coeffs: the 2D-plane coefficients
    '''
    
    df = pd.read_csv(coeffs_filename) 
    
    index = df[df['date'] == float(date)].index.tolist()
    if len(index) > 1 : 
        return 'The date appears more than once in the csv file, check!'
    
    m_a1, m_a2, m_c = ([] for _ in range(3)); coeffs = dict()
    for ich, ch in enumerate(globalVars.channels) :
        coeffs[ch[2::]] = dict()
        m_a1.append(df.iloc[index[0] + ich]['a1'])
        coeffs[ch[2::]]['a1'] = df.iloc[index[0] + ich]['a1']
        m_a2.append(df.iloc[index[0] + ich]['a2'])
        coeffs[ch[2::]]['a2'] = df.iloc[index[0] + ich]['a2']
        m_c.append(df.iloc[index[0] + ich]['c'])
        coeffs[ch[2::]]['c'] = df.iloc[index[0] + ich]['c']
        
    return coeffs 



def read_era5(filename, type_file = 'tpd', noise = False) :
    
    '''
    This function reads ERA5 and AMSR2 data projected in the model grid
    
    Input
        filename: netCDF file containing the data, type_file can be tpd or amsr2
    
    Output
        v: array, water vapor; unity:
        w: array, wind speed; unity: m/s        
        l: array, liquid water; unity:
        t: array, air temperature; unity: Kelvin        
        tb: dictionary, AMSR2 TBs; unity: Kelvin
        sic: array, AMSR2 sea ice concentration; unity: []
        dal: array, distance along the line; unity: Kelvin
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
        
        for ch in globalVars.channels:
            tb_sat[ch] = replace_invalid_values(era5_data[f"cice_{ch}"][0, :])
        
        return v, w, l, t, tb_sat
    
    elif type_file == 'amsr2':
        variables = ('tcwv@tb37', 'wind_speed@tb37', 'tclw@tb37', 'air_temp@tb37', 'eia')
        v, w, l, t, eia = [era5_data[var][0, :].isel(yc = slice(None, None, -1)) for var in variables]
        
        dal = (era5_data.dal_SICCI3LF_corrSICCI3LF[0, :] / 100).isel(yc = slice(None, None, -1))
        sic_sat = (era5_data.ct_SICCI3LF_corrSICCI3LF[0, :] / 100).clip(0, 1).isel(yc = slice(None, None, -1))
        
        for ch in globalVars.channels:
            tb_sat[ch] = era5_data[ch][0, :].isel(yc = slice(None, None, -1))
        
        if noise:
            w += np.random.normal(0, 1, np.shape(w))  # mean, std, size
            t += np.random.normal(0, 1.5, np.shape(t))  # mean, std, size
            
        return v, w, l, t, tb_sat, sic_sat, dal, eia
    
    
    
def read_amsr2(filename):
    
    '''
    This function reads AMSR2 data projected in the model grid
    
    Input 
        filename: AMSR2 netCDF file 
    
    Output
        tb: dictionary of AMSR2 TBs; unity: Kelvin        
        sic: array of AMSR2 sea ice concentration; unity: []
    '''
    
    sat_data = xr.open_dataset(filename) 
    sic_sat = ((sat_data.ct_SICCI3LF_corrSICCI3LF[0, :]/100).clip(0, 1)).isel(yc = slice(None, None, -1))
    tb_sat = dict();
    for ch in globalVars.channels :  
        tb_sat[ch] = dict()
        tb_sat[ch] = (sat_data[ch][0, :]).isel(yc = slice(None, None, -1));    
    return tb_sat, sic_sat



def run_rtm(date, sat_dir, model_dir, days_forecast = 10, version = 0, coeffs_filename = None):
    
    '''
    This function reads the coefficients of a 2D-plane: Emissivity = plane(DAL, T2M)
    
    Input
        date: string format (ex: '20240101')
        the 2D-plane coefficients file
        the version of the RTM: 
            version 0 corresponds to the original RTM
            version 1 correspond to the updated RTM
    Output
        tb_rtm: dictionnary of RTM TBs; unity: Kelvin
        sic_model: dictionnary of model SIC; unity: []
        em_rtm if version=1; unity:
    '''    
    
    # Define dates needed to select model file
    if days_forecast == 1 : datei = date;
    else :
        # Convert the string to a datetime object and substract days  
        date2 = datetime.strptime(date, '%Y%m%d') - timedelta(days = days_forecast - 1)
        # Convert datetime object back to string
        datei = date2.strftime("%Y%m%d") 
    print(days_forecast, datei)

    if version == 1 : 
        coeffs = read_csv_coefficients_plan(coeffs_filename, date) 
        
    # Open satellite file    
    files_sat = glob.glob(os.path.join(sat_dir, date[0:4], f"*{date}*nc")) 
    # Read ERA5 and AMSR2 variables from file
    v, w, l, t, tbs, sic, dal, eia = read_era5(files_sat[0], type_file = 'amsr2', noise = False)
    
    sic_model = dict(); tb_rtm = dict(); em_rtm = dict();
    
    for ch in globalVars.channels :  
        tb_rtm[ch] = dict(); em_rtm[ch] = dict();
        print('Computing TBs for channel ', ch[2::])
        for imem in range(1, globalVars.Nens + 1) :            
            mem = 'mem' + "{0:03}".format(imem)
            print('Computing TBs for member ', mem)
            sic_model[mem] = dict()
            
            # Ensure model_dir does not end with a slash
            model_dir = model_dir.rstrip('/')
            # Get the list of matching files
            files_model = glob.glob(os.path.join(model_dir, date[0:4], date[4:6], date[6:8], f"*{date}*{datei}*{mem}*nc"))
            
            model_data = xr.open_dataset(files_model[0]) # Open topaz model files
            
            if imem == 1 :
                model_mask = model_data.model_depth.to_masked_array().mask
                        
            sic_model[mem] = model_data.siconc[0, :].data
            
            if version == 0 : # observed_tb function
                tb_rtm_output = rtm_amsr.observed_tb(v.data, w.data, l.data, t.data, model_data.siconc[0, :].data, eia.data, ch[2::]) 
                tb_rtm[ch][mem] = np.ma.masked_array(tb_rtm_output, mask = model_mask)                
            elif version == 1 : # simulated_tb_v03 function
                tb_rtm_output, em_rtm_output = rtm_amsr.simulated_tb_v03(v.data, w.data, l.data, t.data, dal.data, 
                                                                                     model_data.siconc[0, :].data, eia, ch[2::], dict_coeffs = coeffs)
                tb_rtm[ch][mem] = np.ma.masked_array(tb_rtm_output, mask = model_mask)
                em_rtm[ch][mem] = np.ma.masked_array(em_rtm_output, mask = model_mask)
                
    if version == 0 : return tb_rtm, sic_model
    elif version == 1 : return tb_rtm, em_rtm, sic_model



def save_rtm_tbs(date, tb_rtm, newfiles_dir) :   
    
    '''  
    This function saves the RTM TBs in netCDF files
    
    Input
        date: string format (ex: '20240101')
        tb_rtm: dictionnary of RTM TBs; unity: K        
        newfiles_dir: directory where files are saved
        
    Output
        netCDF files are created and RTM TBs data written
    '''    
   
    for imem in range(1, globalVars.Nens + 1) :        
        mem = 'mem' + "{0:03}".format(imem)
        new_file = f"{newfiles_dir}{globalVars.model}_tb_{date}_{mem}.nc"
        
        with nc.Dataset(new_file, mode = "w") as tb_data, nc.Dataset(globalVars.mask_file, mode = "r") as model_data :
            tb_data.createDimension("time", size = 1)
            tb_data.createDimension("x", size = model_data.dimensions["x"].size)
            tb_data.createDimension("y", size = model_data.dimensions["y"].size)

            y = tb_data.createVariable("y", 'f4', dimensions = ("y"))
            y[:] = np.arange(0, y.size)
            x = tb_data.createVariable("x", float, dimensions = ("x"))
            x[:] = np.arange(0, x.size)
            
            for var in globalVars.channels :
                tb_var = tb_data.createVariable(var, float, dimensions = ("time", "y", "x"))
                tb_var[:] = tb_rtm[var][mem].data
