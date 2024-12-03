#Preparation of observations for SYN and ASYN anlysis (L4 and L3 SIC obs)

# Import python functions for DAL computation
from .dyn_algos_marina import read_dyn_algos
import io_handler

# Import RTM python code
import rtm_dal.rtm_amsr_fcts as rtm_amsr

from sklearn import linear_model
import pandas as pd
import csv

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../steps_da')))
from main_imports import *

def compute_dal() : #date, tpd_dir, json_dir) :    
    
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
    m_dal_norm, m_dal_owf = [np.zeros((interval, config.size_tpd_data)) for _ in range(2)]
    list_tpd_files = []  
    
    # Convert the string to a datetime object
    date_obj = datetime.datetime.strptime(config.date, '%Y%m%d')
    
    # Loop through the interval of 7 days
    for i in range(-(interval - 1), 1) :
         
        current_date = date_obj + datetime.timedelta(days = i); current_date_str = current_date.strftime('%Y%m%d')           
        year = f"{current_date.year:04}"
        month = f"{current_date.month:02}"
        day = f"{current_date.day:02}"
        
        # TPA files (JSON files)
        sat_id = 'amsr_gw1'; corr = 'ucorr';
        algorithms = ['SICCI3LF']; ignore = False; yesterday = False
        if '2024' in config.date[0:4] :
            period = '7'; centre = False; 
            dyn_algos_return = read_dyn_algos(sat_id, f"{config.tpa_data_dir}{year}/{month}/", corr, period, current_date, centre, algorithms, ignore, yesterday)     
        elif '2021' in config.date[0:4] :
            period = '15'; centre = True; 
            dyn_algos_return = read_dyn_algos(sat_id, f"{config.tpa_data_dir}{year}/{month}/{day}/", corr, period, current_date, centre, algorithms, ignore, yesterday)    
            
        
        # TPD files
        if '2024' in config.date[0:4] :
            swath_filename = f"{config.tpd_data_dir}{year}/{month}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
        elif '2021' in config.date[0:4] :
            swath_filename = f"{config.tpd_data_dir}{year}/{month}/{day}/dyn_tp_nh_amsr_gw1_ucorr_{year}{month}{day}_oneday.nc"
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

def compute_dal_old() : #date, tpd_dir, json_dir) :    
    
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
    m_dal_norm, m_dal_owf = [np.zeros((interval, config.size_tpd_data)) for _ in range(2)]
    list_tpd_files = []  
    
    # Convert the string to a datetime object
    date_obj = datetime.datetime.strptime(config.date, '%Y%m%d')
    
    # Loop through the interval of 7 days
    for i in range(-(interval - 1), 1):
         
        current_date = date_obj + datetime.timedelta(days = i); current_date_str = current_date.strftime('%Y%m%d')           
        
        # TPA files (JSON files)
        sat_id = 'amsr_gw1'; corr = 'ucorr'; period = '7'
        centre = False; algorithms = ['SICCI3LF']; ignore = False; yesterday = False
        tpa_file = f"{config.tpa_data_dir}{current_date.strftime('%Y')}/{current_date.strftime('%m')}/"
        #f"{json_dir}{current_date.strftime('%Y')}/{current_date.strftime('%m')}/"
        dyn_algos_return = read_dyn_algos(sat_id, tpa_file, corr, period, current_date, centre, algorithms, ignore, yesterday)
        # TPD files
        swath_filename = f"{config.tpd_data_dir}{current_date.strftime('%Y')}/{current_date.strftime('%m')}/dyn_tp_nh_amsr_gw1_ucorr_{current_date_str}_oneday.nc" 
        #f"{tpd_dir}dyn_tp_nh_amsr_gw1_ucorr_{current_date_str}_oneday.nc" 
        swath_file = io_handler.SwathFactory.get_swath(sat_id, swath_filename)
        list_tpd_files.append(swath_filename) # TPD file list to be used in compute_coeffs function        
        #print('TPA file: ', tpa_file)
        #print('TPD file: ', swath_filename)
        
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


def compute_coeffs(dal_norm, list_tpd_files) : #date, dal_norm, list_tpd_files, coeffs_filename) :
    
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
    m_em = np.zeros((len(config.channels), len(list_tpd_files), config.size_tpd_data));
    m_t2m = np.zeros((len(list_tpd_files), config.size_tpd_data));
    
    # Loop that computes Effective Emissivity data to be saved in m_em and reads T2M data to be saved in m_t2m    
    for ifile, file_tpd in enumerate(list_tpd_files):                
        v, w, l, t, tb_obs = read_era5(file_tpd) 
        
        if t.shape[0] < m_t2m.shape[1] :
            print(t.shape[0], m_t2m.shape[1])
            m_t2m[ifile, :t.shape[0]] = t
            m_t2m[ifile, t.shape[0]:] = np.nan
            for ich, ch in enumerate(config.channels) : 
                m_em[ich, ifile, :t.shape[0]] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])
                m_em[ich, ifile, t.shape[0]:] = np.nan
        else :
            m_t2m[ifile, :] = t
            for ich, ch in enumerate(config.channels) : 
                m_em[ich, ifile, :] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])        
        
    # Computation of coefficients for 2D-plane corresponding to Emissivity = plane(DAL, T2M)    
    m_a1, m_a2, m_c = [np.zeros(len(config.channels)) for _ in range(3)]
    for ich in range(0, len(config.channels)) :
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
    with open(config.coeffs_filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'ch', 'a1', 'a2', 'c'])
        for ich, ch in enumerate(config.channels):
            row = [config.date if ich == 0 else '', ch, m_a1[ich], m_a2[ich], m_c[ich]]
            writer.writerow(row)  
            
def compute_coeffs_old(dal_norm, list_tpd_files) : #date, dal_norm, list_tpd_files, coeffs_filename) :
    
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
    m_em = np.zeros((len(config.channels), len(list_tpd_files), config.size_tpd_data));
    m_t2m = np.zeros((len(list_tpd_files), config.size_tpd_data));
    
    # Loop that computes Effective Emissivity data to be saved in m_em and reads T2M data to be saved in m_t2m    
    for ifile, file_tpd in enumerate(list_tpd_files):                
        v, w, l, t, tb_obs = read_era5(file_tpd) 
        m_t2m[ifile, :] = t
        for ich, ch in enumerate(config.channels) : 
            m_em[ich, ifile, :] = rtm_amsr.calc_emissivity(v, l, t, tb_obs[ch].data[:], amsr2_eia, ch[2::])        
        
    # Computation of coefficients for 2D-plane corresponding to Emissivity = plane(DAL, T2M)    
    m_a1, m_a2, m_c = [np.zeros(len(config.channels)) for _ in range(3)]
    for ich in range(0, len(config.channels)) :
        x1, y1, z1 = m_t2m.flatten(), dal_norm.flatten(), m_em[ich, :].flatten() # Z = a1*X + a2*Y + c
        X_data = np.array([x1, y1]).T; Y_data = z1
        reg = linear_model.LinearRegression().fit(X_data, Y_data) 
        #print('Channel :', config.channels[ich])
        #print("coefficients of equation of plane, (a1, a2): ", reg.coef_)
        #print("value of intercept, c:", reg.intercept_)
        m_a1[ich] = reg.coef_[0]; m_a2[ich] = reg.coef_[1]; m_c[ich] = reg.intercept_
        
    # Writing 2D plane coefficients to a CSV file
    with open(config.coeffs_filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'ch', 'a1', 'a2', 'c'])
        for ich, ch in enumerate(config.channels):
            row = [config.date if ich == 0 else '', ch[2::], m_a1[ich], m_a2[ich], m_c[ich]]
            writer.writerow(row)  
            
            

def read_csv_coefficients_plan() : #coeffs_filename, date) :
        
    '''
    This function reads the coefficients of a 2D-plane: Emissivity = plane(DAL, T2M)
    
    Input
        date: string format (ex: '20240101')
    
    Output
        coeffs: the 2D-plane coefficients
    '''
    
    df = pd.read_csv(config.coeffs_filename) 
    
    index = df[df['date'] == float(config.date)].index.tolist()
    if len(index) > 1 : 
        return 'The date appears more than once in the csv file, check!'
    
    m_a1, m_a2, m_c = ([] for _ in range(3)); coeffs = dict()
    for ich, ch in enumerate(config.channels) :
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
        
        for ch in config.channels:
            tb_sat[ch] = replace_invalid_values(era5_data[f"cice_{ch}"][0, :])
        
        return v, w, l, t, tb_sat
    
    elif type_file == 'amsr2':
        variables = ('tcwv@tb37', 'wind_speed@tb37', 'tclw@tb37', 'air_temp@tb37', 'eia')
        v, w, l, t, eia = [era5_data[var][0, :].isel(yc = slice(None, None, -1)) for var in variables]
            
        if '2024' in config.date[0:4] :
            v, w, l, t, eia = [era5_data[var][0, :].isel(yc = slice(None, None, -1)) for var in variables]
            dal = (era5_data.dal_SICCI3LF_corrSICCI3LF[0, :] / 100).isel(yc = slice(None, None, -1))
            sic_sat = (era5_data.ct_SICCI3LF_corrSICCI3LF[0, :] / 100).clip(0, 1).isel(yc = slice(None, None, -1))  
            for ch in config.channels:
                tb_sat[ch] = era5_data[ch][0, :].isel(yc = slice(None, None, -1))
                
        elif '2021' in config.date[0:4] :
            v, w, l, t, eia = [era5_data[var][0, :] for var in variables]
            dal = (era5_data.dal_SICCI3LF_corrSICCI3LF[0, :] / 100)
            sic_sat = (era5_data.ct_SICCI3LF_corrSICCI3LF[0, :] / 100).clip(0, 1)            
            for ch in config.channels:
                tb_sat[ch] = era5_data[ch][0, :]
        
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
    
    sat_data = xr.open_dataset(file_sat) 
    tb_sat = dict();
    if '2024' in config.date[0:4] :
        sic_sat = ((sat_data.ct_SICCI3LF_corrSICCI3LF[0, :]/100).clip(0, 1)).isel(yc = slice(None, None, -1))        
        for ch in config.channels :  
            tb_sat[ch] = dict()
            tb_sat[ch] = (sat_data[ch][0, :]).isel(yc = slice(None, None, -1))   
    else :
        sic_sat = ((sat_data.ct_SICCI3LF_corrSICCI3LF[0, :]/100).clip(0, 1))
        for ch in config.channels :  
            tb_sat[ch] = dict()
            tb_sat[ch] = (sat_data[ch][0, :]) 
            
    return tb_sat, sic_sat


def run_rtm(version = 1, days_forecast = 10) : #date, sat_dir, model_dir, days_forecast = 10, version = 0, coeffs_filename = None):
    
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
    if '2024' in config.date[0:4] :
        # Define dates needed to select model file
        if days_forecast == 1 : datei = config.date;
        else :
            # Convert the string to a datetime object and substract days  
            date2 = datetime.strptime(config.date, '%Y%m%d') - timedelta(days = days_forecast - 1)
            # Convert datetime object back to string
            datei = date2.strftime("%Y%m%d") 
        #print(days_forecast, datei)

    if version == 1 : 
        coeffs = read_csv_coefficients_plan() #config.coeffs_filename, config.date) 
        
    # Open satellite file  
    print(os.path.join(config.sat_data_dir, f"*{config.date}*nc"))
    files_sat = glob.glob(os.path.join(config.sat_data_dir, f"*{config.date}*nc"))
    # Read ERA5 and AMSR2 variables from file
    v, w, l, t, tbs, sic, dal, eia = read_era5(files_sat[0], type_file = 'amsr2', noise = False)
    
    sic_model = dict(); tb_rtm = dict(); em_rtm = dict();
    
    for ch in config.channels :  
        tb_rtm[ch] = dict(); em_rtm[ch] = dict();
        #print('Computing Tbs for channel ', ch[2::])
        for imem in range(1, config.Nens + 1) :            
            mem = 'mem' + "{0:03}".format(imem)
            print(f'Computing Tbs for member {mem[3::]}, channel {ch[2::]}...')
            sic_model[mem] = dict()
            
            # Ensure model_dir does not end with a slash
            model_dir = config.model_data_dir.rstrip('/')
           
            # Get the list of matching files
            var_sic = config.varnames[0] 
            
            # Get the list of matching files
            if '2024' in config.date[0:4] :
                files_model = glob.glob(os.path.join(config.model_data_dir, config.date[0:4], config.date[4:6], config.date[6:8], f"*{config.date}*{datei}*{mem}*nc"))
                var_mask = 'model_depth'; #var_sic = 'siconc'                
            elif '2021' in config.date[0:4] :
                files_model = glob.glob(os.path.join(config.model_data_dir, mem, 'daily', f"iceh.{config.date[0:4]}-{config.date[4:6]}-{config.date[6:8]}.nc"))
                var_mask = 'aice_d'; #var_sic = 'aice_d'
                
            model_data = xr.open_dataset(files_model[0]) # Open topaz model files
            
            if imem == 1 : model_mask = model_data[var_mask].to_masked_array().mask
                        
            sic_model[mem] = model_data[var_sic][0, :].data
            
            if version == 0 : # observed_tb function
                tb_rtm_output = rtm_amsr.observed_tb(v.data, w.data, l.data, t.data, model_data[var_sic][0, :].data, eia.data, ch[2::]) 
                tb_rtm[ch][mem] = np.ma.masked_array(tb_rtm_output, mask = model_mask)                
            elif version == 1 : # simulated_tb_v03 function
                tb_rtm_output, em_rtm_output = rtm_amsr.simulated_tb_v03(v.data, w.data, l.data, t.data, dal.data, 
                                                                                     model_data[var_sic][0, :].data, eia, ch[2::], dict_coeffs = coeffs)
                tb_rtm[ch][mem] = np.ma.masked_array(tb_rtm_output, mask = model_mask)
                em_rtm[ch][mem] = np.ma.masked_array(em_rtm_output, mask = model_mask)
                
    if version == 0 : return tb_rtm, sic_model
    elif version == 1 : return tb_rtm, em_rtm, sic_model


def run_rtm_swaths(version = 1) :
    
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
    
    if version == 1 : 
        coeffs = read_csv_coefficients_plan() #config.coeffs_filename, config.date) 
        
    # Open satellite file  
    files_sat = glob.glob(os.path.join(config.sat_data_dir, f"*{config.date}*nc"))
    # Read ERA5 and AMSR2 variables from file
    v, w, l, t, tbs, sic, dal, eia = read_era5(files_sat[0], type_file = 'amsr2', noise = False)
    
    sic_model = dict(); tb_rtm = dict(); em_rtm = dict();
    
    for ch in config.channels :  
        tb_rtm[ch] = dict(); em_rtm[ch] = dict();
        #print('Computing Tbs for channel ', ch[2::])
        for imem in range(1, config.Nens + 1) :            
            mem = 'mem' + "{0:03}".format(imem)
            print(f'Computing Tbs for member {mem[3::]}, channel {ch[2::]}...')
            sic_model[mem] = dict()
            tb_rtm[ch][mem] = dict(); em_rtm[ch][mem] = dict()
            
            # Ensure model_dir does not end with a slash
            model_dir = config.model_data_dir.rstrip('/')
           
            # Get the list of matching files
            var_sic = 'aice' #config.varnames[0] 
            
            # Get the list of matching files
            files_model = glob.glob(os.path.join(config.model_data_dir, mem, 'hourly', f"iceh_concat.{config.date[0:4]}-{config.date[4:6]}-{config.date[6:8]}.nc"))
            var_mask = 'aice'; 
                
            model_data = xr.open_dataset(files_model[0]) # Open topaz model files
            
            if imem == 1 : model_mask = model_data[var_mask][0, :].to_masked_array().mask
            
            for it in range(0, np.shape(model_data[var_sic])[0]) :
                sic_model[mem] = model_data[var_sic][it, :].data

                if version == 0 : # observed_tb function
                    tb_rtm_output = rtm_amsr.observed_tb(v.data, w.data, l.data, t.data, model_data[var_sic][it, :].data, eia.data, ch[2::]) 
                    tb_rtm[ch][mem][it] = np.ma.masked_array(tb_rtm_output, mask = model_mask)                
                elif version == 1 : # simulated_tb_v03 function
                    tb_rtm_output, em_rtm_output = rtm_amsr.simulated_tb_v03(v.data, w.data, l.data, t.data, dal.data, 
                                                                                         model_data[var_sic][it, :].data, eia, ch[2::], dict_coeffs = coeffs)
                    tb_rtm[ch][mem][it] = np.ma.masked_array(tb_rtm_output, mask = model_mask)
                    em_rtm[ch][mem][it] = np.ma.masked_array(em_rtm_output, mask = model_mask)
                
    if version == 0 : return tb_rtm, sic_model
    elif version == 1 : return tb_rtm, em_rtm, sic_model


def save_rtm_tbs(tb_rtm, newfiles_dir, swaths = False) : #date, tb_rtm, newfiles_dir) :   
    
    '''  
    This function saves the RTM TBs in netCDF files
    
    Input
        date: string format (ex: '20240101')
        tb_rtm: dictionnary of RTM TBs; unity: K        
        newfiles_dir: directory where files are saved
        
    Output
        netCDF files are created and RTM TBs data written
    '''    

    if '2024' in config.date[0:4] : dimx, dimy = 'x', 'y'
    elif '2021' in config.date[0:4] : dimx, dimy = 'ni', 'nj'    
    
    if swaths : nsize = 23
    else : nsize = 1
    
    for imem in range(1, config.Nens + 1) :        
        mem = 'mem' + "{0:03}".format(imem)
        new_file = f"{newfiles_dir}{config.model}_tb_{config.date}_{mem}.nc"
        
        with netCDF4.Dataset(new_file, mode = "w") as tb_data, netCDF4.Dataset(config.mask_file, mode = "r") as model_data :
            tb_data.createDimension("time", size = nsize)
            tb_data.createDimension(dimx, size = model_data.dimensions[dimx].size)
            tb_data.createDimension(dimy, size = model_data.dimensions[dimy].size)

            y = tb_data.createVariable(dimy, 'f4', dimensions = (dimy))
            y[:] = np.arange(0, y.size)
            x = tb_data.createVariable(dimx, float, dimensions = (dimx))
            x[:] = np.arange(0, x.size)
            
            for var in config.channels :
                print(f'Saving {var} for member {mem[3::]}...')
                tb_var = tb_data.createVariable(var, float, dimensions = ("time", dimy, dimx))
                if not swaths :
                    tb_var[:] = tb_rtm[var][mem].data                
                elif swaths :
                    for it in range(0, nsize) :
                        tb_var[:] = tb_rtm[var][mem][it].data