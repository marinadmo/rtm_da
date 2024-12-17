import sys, os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from main_imports import *
import config


def read_obs() :

    sat_data = netCDF4.Dataset(f"{config.exps_dir}/observations/amsr2_topaz_obs_{config.date}.nc")
    sic_sat = sat_data['ice_conc'][0, :]
    
    return sic_sat

def read_obs_tb(var_tb = 0) :

    sat_data = netCDF4.Dataset(f"{config.exps_dir}/observations/amsr2_topaz_obs_{config.date}.nc")
    tb_sat = sat_data[config.channels[var_tb]][0, :]
    
    return tb_sat

def read_mask() :
    
    mask_data = xr.open_dataset(config.mask_file)['model_mask'][:]
    # Step 1: Invert the values (0 becomes 1, 1 becomes 0)
    mask_data_inverted = 1 - mask_data

    # Step 2: Convert to boolean (1 becomes True, 0 becomes False)
    mask_data_bool = mask_data_inverted.astype(bool)
    
    return mask_data_bool

def read_model_matrix(opt = 0, vari = 0) :

    sic_model_ens = dict()  
    for imem in range(1, config.Nens + 1) :
        mem = 'mem' + "{0:03}".format(imem)
        print(mem)
        
        if opt == 0 : filename = f"{config.storage_dir}/ensb/{mem}_{config.varnames[vari]}.nc"
        else : filename = f"{config.storage_dir}/ensa/{mem}_{config.varnames[vari]}.nc.analysis" 
        print(filename)
        # Open topaz model files
        model_data = xr.open_dataset(glob.glob(filename)[0])
        sic_model_ens[mem] = model_data[config.varnames[vari]][0, :]
    
    return sic_model_ens
    
def read_model_matrix_tb(var_tb = 0) :
    
    tb_model_ens = dict()  
    for imem in range(1, config.Nens + 1) :
        mem = 'mem' + "{0:03}".format(imem)
        print(mem)
        
        # Background Tbs
        filename = f"{config.rtm_tbs_dir}/topaz_tb_{config.date}_{mem}.nc" 
        # Open topaz model files
        print(glob.glob(filename)[0])
        model_data = xr.open_dataset(glob.glob(filename)[0])
        tb_model_ens[mem] = model_data[config.channels[var_tb]][0, :]
    
    return tb_model_ens


def lon_lat() :
    
    dataset = xr.open_dataset(config.mask_file)
    lon, lat = dataset.longitude.data, dataset.latitude.data
    
    return lon, lat