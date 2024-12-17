# Code that generates model mask from RTM TBs

import sys, os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_imports import *
import config

def generate_mask() :

    var_depth = 'model_depth'
    dimx, dimy = 'x', 'y'
    
    # Reference model mask file (longitude, latitude, model_depth, model_mask)
    mask_file = f"{config.main_data_dir}marina/enkf_exps/conf/{config.date[0:4]}/topaz5_grid_{config.date[0:4]}.nc" #mask_topaz5.nc"
    model_data = xr.open_dataset(mask_file)
    print(np.shape(model_data[var_depth].data))
    
    model_file = f'{config.storage_dir}/ensb/mem001_tb19v.nc'
    mask_tbs = xr.open_dataset(model_file)
    new_mask = mask_tbs.tb19v[0, :].to_masked_array().mask.copy()
    new_mask[mask_tbs.tb19v[0, :].to_masked_array().mask == 0] = 1
    new_mask[mask_tbs.tb19v[0, :].to_masked_array().mask == 1] = 0    


    new_mask_file = f"{config.main_data_dir}marina/enkf_exps/conf/mask_topaz_tbs.nc"
    cmd('rm ' + new_mask_file)

    ds = xr.Dataset(

        {"longitude": ((dimy, dimx), model_data.longitude.data),
        "latitude": ((dimy, dimx), model_data.latitude.data),
        "model_mask": ((dimy, dimx), new_mask), #*5),
        "model_depth": ((dimy, dimx), model_data[var_depth].data),
        "zt": (("z"), [1, 10, 20, 40, 50])},

        coords = {

            "x": model_data[dimx].data,

            "y": model_data[dimy].data,

            "z": np.arange(0, 5, 1),

        },

    )

    ds.to_netcdf(new_mask_file)