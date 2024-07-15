# Code that generates model mask from RTM TBs

from .main_imports import *

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def generate_mask() :

    # Reference model mask file (longitude, latitude, model_depth, model_mask)
    mask_file = f"{config.main_data_dir}marina/enkf_exps/conf/mask_topaz5.nc"
    model_data = xr.open_dataset(mask_file)
    
    model_file = f"{config.main_data_dir}marina/enkf_exps/exp_sic_10fdays/{config.date}/ensb/mem001_tb19v.nc"
    mask_tbs = xr.open_dataset(model_file)
    new_mask = mask_tbs.tb19v[0, :].to_masked_array().mask.copy()
    new_mask[mask_tbs.tb19v[0, :].to_masked_array().mask == 0] = 1
    new_mask[mask_tbs.tb19v[0, :].to_masked_array().mask == 1] = 0

    new_mask_file = f"{config.main_data_dir}marina/enkf_exps/conf/mask_topaz_tbs.nc"
    cmd('rm ' + new_mask_file)

    ds = xr.Dataset(

        {"longitude": (("y", "x"), model_data.longitude.data),
        "latitude": (("y", "x"), model_data.latitude.data),
        "model_mask": (("y", "x"), new_mask), #*5),
        "model_depth": (("y", "x"), model_data.model_depth.data),
        "zt": (("z"), [1, 10, 20, 40, 50])},

        coords = {

            "x": model_data.x.data,

            "y": model_data.y.data,

            "z": np.arange(0, 5, 1),

        },

    )

    ds.to_netcdf(new_mask_file)

#generate_mask('20240113')
