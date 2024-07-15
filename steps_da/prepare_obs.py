#Preparation of observations for SYN and ASYN anlysis (L4 and L3 SIC obs)

from .main_imports import *

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def prep_topaz() :
	
    # Topaz5 data
    model_file = f"{config.model_data_dir}/{config.date[0:4]}/{config.date[4:6]}/{config.date[6:8]}/{config.date}_dm-metno-MODEL-topaz5-ARC-b{config.date}-fv02.0_mem010.nc"
    print(model_file)
    model_data = xr.open_dataset(model_file)
    mask = model_data.model_depth.to_masked_array().mask

    obs_dir = f"{config.main_data_dir}marina/enkf_exps/observations/"
    sat_dir = f"{config.main_data_dir}atlems/topaz_l3/"
    
    files_sat = glob.glob(f"{sat_dir}{config.date[0:4]}/*{config.date}*nc")[0]
    print(files_sat)
    new_obs_file = f"{obs_dir}amsr2_topaz_obs_{config.date}.nc"
    cmd('rm ' + new_obs_file)
    tmpfile = obs_dir + '/tmp/tmp.nc'
    tmpfile2 = obs_dir + '/tmp/tmp2.nc'
    cmd('ncks -v lat,lon,ct_SICCI3LF_corrSICCI3LF,tb19v,tb19h,tb37v,tb37h ' + files_sat + ' ' + tmpfile )
    cmd('ncrename -v ct_SICCI3LF_corrSICCI3LF,ice_conc ' + tmpfile + ' ' + tmpfile2 )
    cmd('rm ' + tmpfile); 

    # New file
    sat_data = xr.open_dataset(tmpfile2)
    ds = xr.Dataset(

        {"longitude": (("y", "x"), sat_data.lon.isel(yc = slice(None, None, -1)).data),
        "latitude": (("y", "x"), sat_data.lat.isel(yc = slice(None, None, -1)).data),
        "ice_conc": (("time", "y", "x"), np.ma.masked_array(((sat_data.ice_conc[:]/100).clip(0, 1).isel(yc = slice(None, None, -1)).data), mask = mask)),
        "tb19v": (("time", "y", "x"), np.ma.masked_array((sat_data.tb19v.isel(yc = slice(None, None, -1)).data), mask = mask)),
        "tb19h": (("time", "y", "x"), np.ma.masked_array((sat_data.tb19h.isel(yc = slice(None, None, -1)).data), mask = mask)),
        "tb37v": (("time", "y", "x"), np.ma.masked_array((sat_data.tb37v.isel(yc = slice(None, None, -1)).data), mask = mask)),
        "tb37h": (("time", "y", "x"), np.ma.masked_array((sat_data.tb37h.isel(yc = slice(None, None, -1)).data), mask = mask))
            },

        coords = {

            "x": sat_data.xc.data[::-1],

            "y": sat_data.yc.data[::-1],

            "time": sat_data.time.data,

        },

    ) 

    ds.to_netcdf(new_obs_file)
    cmd('rm ' + tmpfile2)


#prep_topaz('20240112')
