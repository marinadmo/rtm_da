# Produce RTM TBs
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

from rtm_dal.main_fcts_rtm_tbs import *

### Computation of DAL (Distance Along the Line) from TPD and TPA files
dal_norm, _, list_tpd_files = compute_dal() #config.date, config.tpd_data_dir, config.tpa_data_dir)

### Computation of 2D-plane coefficients. This 2D-plane is defined by the relation between Emissivity, DAL and T2M
#compute_coeffs(dal_norm, list_tpd_files) #config.date, dal_norm, list_tpd_files, config.coeffs_filename_date)

### Computation of RTM TBs
#res = run_rtm(version = 1) #config.date, config.sat_data_dir, config.model_data_dir, days_forecast = config.fdays, version = 1, coeffs_filename = config.coeffs_filename_date)

### Create directory where RTM TBs will be saved
#cmd('mkdir -p ' + f"{config.rtm_tbs_dir}{config.date}")
#cmd('mkdir -p ' + f"{config.rtm_tbs_dir}{config.date}/tbs_{config.fdays}fdays/")

### Create netCDF files containing RTM TBs and saved them in the previously defined directory
#save_rtm_tbs(res[0], f"{config.rtm_tbs_dir}{config.date}/tbs_{config.fdays}fdays/") #config.date, res[0], f"{rtm_tbs_dir}{date}/tbs_{fdays}fdays/")
