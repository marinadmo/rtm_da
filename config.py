# config.py

#import os

# You can also add other configuration variables if needed
Nens = 10 # Model ensemble size
date = '20240118' # Date for DA analysis
fdays = 10 # Forecast day of the model background ensemble to be corrected (for TOPAZ is 10 days the maximum forecast length)
assim = 'exp_tb' # Assimilation of SIC (exp_sic) or TB (exp_tb)

# Define your paths here
main_data_dir = '/lustre/storeB/project/fou/fd/project/acciberg/' # acciberg data directory
model_data_dir = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/' # TOPAZ data directory
rtm_tbs_dir = f"{main_data_dir}marina/topaz_tbs/{date}/tbs_{fdays}fdays/" # RTM TOPAZ TBs directory
my_exp = f"{assim}_{fdays}fdays" # my experiment subfolder
storage_dir = f"{main_data_dir}marina/enkf_exps/{my_exp}/{date}/" # storage directory (input and output data to/from EnKF)
enkf_run_dir = f"/lustre/storeB/users/marinadm/enkf_run/acciberg/{assim}/" # EnKF run directory

