# config.py

###################################################################################################################################
##### Define configuration variables

# RTM and EnKF variables
Nens = 10 # Model ensemble size
date = '20240119' # Date for DA analysis
fdays = 10 # Forecast day of the model background ensemble to be corrected (for TOPAZ is 10 days the maximum forecast length)

# RTM variables
channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
model = 'topaz' # Model used for the RTM simulation
size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000

# EnKF variables
assim = 'exp_sic' # Assimilation of SIC (exp_sic) or TB (exp_tb)

###################################################################################################################################
##### Define data directories and file names

main_data_dir = '/lustre/storeB/project/fou/fd/project/acciberg/' # acciberg data directory

### External data 
# RTM and EnKF input
model_data_dir = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/' # TOPAZ data directory
sat_data_dir = f"{main_data_dir}atlems/topaz_l3/" # AMSR2 daily means
# RTM input
mask_file = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/2024/01/07/20240107_dm-metno-MODEL-topaz5-ARC-b20231229-fv02.0_mem005.nc' # File used for masking RTM output
tpd_data_dir = f"{main_data_dir}marina/tpd_files/" # TPD files directory
tpa_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpa/' # TPA files directory

### Produced data 
# RTM output
rtm_tbs_dir = f"{main_data_dir}marina/topaz_tbs/{date}/tbs_{fdays}fdays/" # RTM TOPAZ TBs directory
coeffs_filename = f"{main_data_dir}marina/topaz_tbs/coefficients_files/coefficients_{date}.csv" # file name that will contain 2D-plane coefficients
# EnKF output
my_exp = f"{assim}_{fdays}fdays" # my experiment subfolder
storage_dir = f"{main_data_dir}marina/enkf_exps/{my_exp}/{date}/" # storage directory (input and output data to/from EnKF)
enkf_run_dir = f"/lustre/storeB/users/marinadm/enkf_run/acciberg/{assim}/" # EnKF run directory
# Figures
figures_dir = f"{main_data_dir}marina/figures/"
mask_file_plots = f"{main_data_dir}marina/enkf_exps/conf/mask_topaz5.nc"
