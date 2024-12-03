# config.py

###################################################################################################################################
##### Define configuration variables

# RTM and EnKF variables
Nens = 10 # Model ensemble size
date = '20211201'
#fdays = 100

# RTM variables
channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
model = 'topaz' # Model used for the RTM simulation
size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000
# Daily variables
varnames = ('aice_d', 'hi_d', 'hs_d', 'Tsfc_d', 'sst_d', 'sss_d', 'alvl_d', 'vlvl_d', 'iage', 'FYarea') #, 'vicen_d') #, 'vsnon_d') 
# Hourly variables

# EnKF variables
assim = 'exp_tb' # Assimilation of SIC (exp_sic) or TB (exp_tb)
opt = '' #_test

###################################################################################################################################
##### Define data directories and file names

main_data_dir = '/lustre/storeB/project/fou/fd/project/acciberg/' # acciberg data directory

### External data 
# RTM and EnKF input
#model_data_dir = '/lustre/storeB/project/copernicus/acciberg/metnotopaz5_ens/' # TOPAZ data directory
model_data_dir = f'/lustre/storeB/project/fou/fd/project/acciberg/marina/topaz5_nersc/{date[0:4]}/'#mem001/daily/'
#sat_data_dir = f"{main_data_dir}atlems/topaz_l3/{date[0:4]}" # AMSR2 daily means
#sat_data_dir = f"{main_data_dir}atlems/topaz_l3/new_20240919" # AMSR2 daily means
# DAILY MEANS
sat_data_dir = f'{main_data_dir}atlems/tp5/l3_{date[0:4]}/'
sat_data_dir2 = f"{main_data_dir}atlems/topaz_l3/{date[0:4]}/"
# SWATHS
#sat_data_dir = f'{main_data_dir}atlems/tp5/l3u_{date[0:4]}/'

# RTM input
mask_file = f'/lustre/storeB/project/fou/fd/project/acciberg/marina/topaz5_nersc/{date[0:4]}/mem001/daily/iceh.2021-12-01.nc'
#tpd_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpd/' # TPD files directory
#tpa_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpa/' # TPA files directory
tpd_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpd/amsr_gw1/'
tpa_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpa/amsr_gw1/'
### Produced data 
# RTM output
rtm_tbs_dir = f"{main_data_dir}marina/topaz_tbs/{date}/" # RTM TOPAZ TBs directory
coeffs_filename = f"{main_data_dir}marina/coefficients/{date[0:4]}/coefficients_{date}.csv" # file name that will contain 2D-plane coefficients
# Figures
#figures_dir = f"{main_data_dir}marina/figures/"
#mask_file_plots = f"{main_data_dir}marina/enkf_exps/conf/mask_topaz5.nc"

# EnKF output
my_exp = f"{assim}" #_{fdays}fdays{opt}" # my experiment subfolder
storage_dir = f"{main_data_dir}marina/enkf_exps/exps_{date[0:4]}/{my_exp}/{date}/" # storage directory (input and output data to/from EnKF)
enkf_run_dir = f"/lustre/storeB/users/marinadm/enkf_run/acciberg/{assim}/" # EnKF run directory
# Figures
figures_dir = f"{main_data_dir}marina/figures/rtm_git_figures/"
mask_file_plots = f"{main_data_dir}marina/enkf_exps/conf/mask_topaz5.nc"