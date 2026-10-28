# config.py

##### 1-Define configuration variables

Nens = 10 # Model ensemble size
date = '20211201'
channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
model = 'topaz' # Model used for the RTM simulation
size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000
varnames = ('aice_d', 'hi_d', 'hs_d', 'Tsfc_d', 'sst_d', 'sss_d', 'alvl_d', 'vlvl_d', 'iage', 'FYarea') # model variable names

''' Assimilation of SIC ('exp_sic') 
    or Tbs ('exp_tb'), 
    if asyn then 'exp_sic_asyn' 
    or 'exp_tb_asyn'
'''
assim = 'exp_tb_asyn' 

##### 2-Define data directories and file names

main_data_dir = '' # main data directory to store results

# Model data
model_data_dir = '' 

# Satellite (AMSR2) and atmospheric (ERA5) data
if 'asyn' in assim :
    '''Passes'''
    sat_data_dir = f'{main_data_dir}{}/' # PASSES
    #sat_data_dir2 = f"{main_data_dir}{}/"
else : 
    '''Daily means'''
    sat_data_dir = f'{main_data_dir}{}/' # DAILY MEANS
    #sat_data_dir2 = f'{main_data_dir}{}/'
    
# TPD and TPA data
tpd_data_dir = ''
tpa_data_dir = ''

# RTM output
rtm_tbs_dir = f"{main_data_dir}{}/" # RTM TOPAZ Tbs directory
coeffs_filename = f"{main_data_dir}{}/" # file name that will contain 2D-plane coefficients

# EnKF output
exps_dir = f"{main_data_dir}{}/" # general experiment directory
storage_dir = f"{exps_dir}{}/" # storage directory (input and output data to/from EnKF) for this specific experiment
enkf_run_dir = f"{main_data_dir}{}{assim}/" # EnKF run directory

# Figures
figures_dir = f"{main_data_dir}{}/" # Figure folder
mask_file = f"{main_data_dir}{}/mask_file.nc" # Mask file used for data reading and plots