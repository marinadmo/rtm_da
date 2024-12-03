# config.py

###################################################################################################################################
##### 1-Define configuration variables

# RTM and EnKF variables
Nens = 10 # Model ensemble size
date = '20211201'

# RTM variables
channels = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # AMSR2 channels used in the RTM simulation
model = 'topaz' # Model used for the RTM simulation
size_tpd_data = 5000 # TPD files contain arrays with dimension equal to 5000
# Daily variables
varnames = ('aice_d', 'hi_d', 'hs_d', 'Tsfc_d', 'sst_d', 'sss_d', 'alvl_d', 'vlvl_d', 'iage', 'FYarea') #, 'vicen_d') #, 'vsnon_d') 

# EnKF variables
assim = 'exp_tb_asyn' # Assimilation of SIC ('exp_sic') or Tbs ('exp_tb'), if asyn then 'exp_sic_asyn' or 'exp_tb_asyn'
#opt = '' #_test

###################################################################################################################################
##### 2-Define data directories and file names

main_data_dir = '/lustre/storeB/project/fou/fd/project/acciberg/' # acciberg data directory

### INPUT DATA
# RTM and EnKF input
model_data_dir = f'/lustre/storeB/project/fou/fd/project/acciberg/marina/topaz5_nersc/{date[0:4]}/' # TOPAZ data directory
if 'asyn' in assim :
    sat_data_dir = f'{main_data_dir}atlems/tp5/l3u_{date[0:4]}/' # PASSES
    sat_data_dir2 = f"{main_data_dir}atlems/topaz_l3u/{date[0:4]}/"
else : 
    sat_data_dir = f'{main_data_dir}atlems/tp5/l3_{date[0:4]}/' # DAILY MEANS
    sat_data_dir2 = f"{main_data_dir}atlems/topaz_l3/{date[0:4]}/"
# RTM input
mask_file = f'/lustre/storeB/project/fou/fd/project/acciberg/marina/topaz5_nersc/{date[0:4]}/mem001/daily/iceh.2021-12-01.nc'
#tpd_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpd/' # TPD files directory
#tpa_data_dir = '/lustre/storeB/project/fou/fd/project/osisaf/osisaf-archive-data-oper/ice/conc/dm1-v3p0/tpa/' # TPA files directory
tpd_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpd/amsr_gw1/'
tpa_data_dir = '/lustre/storeB/project/fou/fd/project/sicci/cci3y3/amsr_2021-22/tpa/amsr_gw1/'

### OUTPUT DATA
# RTM output
rtm_tbs_dir = f"{main_data_dir}marina/topaz_tbs/{date}/" # RTM TOPAZ Tbs directory
coeffs_filename = f"{main_data_dir}marina/coefficients/{date[0:4]}/coefficients_{date}.csv" # file name that will contain 2D-plane coefficients

# EnKF output
exps_dir = f"{main_data_dir}marina/enkf_exps/"
#my_exp = f"{assim}" # experiment subfolder
storage_dir = f"{exps_dir}/exps_{date[0:4]}/{assim}/{date}/" # storage directory (input and output data to/from EnKF)
enkf_run_dir = f"/lustre/storeB/users/marinadm/enkf_run/acciberg/{assim}/" # EnKF run directory
# Figures
figures_dir = f"{main_data_dir}marina/figures/rtm_git_figures/" # Figure folder
mask_file_plots = f"{main_data_dir}marina/enkf_exps/conf/mask_topaz5.nc" # Mask used for plots