# Code to build ensemble files from model output to feed enkf-c
from .main_imports import *
from .checks_enkf import check_dfs_srf as chk

def run_enkf() :
    cwd = os.getcwd()

    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Now you can import the config module
    import config

    # Remove previous files
    cmd('rm ' + config.enkf_run_dir + 'ensemble_100/*')
    cmd('rm ' + config.enkf_run_dir + 'obs/*')
    cmd('rm ' + config.enkf_run_dir + 'conf/mask_topaz_tbs.nc')
    
    # Go to enkf_dir and clean
    os.chdir(config.enkf_run_dir); #print(os.getcwd())
    cmd('make clean')
    
    # Copy prm files
    prm_files = f'/lustre/storeB/users/marinadm/enkf_run/acciberg/{config.date[0:4]}_prm_files/{config.assim}/' 
    cmd(f'cp  {prm_files}*prm {config.enkf_run_dir}')
    
    # Link observation
    if 'asyn' in config.assim : obs_dir = f'{config.main_data_dir}marina/enkf_exps/observations/passes/'
    else : obs_dir = f'{config.main_data_dir}marina/enkf_exps/observations/means/'
    cmd(f'ln -s {obs_dir}amsr2_topaz_obs_{config.date}*.nc {config.enkf_run_dir}obs/')
    
    # Link background ensemble
    cmd('ln -s ' + config.storage_dir + 'ensb/* ' + config.enkf_run_dir + 'ensemble_100/')
    
    # Link masks
    cmd('ln -s ' + config.main_data_dir + 'marina/enkf_exps/conf/mask_topaz5.nc ' + config.enkf_run_dir + 'conf/')
    cmd('ln -s ' + config.main_data_dir + 'marina/enkf_exps/conf/mask_topaz_tbs.nc ' + config.enkf_run_dir + 'conf/')
    
    # Run enkf
    cmd('make')
    
    # Move back and update enkf ensemble
    os.chdir(cwd); #print(os.getcwd())

    # Create ensa file and copy analysis data
    cmd('mkdir -p ' + config.storage_dir + 'ensa/'); cmd('mkdir -p ' + config.storage_dir + 'ensa/update2run/')
    cmd('cp ' + config.enkf_run_dir + 'ensemble_100/*analysis ' + config.storage_dir + 'ensa/')    
    
    # Check EnKF parameters
    for infile in glob.glob(config.enkf_run_dir +  'enkf_diag*nc') :
           chk(config.Nens, infile)
            
    # Create enkf_files folder and copy data
    cmd('mkdir ' + config.storage_dir +  'enkf_files')
    cmd('cp ' + config.enkf_run_dir + 'enkf_diag*nc ' + config.storage_dir +  'enkf_files')
    cmd('cp ' + config.enkf_run_dir + 'observations.nc ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + 'spread.nc ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + '*prm ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + '*out ' + config.storage_dir + 'enkf_files')