# Code to build ensemble files from model output to feed enkf-c
from .main_imports import *
from .checks_enkf import check_dfs_srf as chk

def run_enkf() :
    cwd = os.getcwd()

    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Now you can import the config module
    import config

    # Run enkf
    cmd('rm ' + config.enkf_run_dir + 'ensemble_100/*')
    cmd('rm ' + config.enkf_run_dir + 'obs/*')
    cmd('rm ' + config.enkf_run_dir + 'conf/mask_topaz_tbs.nc')
    # Go to enkf_dir
    os.chdir(config.enkf_run_dir); print(os.getcwd())
    #cmd('./modules_da.sh')
    #cmd('module use /modules/MET/rhel8/user-modules/')
    #cmd('module load enkfc/2.9.9') # enkf module
    cmd('make clean')

    # Observation
    cmd('ln -s ' + config.main_data_dir + 'marina/enkf_exps/observations/amsr2_topaz_obs_' + config.date + '.nc ' + config.enkf_run_dir + 'obs/')
    # Background ensemble
    cmd('ln -s ' + config.storage_dir + 'ensb/* ' + config.enkf_run_dir + 'ensemble_100/')
    # Mask
    cmd('ln -s ' + config.main_data_dir + 'marina/enkf_exps/conf/mask_topaz_tbs.nc ' + config.enkf_run_dir + 'conf/')
    cmd('make')
    # Move back and update enkf ensemble
    os.chdir(cwd); print(os.getcwd())

    cmd('mkdir -p ' + config.storage_dir + 'ensa/'); cmd('mkdir -p ' + config.storage_dir + 'ensa/update2run/')
    cmd('cp ' + config.enkf_run_dir + 'ensemble_100/*analysis ' + config.storage_dir + 'ensa/')

    cmd('mkdir ' + config.storage_dir +  'enkf_files')
    # Check EnKF parameters
    for infile in glob.glob(config.enkf_run_dir +  'enkf_diag*nc') :
           chk(config.Nens, infile)

    cmd('cp ' + config.enkf_run_dir + 'enkf_diag*nc ' + config.storage_dir +  'enkf_files')
    cmd('cp ' + config.enkf_run_dir + 'observations.nc ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + 'spread.nc ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + '*prm ' + config.storage_dir + 'enkf_files')
    cmd('cp ' + config.enkf_run_dir + '*out ' + config.storage_dir + 'enkf_files')

