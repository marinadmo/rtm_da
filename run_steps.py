#

# First load modules!
# source ./modules.sh

from steps_da.main_imports import cmd  
import config


#################################################################################################################
### BACKGROUND ENSEMBLE, PREPARE OBSERVATIONS AND MASK FOR EnKF
cmd('mkdir -p ' + config.storage_dir)
cmd('mkdir -p ' + config.storage_dir + 'ensb');
cmd('rm ' + config.storage_dir + 'ensb/*')

if config.assim == 'exp_sic' :
    from steps_da.prepare_ens import prep_ensemble
    from steps_da.prepare_obs import prep_topaz
    from steps_da.model_mask import generate_mask
    
    # Prepare background ensemble for EnKF
    # ENSB preparation: prepare before TB from model SIC data!!
    prep_ensemble() 

    # Prepare observations
    prep_topaz()

    # Generate topaz tbs mask
    generate_mask() 
    #config.date)

else :
    storage_dir2 = f"{config.main_data_dir}marina/enkf_exps/exp_sic_{config.fdays}fdays/{config.date}/"
    cmd('ln -s ' + storage_dir2 + 'ensb/* ' + config.storage_dir + 'ensb/')

#################################################################################################################
### Run EnKF
from steps_da.run_da import run_enkf
run_enkf()

# Plot diagnostics

# Update ensemble?
