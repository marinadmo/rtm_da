# First load modules!
# source ./modules.sh

run_all_steps = True
if run_all_steps: run_step1 = run_step2 = run_step3 = run_step4 = True
else :
    run_step1 = False
    run_step2 = False
    run_step3 = False
    run_step4 = True

from steps_da.main_imports import cmd  
import config

#################################################################################################################
if run_step1 :
    ### Produce RTM TBs
    if config.assim == 'exp_sic' :
        from rtm_dal.main_fcts_rtm_tbs import *

        ### Computation of DAL (Distance Along the Line) from TPD and TPA files
        dal_norm, _, list_tpd_files = compute_dal() 

        ### Computation of 2D-plane coefficients. This 2D-plane is defined by the relation between Emissivity, DAL and T2M
        compute_coeffs(dal_norm, list_tpd_files)

        ### Computation of RTM TBs
        res = run_rtm(version = 1) 

        ### Create directory where RTM TBs will be saved
        cmd('mkdir -p ' + f"{config.rtm_tbs_dir}")

        ### Create netCDF files containing RTM TBs and saved them in the previously defined directory
        save_rtm_tbs(res[0], f"{config.rtm_tbs_dir}/") 

#################################################################################################################
if run_step2 :
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
if run_step3 :
    ### Run EnKF
    from steps_da.run_da import run_enkf
    run_enkf()

#################################################################################################################
if run_step4 :
    ### Figures and diagnostics
    from steps_da.plotting import *
    if config.assim == 'exp_sic' :
        background_maps()
        analysis_maps()

    else :
        analysis_maps()
