# First load modules!
# source ./modules.sh

'''
Assimilation of TBs to correct TOPAZ5 forecasts

    Standalone analysis using the EnKF-C software
    Model data: TOPAZ5 outputs
    Observations: sea-ice concentration (SIT) and Tbs (19v, 19h, 37v, 37h) from AMSR2

Four data assimilation experiments are defined. Experiments are to be performed in this order:

    exp_sic: synchronous assimilation of SIC
    exp_tb: synchronous assimilation of Tbs
    exp_sic_asyn: asynchronous assimilation of SIC
    exp_tb_asyn: asynchronous assimilation of Tbs

This notebook is composed of five steps:

    Computation of DAL and 2D-plane coefficients
    RTM Tbs simulation
    Plotting of RTM Tbs
    Preparation of background ensemble, observations and mask for EnKF
    Run EnKF
    Plotting
'''

import importlib
from main_imports import cmd

config_filename = f'config_{config.date[0:4]}.py' # Name of you configuration file
cmd(f'cp {config_filename} config.py')
import config; importlib.reload(config)

# Model mask file to be used in the RTM simulations and DA analysis
cmd(f'rm {config.exps_dir}/conf/mask_topaz5.nc')
cmd(f'ln -s {config.exps_dir}/conf/{config.date[0:4]}/topaz5_grid_{config.date[0:4]}.nc  {config.exps_dir}/conf/mask_topaz5.nc')

print("DA experiment: ", config.assim)

run_all_steps = False 
if run_all_steps: compute_coeffs = run_rtm = make_plots_rtm = prepare_enkf = run_enkf = make_plots_da = True
else :
    compute_coeffs = False 
    run_rtm = False
    make_plots_rtm = False
    prepare_enkf = False
    run_enkf = False
    make_plots_da = True     
    
'''
    1. Computation of DAL and 2D-plane coefficients
'''
if compute_coeffs :
    
    from rtm_dal.main_fcts_rtm_tbs import *

    print('Computation of DAL (Distance Along the Line) from TPD and TPA files...')
    dal_norm, _, list_tpd_files = compute_dal() 

    print('\nComputation of 2D-plane coefficients. This 2D-plane is defined by the relation between Emissivity, DAL and T2m...')
    compute_coeffs(dal_norm, list_tpd_files)  
    
    print('...computation finished.')

'''
    2. RTM Tb simulation
'''
if run_rtm :
    
    from rtm_dal.main_fcts_rtm_tbs import *
    
    if 'exp_tb_asyn' in config.assim :
        
        print('\nComputation of RTM Tbs (swaths)...')
        import warnings
        warnings.filterwarnings("ignore") # Ignore warning related to division in eq35
        res = run_rtm_swaths(version = 1)
        warnings.resetwarnings()        

        print('\nCreate directory where RTM Tbs will be saved')
        cmd('mkdir -p ' + f"{config.rtm_tbs_dir}"); cmd('mkdir -p ' + f"{config.rtm_tbs_dir}/passes/")

        print('\nCreate netCDF files containing RTM Tbs and saved them in the previously defined directory')
        save_rtm_tbs(res[0], f"{config.rtm_tbs_dir}/passes/", swaths = True)
        
    elif 'exp_sic' in config.assim :
        
        print('\nComputation of RTM Tbs (daily means)')
        import warnings
        warnings.filterwarnings("ignore") # Ignore warning related to division in eq35
        res = run_rtm(version = 1) 
        warnings.resetwarnings()        

        print('\nCreate directory where RTM Tbs will be saved')
        cmd('mkdir -p ' + f"{config.rtm_tbs_dir}"); cmd('mkdir -p ' + f"{config.rtm_tbs_dir}/means/")

        print('\nCreate netCDF files containing RTM Tbs and saved them in the previously defined directory')
        save_rtm_tbs(res[0], f"{config.rtm_tbs_dir}/means/")
    
    print('...RTM simulation finished.')
    

'''
    3. Plotting from RTM Tb simulation (comparison to AMSR2 Tbs). Plot scripts might need to be adapted depending on your data format and dimensions
'''
if make_plots_rtm :
    
    # Create a RTM Tbs map and difference to AMSR2 (two rows)
    from plotting_codes import plotting_rtm
    #from plotting_codes.plotting_rtm import *
    importlib.reload(plotting_rtm)
    
    plot_rtm = plotting_rtm.subplots_rtm_tbs #metrics
    plot_rtm() 
    
    plot_hist = plotting_rtm.plot_histograms
    plot_hist()
    
    plot_diags = plotting_rtm.plot_diagrams
    plot_diags()    
        
    print('...plotting finished.')

'''
    4. Preparation of background ensemble, observations and mask for EnKF  
'''
if prepare_enkf :
    
    print('Preparation of background ensemble, observations and mask for EnKF-C')
    cmd('mkdir -p ' + config.storage_dir)
    cmd('mkdir -p ' + config.storage_dir + 'ensb');
    cmd('rm ' + config.storage_dir + 'ensb/*nc')

    if config.assim == 'exp_sic' :
        
        from steps_da import prepare_ens; importlib.reload(prepare_ens); prep_ens = prepare_ens.prep_ensemble        
        from steps_da import prepare_obs; importlib.reload(prepare_obs); prep_topaz = prepare_obs.prep_topaz        
        from steps_da import model_mask; importlib.reload(model_mask); prep_mask = model_mask.generate_mask
        
        print('Ensemble preparation...'); prep_ens()        
        print('Prepare observations (means)...'); prep_topaz()        
        print('Generation of TOPAZ mask...'); prep_mask() 

    elif config.assim == 'exp_tb' :      
        
        storage_dir_tbs = f"{config.exps_dir}/exps_2021/exp_sic/{config.date}/"
        cmd('ln -s ' + storage_dir_tbs + 'ensb/* ' + config.storage_dir + 'ensb/') # Link to background ensemble
        
    elif config.assim == 'exp_sic_asyn' :
        
        storage_dir_asyn = f"{config.exps_dir}/exps_2021/exp_sic/{config.date}/"
        cmd('ln -s ' + storage_dir_asyn + 'ensb/* ' + config.storage_dir + 'ensb/') # Link to background ensemble
        
        from steps_da import prepare_ens; importlib.reload(prepare_ens); prep_ens = prepare_ens.prep_ensemble_asyn
        from steps_da import prepare_obs; importlib.reload(prepare_obs); prep_topaz = prepare_obs.prep_topaz_passes
        from steps_da import change_date; importlib.reload(change_date); update = change_date.update_enkf_prm
        
        print('Ensemble preparation for ASYN experiment...'); prep_ens()        
        print('Prepare observations (passes)...'); prep_topaz()
        print('Update the EnKF date...'); update()
        
    elif config.assim == 'exp_tb_asyn' : 
        
        storage_dir_asyn = f"{config.exps_dir}/exps_2021/exp_sic/{config.date}/"
        cmd('ln -s ' + storage_dir_asyn + 'ensb/* ' + config.storage_dir + 'ensb/') # Link to background ensemble
        
        from steps_da import prepare_ens; importlib.reload(prepare_ens); prep_ens = prepare_ens.prep_ensemble_asyn
        
        print('Ensemble preparation for ASYN experiment...'); prep_ens()   

    print('...preparation of model and observation data finished.')
    
'''
    5. Run EnKF  
'''
if run_enkf :
    
    cmd('module use /modules/MET/rhel8/user-modules/')
    cmd('module load enkfc/2.9.9') 

    print('Running EnKF-C...')
    from steps_da.run_da import run_enkf
    run_enkf()
    
    print('...run finished.')

'''
    6. Plotting from DA results
         vari = 0 is aice
         vari = 1 is hi
         vari = 8 is iage
'''
vari = 0
metrics = True
if make_plots_da :
    
    print(f'Making figures for variable {config.varnames[vari]}...')
    
    from plotting_codes import plotting_da
    importlib.reload(plotting_da)
    from plotting_codes.plotting_da import *
    
    plot_metrics = plotting_da.plot_metrics
    
    if config.assim == 'exp_sic' :
        if vari == 0 :
            background_maps()
            background_maps(var = 'tb', var_tb = 0)
            analysis_maps(vari)
        elif vari > 0 : no_obs_maps(vari)

    else :
        if vari == 0 :
            analysis_maps(vari)
        elif vari > 0 : no_obs_maps(vari)
        
    if metrics : 
        
        print('\nPlotting EnKF metrics (DFS and SRF)...')
        plot_metrics()
        
    print('...plotting finished.')