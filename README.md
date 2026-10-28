#Folder information
config_files: config.py example files with paths and variables names
plotting_codes: scripts to make figures from results
rtm_dal_param: RTM code and DAL parameterization scripts
    rtm_atms_fcts.py: main RTM script (it can use different ice emissivity parameterizations)
    main_fct_rtm_tb.py: a series of function to run the RTM, read and save data
    dynamic_tiepoints and pmr_codes folders: they contain scripts needed for DAL parameterization (computation of coefficients)
steps_da: assimilation scripts

#1 Fill in the configuration file
Define your configuration file (config.py), such as data paths, experiment name, etc

#2 Load modules
Load the necessary modules (Python or conda, EnKF-C module)
conda activate production-10-2022
module load enkfc/2.9.9 (https://github.com/sakov/enkf-c)

#3 Run run_rtm_enkf_steps.py to perform standalone analysis
    1. Produce the RTM Tbs from model data
    2. Prepare data and perform assimilation wiht EnKF
    3. Generate figures
