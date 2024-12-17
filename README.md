#1 Fill in the configuration file
Define your configuration file (config.py), such as data paths, experiment name, etc

#2 Load modules
Load the necessary modules (Python or conda, EnKF-C module)
conda activate production-10-2022
module load enkfc/2.9.9 (https://github.com/sakov/enkf-c)

#3 Run the jupyter notebook to run the standalone analysis
Run the jupyter notebook (rtm_enkf_notebook.ipynb) to:
1. Produce the RTM Tbs from model data
2. Prepare data and perform assimilation wiht EnKF
3. Generate figures
