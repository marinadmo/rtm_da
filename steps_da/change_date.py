import datetime
import sys, os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def update_enkf_prm() :
    d0 = datetime.date(1990, 1, 1)
    d1 = datetime.date(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8]))
    delta = d1 - d0
    #print(delta.days)
    
    enkf_path = f"{config.enkf_run_dir}"
    with open(f"{config.enkf_run_dir}/enkf.prm", 'r') as txt:
        text = txt.readlines()
        #print(text[8])
        text[8] = 'DATE = ' + str(delta.days) + ' days since 1990-01-01\n'
        with open(f"{config.enkf_run_dir}/enkf.prm", 'w') as txt:
            txt.writelines(text)
        #print(text[8])
