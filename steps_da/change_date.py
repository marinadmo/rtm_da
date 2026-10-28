from datetime import datetime, timedelta, date
import sys, os

import config
print(f"Config module loaded from: {config.__file__}")

def update_enkf_prm() :
    d0 = date(1990, 1, 1)
    previous_day = (datetime.strptime(config.date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
    d1 = date(int(previous_day[0:4]), int(previous_day[4:6]), int(previous_day[6:8]))
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
