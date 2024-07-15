# Code to build ensemble files from model output to feed enkf-c
from .main_imports import *

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def prep_ensemble() : 
    ice_vars = ('siconc', 'sithick', 'tb19v', 'tb19h', 'tb37v', 'tb37h')
    tb_vars = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # TB data is saved in different files

    ens_date = datetime.datetime(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8])) # Day of DA analysis

    print(config.date)
    if config.fdays == 1 : data_dir3 = config.date;
    else : 
        data_dir2 = ens_date - datetime.timedelta(days = config.fdays - 1)
        data_dir3 = f"{data_dir2.year:04d}{data_dir2.month:02d}{data_dir2.day:02d}"
    print(config.fdays, data_dir3)

    for ens in range(0, config.Nens) :
        for var in ice_vars:
            if var in tb_vars :
                ice_rst_file = f"{config.rtm_tbs_dir}/topaz_tb_{config.date}_mem{ens + 1:03d}.nc"
            else :
                files_model = glob.glob(f"{config.model_data_dir}{config.date[0:4]}/{config.date[4:6]}/{config.date[6:8]}/{config.date}*{data_dir3}*mem{ens + 1:03d}*nc")
                print(files_model)
                ice_rst_file = files_model[0]
                print("Background TOPAZ file: ", ice_rst_file)

            ice_enkf_file = os.path.join(config.storage_dir, 'ensb', 'mem{:03d}_{}.nc'.format(ens + 1, var))

            with netCDF4.Dataset(ice_rst_file, mode = "r") as ice_rst, netCDF4.Dataset(ice_enkf_file, mode = "w") as ice_enkf:
                ice_enkf.createDimension("time", size = 1)
                ice_enkf.createDimension("dx", size = ice_rst.dimensions["x"].size)
                ice_enkf.createDimension("dy", size = ice_rst.dimensions["y"].size)

                time = ice_enkf.createVariable("time", float, dimensions = ("time"))
                time.units = "days since 1990-01-01"
                time.calendar = "gregorian"
                time[0] = netCDF4.date2num(ens_date, time.units)


                dy = ice_enkf.createVariable("dy", float, dimensions = ("dy"))
                dy[:] = np.arange(0, dy.size)

                dx = ice_enkf.createVariable("dx", float, dimensions = ("dx"))
                dx[:] = np.arange(0, dx.size)

                print('write ', var)
                v = ice_enkf.createVariable(var, float, dimensions = ("time", "dy", "dx"))
                v[:] = ice_rst.variables[var][:]
