import sys, os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Code to build ensemble files from model output to feed enkf-c
from main_imports import *
# Now you can import the config module
import config

def prep_ensemble() : 
    
    tb_vars = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # Tb data is saved in different files
    ice_vars = config.varnames + tb_vars

    ens_date = datetime.datetime(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8])) # Day of DA analysis
    
    if '2024' in config.date :
        if config.fdays == 1 : data_dir3 = config.date;
        else : 
            data_dir2 = ens_date - datetime.timedelta(days = config.fdays - 1)
            data_dir3 = f'{data_dir2.year:04d}{data_dir2.month:02d}{data_dir2.day:02d}'   
    

    for ens in range(0, config.Nens) :
        
        if '2024' in config.date :
            files_model_path = f'{config.model_data_dir}{config.date[0:4]}/{config.date[4:6]}/{config.date[6:8]}/{config.date}*{data_dir3}*mem{ens + 1:03d}*nc'
            dimx, dimy = 'x', 'y'
        elif '2021' in config.date :
            files_model_path = f'{config.model_data_dir}/mem{ens + 1:03d}/daily/iceh.{config.date[0:4]}-{config.date[4:6]}-{config.date[6:8]}.nc'
            dimx, dimy = 'ni', 'nj'
                
        for var in ice_vars:
            if var in tb_vars :
                ice_rst_file = f'{config.rtm_tbs_dir}/means/topaz_tb_{config.date}_mem{ens + 1:03d}.nc'
                dimx, dimy = 'x', 'y'
            else :
                files_model = glob.glob(files_model_path)
                #print(files_model)
                ice_rst_file = files_model[0]                
            
            print('Background TOPAZ file: ', ice_rst_file)

            ice_enkf_file = os.path.join(config.storage_dir, 'ensb', 'mem{:03d}_{}.nc'.format(ens + 1, var))

            with netCDF4.Dataset(ice_rst_file, mode = "r") as ice_rst, netCDF4.Dataset(ice_enkf_file, mode = "w") as ice_enkf:
                ice_enkf.createDimension("time", size = 1)
                ice_enkf.createDimension("dx", size = ice_rst.dimensions[dimx].size)
                ice_enkf.createDimension("dy", size = ice_rst.dimensions[dimy].size)

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


### ASYNCHRONOUS assimilation

def create_netcdf(filename, ens_date, var_name, var_fill) :

    if os.path.exists(filename): os.remove(filename)
    ds = netCDF4.Dataset(filename, 'w', format = 'NETCDF4')

    ### Create dimensions
    time = ds.createDimension('time', None)
    dx = ds.createDimension('dx', var_fill.shape[0])
    dy = ds.createDimension('dy', var_fill.shape[1])

    ### Create variables
    times = ds.createVariable('time', 'f4', ('time',))
    dxs = ds.createVariable('dx', 'f4', ('dx',))
    dys = ds.createVariable('dy', 'f4', ('dy',))
    temps = ds.createVariable(var_name, 'f4', ('time', 'dx', 'dy',))

    ### Fill in variables
    dxs[:] = np.arange(0, var_fill.shape[0], 1.0)
    dys[:] = np.arange(0, var_fill.shape[1], 1.0)
    temps[0, :, :] = var_fill[:, :]
    times[:] = netCDF4.date2num(ens_date, units = 'days since 1990-01-01', calendar = 'gregorian')
    times.units = 'days since 1990-01-01'
    times.calendar = 'gregorian'
    ds.close()

def prep_ensemble_asyn() :
    
    ens_date = datetime.datetime(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8])) # Day of DA analysis
    ens_date_previous = datetime.date.fromordinal(datetime.date.toordinal(datetime.datetime(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8]))) - 1);

    for ens in range(0, config.Nens) :
        
        if 'sic' in config.assim :
            var_asyn = 'aice' 
            ice_history_file_model = f'{config.model_data_dir}/mem{ens + 1:03d}/hourly/iceh_concat.{config.date[0:4]}-{config.date[4:6]}-{config.date[6:8]}.nc'
            datai = netCDF4.Dataset(ice_history_file_model, mode = 'r')
            aicen_h = datai[var_asyn][:]
            datai.close()

        elif 'tb' in config.assim : 
            ich = 0
            for ich in range(0, len(config.channels)) :
                # RTM Tbs files
                tb_history_file_model = f"{config.rtm_tbs_dir}/passes/topaz_tb_{config.date}_mem{ens + 1:03d}.nc"
                datao = xr.open_dataset(tb_history_file_model)
                if ich == 0 : m_tbs = np.zeros((len(config.channels), np.shape(datao[config.channels[ich]][:])[0], np.shape(datao[config.channels[ich]][:])[1], np.shape(datao[config.channels[ich]][:])[2]))
                m_tbs[ich, :] = datao[config.channels[ich]][:]
                datao.close()
       

        list_t = [3, 9, 15, 21]
        #list_t = [0, 6, 12, 18]
        list_m = [-4, -3, -2, -1]
        for i in range(1, 5) :
            ens_date = datetime.datetime(ens_date_previous.year, ens_date_previous.month, ens_date_previous.day, list_t[i - 1]) #int((i-1)*6+6))
            if 'sic' in config.assim :
                var_inn = np.nanmean(aicen_h[(i - 1)*6:(i - 1)*6 + 6, :, :], axis = 0)
                var = config.varnames[0] # aice_d
                fn = f"{config.storage_dir}/ensb/mem{ens + 1:03d}_{var}_{list_m[i-1]}.nc"
                print(fn)
                create_netcdf(fn, ens_date, var, var_inn)

            elif 'tb' in config.assim :
                for ich in range(0, 4) :
                    var_inn = np.nanmean(m_tbs[ich, (i - 1)*6:(i - 1)*6 + 6, :, :], axis = 0)
                    var = config.channels[ich]
                    fn = f"{config.storage_dir}/ensb/mem{ens + 1:03d}_{var}_{list_m[i-1]}.nc"
                    print(fn)
                    create_netcdf(fn, ens_date, var, var_inn)

