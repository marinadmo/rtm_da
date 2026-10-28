# Code to build ensemble files from model output to feed enkf-c

import numpy as np
import netCDF4
import xarray as xr
import datetime
import glob, os

import config
print(f"Config module loaded from: {config.__file__}")

def read_mask() :
    
    model_file = config.mask_file
    model_var = 'model_mask'
    model_data = xr.open_dataset(model_file)
    mask = np.logical_not(model_data[model_var])
    
    return mask

def prep_ensemble() : 
    
    tb_vars = ('tb19v', 'tb19h', 'tb37v', 'tb37h') # Tb data is saved in different files
    ice_vars = config.varnames + config.varnames_ncat + tb_vars

    ens_date = datetime.datetime(int(config.date[0:4]), int(config.date[4:6]), int(config.date[6:8])) # Day of DA analysis
    
    if '2024' in config.date :
        if config.fdays == 1 : data_dir3 = config.date;
        else : 
            data_dir2 = ens_date - datetime.timedelta(days = config.fdays - 1)
            data_dir3 = f'{data_dir2.year:04d}{data_dir2.month:02d}{data_dir2.day:02d}'   
    

    for ens in range(0, config.Nens) :
        
        if '2024' in config.date :
            files_model_path = f'{config.model_data_dir}{config.date[0:4]}/{config.date[4:6]}/{config.date[6:8]}/{config.date}*{data_dir3}*mem{ens + 1:03d}*nc'
            #dimx, dimy = 'x', 'y'
        elif '2021' in config.date :
            files_model_path = f'{config.model_data_dir}/mem{ens + 1:03d}/daily/iceh.{config.date[0:4]}-{config.date[4:6]}-{config.date[6:8]}.nc'
            #dimx, dimy = 'ni', 'nj'
                
        for var in ice_vars:
            if var in tb_vars :
                ice_rst_file = f'{config.rtm_tbs_dir}{config.date}/means/topaz_tb_{config.date}_mem{ens + 1:03d}.nc'
                #dimx, dimy = 'x', 'y'
            else :
                files_model = glob.glob(files_model_path)
                #print(files_model)
                ice_rst_file = files_model[0]                
            
            print('Background TOPAZ file: ', ice_rst_file)

            ice_enkf_file = os.path.join(config.storage_dir, 'ensb', 'mem{:03d}_{}.nc'.format(ens + 1, var))

            with netCDF4.Dataset(ice_rst_file, mode = "r") as ice_rst, netCDF4.Dataset(ice_enkf_file, mode = "w") as ice_enkf:
                create_netcdf(ice_rst, ice_enkf, ens_date, var)
                

def create_netcdf(rst_ds, enkf_mem_ds, ens_date, var_name) :
    
    var_fill = rst_ds.variables[var_name] #[:]
    dimension_list = list(var_fill.dimensions)  
    print('check: ', np.shape(var_fill), len(np.shape(var_fill)))
    if len(np.shape(var_fill)) == 4 :        
        ncat = dimension_list[1]; ny, nx = 2, 3
    elif len(np.shape(var_fill)) == 3 :
        ny, nx = 1, 2
    elif len(np.shape(var_fill)) == 2 :
        ny, nx = 0, 1
    
    ### Create dimensions
    enkf_mem_ds.createDimension("time", size = 1)
    enkf_mem_ds.createDimension("dx", var_fill.shape[nx]) #size = rst_ds.dimensions[dimx].size)
    enkf_mem_ds.createDimension("dy", var_fill.shape[ny]) #size = rst_ds.dimensions[dimy].size)
    
    ### Create variables and fill in variables
    time = enkf_mem_ds.createVariable("time", float, dimensions = ("time"))
    dy = enkf_mem_ds.createVariable("dy", float, dimensions = ("dy"))
    dx = enkf_mem_ds.createVariable("dx", float, dimensions = ("dx"))
    
    ### Fill in variables
    time.units = "days since 1990-01-01"; time.calendar = "gregorian"
    time[0] = netCDF4.date2num(ens_date, time.units, time.calendar)    
    dy[:] = np.arange(0, dy.size)    
    dx[:] = np.arange(0, dx.size)
         
    dimension_list_in = ("time", "dy", "dx")
    if len(dimension_list) == 4 :
        enkf_mem_ds.createDimension("dz", size = rst_ds.dimensions[ncat].size)
        dz = enkf_mem_ds.createVariable("dz", float, dimensions = ("dz"))
        dz[:] = np.arange(0, dz.size)
        dimension_list_in = ("time", "dz", "dy", "dx")
        
    v = enkf_mem_ds.createVariable(var_name, float, dimensions = dimension_list_in)
    print('write ', var_name)
    v[:] = var_fill[:] #rst_ds.variables[var_name][:]        
                
### ASYNCHRONOUS assimilation

def create_netcdf_asyn(var_fill, enkf_mem_ds, ens_date, var_name) :

    # Masking model data
    #var_fill = np.ma.masked_array(var_fill.data, mask = read_mask())
    
    ny, nx = 0, 1
    
    ### Create dimensions
    enkf_mem_ds.createDimension('time', size = None)
    enkf_mem_ds.createDimension('dy', var_fill.shape[ny])
    enkf_mem_ds.createDimension('dx', var_fill.shape[nx])    

    ### Create variables
    time = enkf_mem_ds.createVariable("time", float, dimensions = ("time")) #('time', 'f4', ('time',))
    dy = enkf_mem_ds.createVariable("dy", float, dimensions = ("dy"))
    dx = enkf_mem_ds.createVariable("dx", float, dimensions = ("dx"))
    #dy = enkf_mem_ds.createVariable('dy', 'f4', ('dy',))
    #dx = enkf_mem_ds.createVariable('dx', 'f4', ('dx',))
    v = enkf_mem_ds.createVariable(var_name, float, ('time', 'dy', 'dx',))

    ### Fill in variables
    time.units = 'days since 1990-01-01'; time.calendar = 'gregorian'
    time[0] = netCDF4.date2num(ens_date, time.units, time.calendar)
    dy[:] = np.arange(0, dy.size) # var_fill.shape[1], 1.0)
    dx[:] = np.arange(0, dx.size) #var_fill.shape[0], 1.0)    
    v[0, :, :] = var_fill[:, :]    

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
                tb_history_file_model = f"{config.rtm_tbs_dir}{config.date}/passes/topaz_tb_{config.date}_mem{ens + 1:03d}.nc"
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
                filename = f"{config.storage_dir}/ensb/mem{ens + 1:03d}_{var}_{list_m[i-1]}.nc"
                print(filename)
                if os.path.exists(filename): os.remove(filename)
                data_write = netCDF4.Dataset(filename, 'w', format = 'NETCDF4')
                create_netcdf_asyn(var_inn, data_write, ens_date, var)
                data_write.close()

            elif 'tb' in config.assim :
                for ich in range(0, 4) :
                    var_inn = np.nanmean(m_tbs[ich, (i - 1)*6:(i - 1)*6 + 6, :, :], axis = 0)
                    var = config.channels[ich]
                    filename = f"{config.storage_dir}/ensb/mem{ens + 1:03d}_{var}_{list_m[i-1]}.nc"
                    print(filename)
                    if os.path.exists(filename): os.remove(filename)
                    data_write = netCDF4.Dataset(filename, 'w', format = 'NETCDF4')
                    create_netcdf_asyn(var_inn, data_write, ens_date, var)
                    data_write.close()

