# Figures and diagnostics before and after DA
import matplotlib.pyplot as plt
import cmocean
#
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .main_imports import *

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def read_obs() :

    sat_data = netCDF4.Dataset(f"{config.main_data_dir}/marina/enkf_exps/observations/amsr2_topaz_obs_{config.date}.nc")
    sic_sat = sat_data['ice_conc'][0, :]
    
    return sic_sat


def read_model_matrix(opt = 0) :

    sic_model_ens = dict()  
    for imem in range(1, config.Nens + 1) :
        mem = 'mem' + "{0:03}".format(imem)
        print(mem)
        #####
        # SIC
        if opt == 0 : filename = f"{config.storage_dir}/ensb/{mem}_siconc.nc"
        elif opt == 1 : 
            my_exp = f"exp_sic_{config.fdays}fdays"
            filename = f"{config.main_data_dir}marina/enkf_exps/{my_exp}/{config.date}/ensa/{mem}_siconc.nc.analysis" 
        else : 
            my_exp = f"exp_tb_{config.fdays}fdays"
            filename = f"{config.main_data_dir}marina/enkf_exps/{my_exp}/{config.date}/ensa/{mem}_siconc.nc.analysis"

        # Open topaz model files
        print(glob.glob(filename)[0])
        model_data = xr.open_dataset(glob.glob(filename)[0])
        if imem == 1 :
            new_mask = np.zeros((np.shape(model_data.siconc[0, :])[0], np.shape(model_data.siconc[0, :])[1]), dtype = bool)
            new_mask[model_data.siconc[0, :] > 100] = True
        ###
        sic_model_ens[mem] = model_data.siconc[0, :]
    
    return sic_model_ens, new_mask

def lon_lat() :
    dataset = xr.open_dataset(config.mask_file_plots)
    lon, lat = dataset.longitude.data, dataset.latitude.data
    return lon, lat

def background_maps() :
    lon, lat = lon_lat()

    sic_sat = read_obs()
    sic_model_ens, new_mask = read_model_matrix()

    data1 = sic_sat
    data2 = np.ma.masked_array(np.nanmean(list(sic_model_ens.values()), axis = 0), mask = new_mask)  
    data3 = np.nanmean(list(sic_model_ens.values()), axis = 0) - sic_sat
    data4 = np.ma.masked_array(np.nanstd(list(sic_model_ens.values()), axis = 0), mask = new_mask)

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(1, 4, figsize = (20, 8), subplot_kw = {'projection': ccrs.NorthPolarStereo()})

    # List of data arrays and axes for iteration
    data_list = [data1, data2, data3, data4]
    titles = ['AMSR2 SIC', r'TOPAZ SIC ($\overline{x_b}$)', 'SIC bias', r'TOPAZ SIC spread$_b$']

    # Dictionary of visualization parameters
    viz_params = {
        'data1': {'cmap': cmocean.cm.ice, 'vmin': 0, 'vmax': 1, 'text' : None},
        'data2': {'cmap': cmocean.cm.ice, 'vmin': 0, 'vmax': 1, 'text' : None},
        'data3': {'cmap': cmocean.cm.balance, 'vmin': -0.5, 'vmax': 0.5, 'text' : f'MAE={str(round(np.nanmean(abs(data3)), 4))}'},
        'data4': {'cmap': 'Greens', 'vmin': 0., 'vmax': 0.3, 'text' : r'$\overline{spread_b}$=' + f'{str(round(np.nanmean(data4), 4))}'},
    }

    # Iterate over axes, data, titles, and visualization parameters
    for ax, data, title, (key, params) in zip(axes.flatten(), data_list, titles, viz_params.items()):
        # Set the map extent
        ax.set_extent([-180, 180, 45, 90], crs = ccrs.PlateCarree())

        # Add land with light gray color
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor = 'face',
                                            facecolor = 'lightgray')
        ax.add_feature(land, zorder = 1)

        # Plot the data with the appropriate transform
        mesh = ax.pcolormesh(lon, lat, data, transform = ccrs.PlateCarree(), zorder = 2,
                             cmap = params['cmap'], vmin = params['vmin'], vmax = params['vmax'])

        # Add coastlines and gridlines
        ax.coastlines(resolution = '50m', color = 'gray', linewidth = 0.5, zorder = 3)
        ax.gridlines()

        # Add a title to each subplot
        ax.set_title(title, fontsize = 16)
        
        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)
        
        # Add annotation in the upper left corner:w
        ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

    # Savefig
    fig.savefig(config.figures_dir + 'subplots_background_' + config.date + '_' + str(config.fdays) + 'fdays.png', bbox_inches = 'tight', dpi = 300)

def analysis_maps() :
    lon, lat = lon_lat()
    sic_sat = read_obs()
    sic_model_ensb, new_mask = read_model_matrix(opt = 0)
    if 'tb' in config.my_exp : opta = 2
    else : opta = 1
    sic_model_ensa, _ = read_model_matrix(opt = opta)
    #
    data2 = np.ma.masked_array((np.nanmean(list(sic_model_ensa.values()), axis = 0) - np.nanmean(list(sic_model_ensb.values()), axis = 0)), mask = new_mask)
    data3 = np.ma.masked_array(np.nanstd(list(sic_model_ensa.values()), axis = 0), mask = new_mask)
    data4 = np.nanmean(list(sic_model_ensa.values()), axis = 0) - sic_sat
    mad_xb = np.nanmean(abs(np.nanmean(list(sic_model_ensb.values()), axis = 0) - sic_sat))
    mad_xa = np.nanmean(abs(data4))
    incr = np.nanmean(abs(data2))

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(1, 3, figsize = (15, 8), subplot_kw={'projection': ccrs.NorthPolarStereo()})

    # List of data arrays and axes for iteration
    data_list = [data2, data3, data4]
    titles = ['SIC increment', r'TOPAZ SIC spread$_a$', 'Updated SIC bias']

    # Dictionary of visualization parameters
    viz_params = {
        'data2': {'cmap': cmocean.cm.balance, 'vmin': -0.5, 'vmax': 0.5, 'text' : r'$\overline{incr}$=' + f'{str(round(np.nanmean(abs(data2)), 4))}'},
        'data3': {'cmap': 'Greens', 'vmin': 0, 'vmax': 0.3, 'text' : r'$\overline{spread_a}$=' + f'{str(round(np.nanmean(data3), 4))}'},
        'data4': {'cmap': cmocean.cm.balance, 'vmin': -0.5, 'vmax': 0.5, 'text' : f'MAE={str(round(np.nanmean(abs(data4)), 4))}'}
    }

    # Iterate over axes, data, titles, and visualization parameters
    for ax, data, title, (key, params) in zip(axes.flatten(), data_list, titles, viz_params.items()):
        # Set the map extent
        ax.set_extent([-180, 180, 45, 90], crs = ccrs.PlateCarree())

        # Add land with light gray color
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor = 'face',
                                            facecolor = 'lightgray')
        ax.add_feature(land, zorder = 1)

        # Plot the data with the appropriate transform
        mesh = ax.pcolormesh(lon, lat, data, transform = ccrs.PlateCarree(), zorder = 2,
                             cmap = params['cmap'], vmin = params['vmin'], vmax = params['vmax'])

        # Add coastlines and gridlines
        ax.coastlines(resolution = '50m', color = 'gray', linewidth = 0.5, zorder = 3)
        ax.gridlines()

        # Add a title to each subplot
        ax.set_title(title, fontsize = 16)

        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)

        # Add annotation in the upper left corner
        ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

    # Savefig
    fig.savefig(f"{config.figures_dir}subplots_{config.my_exp}_{config.date}_{config.fdays}fdays.png", bbox_inches = 'tight', dpi = 300)  
