# Figures and diagnostics before and after DA
import matplotlib.pyplot as plt
import cmocean
#
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .main_imports import *
from .checks_enkf import check_dfs_srf

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the config module
import config

def read_obs() :

    sat_data = netCDF4.Dataset(f"{config.main_data_dir}/marina/enkf_exps/observations/amsr2_topaz_obs_{config.date}.nc")
    sic_sat = sat_data['ice_conc'][0, :]
    
    return sic_sat

def read_obs_tb(var_tb = 0) :

    sat_data = netCDF4.Dataset(f"{config.main_data_dir}/marina/enkf_exps/observations/amsr2_topaz_obs_{config.date}.nc")
    tb_sat = sat_data[config.channels[var_tb]][0, :]
    
    return tb_sat

#data3 = np.ma.masked_array(np.nanstd(list(sic_model_ensa.values()), axis = 0), mask = new_mask)
def read_mask() :
    
    mask_data = xr.open_dataset(config.mask_file_plots)['model_mask'][:]
    # Step 1: Invert the values (0 becomes 1, 1 becomes 0)
    mask_data_inverted = 1 - mask_data

    # Step 2: Convert to boolean (1 becomes True, 0 becomes False)
    mask_data_bool = mask_data_inverted.astype(bool)
    
    return mask_data_bool
    
def read_model_matrix(opt = 0, vari = 0) :

    sic_model_ens = dict()  
    for imem in range(1, config.Nens + 1) :
        mem = 'mem' + "{0:03}".format(imem)
        print(mem)
        #####
        # SIC
        if opt == 0 : filename = f"{config.storage_dir}/ensb/{mem}_{config.varnames[vari]}.nc"
        else : filename = f"{config.storage_dir}/ensa/{mem}_{config.varnames[vari]}.nc.analysis" 
        print(filename)
        # Open topaz model files
        model_data = xr.open_dataset(glob.glob(filename)[0])
        sic_model_ens[mem] = model_data[config.varnames[vari]][0, :]
    
    return sic_model_ens #, new_mask

def read_model_matrix_tb(var_tb = 0) :
    
    tb_model_ens = dict()  
    for imem in range(1, config.Nens + 1) :
        mem = 'mem' + "{0:03}".format(imem)
        print(mem)
        
        # Background Tbs
        filename = f"{config.rtm_tbs_dir}/topaz_tb_{config.date}_{mem}.nc" 
        # Open topaz model files
        print(glob.glob(filename)[0])
        model_data = xr.open_dataset(glob.glob(filename)[0])
        tb_model_ens[mem] = model_data[config.channels[var_tb]][0, :]
    
    return tb_model_ens


def lon_lat() :
    dataset = xr.open_dataset(config.mask_file_plots)
    lon, lat = dataset.longitude.data, dataset.latitude.data
    return lon, lat

def background_maps(var = 'sic', var_tb = 0) :
    lon, lat = lon_lat()

    if var == 'sic' : 
        sic_model_ens = read_model_matrix()
        sic_sat = read_obs()
    elif var == 'tb' : 
        var_tb = 0
        sic_model_ens = read_model_matrix_tb(var_tb = 0)
        sic_sat = read_obs_tb(var_tb = 0)
        
    new_mask = read_mask()

    data1 = sic_sat
    data2 = np.ma.masked_array(np.nanmean(list(sic_model_ens.values()), axis = 0), mask = new_mask)  
    data3 = np.nanmean(list(sic_model_ens.values()), axis = 0) - sic_sat
    data4 = np.ma.masked_array(np.nanstd(list(sic_model_ens.values()), axis = 0), mask = new_mask)

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(1, 4, figsize = (20, 8), subplot_kw = {'projection': ccrs.NorthPolarStereo()})

    # List of data arrays and axes for iteration
    data_list = [data1, data2, data3, data4]

    if var == 'sic' :
        vmin1, vmax1 = 0, 1
        vmin2, vmax2 = 0, 1
        vmin3, vmax3 = -0.5, 0.5
        vmin4, vmax4 = 0, 0.3
        var_title = 'SIC'
        cmap_var = cmocean.cm.ice
        var_fig = var
    elif var == 'tb' :
        vmin1, vmax1 = np.nanmin(sic_sat), np.nanmax(sic_sat)
        vmin2, vmax2 = np.nanmin(sic_sat), np.nanmax(sic_sat)
        vmin3, vmax3 = -30, 30
        vmin4, vmax4 = 0, 20
        var_title = config.channels[var_tb][0].upper() + config.channels[var_tb][1:] + ' (K)'
        cmap_var = cmocean.cm.thermal
        var_fig = config.channels[var_tb]
    
    titles = [f'AMSR2 {var_title}', fr'TOPAZ {var_title} ($\overline{{x_b}}$)', f'{var_title} bias', fr'TOPAZ {var_title} spread$_b$']
    
    # Dictionary of visualization parameters
    viz_params = {
        'data1': {'cmap': cmap_var, 'vmin': vmin1, 'vmax': vmax1, 'text' : None},
        'data2': {'cmap': cmap_var, 'vmin': vmin2, 'vmax': vmax2, 'text' : None},
        'data3': {'cmap': cmocean.cm.balance, 'vmin': vmin3, 'vmax': vmax3, 'text' : f'MAE={str(round(np.nanmean(abs(data3)), 4))}'},
        'data4': {'cmap': 'Greens', 'vmin': vmin4, 'vmax': vmax4, 'text' : r'$\overline{spread_b}$=' + f'{str(round(np.nanmean(data4), 4))}'},
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
        
        # Thickening the axis lines for the second subplot
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)                  # Set axis line width to 2
        
        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)
        
        # Add annotation in the upper left corner:w
        ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

    # Savefig
    fig.savefig(f'{config.figures_dir}subplots_background_{config.my_exp}_{var_fig}_{config.date}.png', bbox_inches = 'tight', dpi = 300)

def no_obs_maps(vari = 0) :
    
    lon, lat = lon_lat()

    data_model_ensb = read_model_matrix(0, vari)
    data_model_ensa = read_model_matrix(1, vari)
    
    new_mask = read_mask()

    # Mean
    data1 = np.ma.masked_array(np.nanmean(list(data_model_ensb.values()), axis = 0), mask = new_mask)  
    # Standard deviation before analysis
    data2 = np.ma.masked_array(np.nanstd(list(data_model_ensb.values()), axis = 0), mask = new_mask)
    # Increment
    data3 = np.ma.masked_array((np.nanmean(list(data_model_ensa.values()), axis = 0) - np.nanmean(list(data_model_ensb.values()), axis = 0)), mask = new_mask)
    # Standard deviation after analysis
    data4 = np.ma.masked_array(np.nanstd(list(data_model_ensa.values()), axis = 0), mask = new_mask)
    
    var_unit = ''
    if vari == 0 :
        vmin1, vmax1 = 0, 1
        vmin2, vmax2 = 0, 0.3
        vmin3, vmax3 = -0.5, 0.5
        vmin4, vmax4 = 0, 0.3
        var_title = 'SIC'
    elif vari == 1 :
        vmin1, vmax1 = 0, 3
        vmin2, vmax2 = 0, 0.2
        vmin3, vmax3 = -0.02, 0.02
        vmin4, vmax4 = 0, 0.2
        var_title = 'SIT'
        var_unit = '(m)'
    elif vari == 2 :
        vmin1, vmax1 = 0, 0.2
        vmin2, vmax2 = 0, 0.1
        vmin3, vmax3 = -0.01, 0.01
        vmin4, vmax4 = 0, 0.1
        var_title = 'hs'
    elif vari == 8 :
        vmin1, vmax1 = 40, 750
        vmin2, vmax2 = 0, 200
        vmin3, vmax3 = -50, 50
        vmin4, vmax4 = 0, 200
        var_title = 'Ice age' #config.varnames[vari]
        var_unit = '(days)'
    elif vari == 9 :
        vmin1, vmax1 = 0, 1
        vmin2, vmax2 = 0, 0.3
        vmin3, vmax3 = -0.1, 0.1
        vmin4, vmax4 = 0, 0.3
        var_title = config.varnames[vari]        
        

    if 'sic' in config.assim :
        # List of data arrays and axes for iteration
        data_list = [data1, data2]
        # List of titles
        titles = [ fr'{var_title} ($\overline{{x_b}}$) {var_unit}', fr'{var_title} spread$_b$ {var_unit}', f'{var_title} increment {var_unit}', fr'{var_title} spread$_a$ {var_unit}']   
        # Dictionary of visualization parameters
        viz_params = {
            'data1': {'cmap': cmocean.cm.ice, 'vmin': vmin1, 'vmax': vmax1, 'text' : None},
            'data2': {'cmap': 'Greens', 'vmin': vmin2, 'vmax': vmax2, 'text' : r'$\overline{spread_b}$=' + f'{str(round(np.nanmean(data2), 4))}'}
        }
    
        # Background maps
        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(1, 2, figsize = (10, 5), subplot_kw = {'projection': ccrs.NorthPolarStereo()})
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

            # Thickening the axis lines for the second subplot
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)                  # Set axis line width to 2

            # Add a colorbar for each subplot
            cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
            cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
            cbar.ax.tick_params(labelsize = 14)

            # Add annotation in the upper left corner
            ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
                fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

        # Savefig
        fig.savefig(f'{config.figures_dir}ssubplots_background_{config.my_exp}_{config.varnames[vari]}_{config.date}.png', bbox_inches = 'tight', dpi = 300)
        
    # Analysis maps
    # List of data arrays and axes for iteration
    data_list = [data3, data4]
    # List of titles
    titles = [f'{var_title} increment {var_unit}', fr'{var_title} spread$_a$ {var_unit}']
    # Dictionary of visualization parameters
    viz_params = {
        'data3': {'cmap': cmocean.cm.balance, 'vmin': vmin3, 'vmax': vmax3, 'text' : r'$\overline{incr}$=' + f'{str(round(np.nanmean(abs(data3)), 4))}'},
        'data4': {'cmap': 'Greens', 'vmin': vmin4, 'vmax': vmax2, 'text' : r'$\overline{spread_a}$=' + f'{str(round(np.nanmean(data4), 4))}'}
    }
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), subplot_kw = {'projection': ccrs.NorthPolarStereo()})
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
        
        # Thickening the axis lines for the second subplot
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)                  # Set axis line width to 2

        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)

        # Add annotation in the upper left corner
        ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

    # Savefig
    fig.savefig(f'{config.figures_dir}ssubplots_analysis_{config.my_exp}_{config.varnames[vari]}_{config.date}.png', bbox_inches = 'tight', dpi = 300)

def analysis_maps(vari = 0) :
    
    lon, lat = lon_lat()
    sic_sat = read_obs()
    sic_model_ensb = read_model_matrix(0, vari)
    new_mask = read_mask()
  
    if 'tb' in config.my_exp : opta = 2
    else : opta = 1
    sic_model_ensa = read_model_matrix(opta, vari)
    #  (ensa - ensb)
    data2 = np.ma.masked_array((np.nanmean(list(sic_model_ensa.values()), axis = 0) - np.nanmean(list(sic_model_ensb.values()), axis = 0)), mask = new_mask)
    # Spread ensa
    data3 = np.ma.masked_array(np.nanstd(list(sic_model_ensa.values()), axis = 0), mask = new_mask)
    # Mean bias
    data4 = np.nanmean(list(sic_model_ensa.values()), axis = 0) - sic_sat
    mad_xb = np.nanmean(abs(np.nanmean(list(sic_model_ensb.values()), axis = 0) - sic_sat))
    mad_xa = np.nanmean(abs(data4))
    incr = np.nanmean(abs(data2))

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(1, 3, figsize = (15, 8), subplot_kw={'projection': ccrs.NorthPolarStereo()})

    # List of data arrays and axes for iteration
    data_list = [data2, data3, data4]
    
    if vari == 0 :
        vmin2, vmax2 = -0.5, 0.5
        vmin3, vmax3 = 0, 0.3
        vmin4, vmax4 = -0.5, 0.5
        var_title = 'SIC'
        
    elif vari == 1 :
        vmin2, vmax2 = -0.005, 0.005
        vmin3, vmax3 = 0, 0.3
        vmin4, vmax4 = -0.5, 0.5
        var_title = 'SIT'
        
    titles = [f'{var_title} increment', fr'TOPAZ {var_title} spread$_a$', f'Updated {var_title} bias']
    
    # Dictionary of visualization parameters
    viz_params = {
        'data2': {'cmap': cmocean.cm.balance, 'vmin': vmin2, 'vmax': vmax2, 'text' : r'$\overline{incr}$=' + f'{str(round(np.nanmean(abs(data2)), 4))}'},
        'data3': {'cmap': 'Greens', 'vmin': vmin3, 'vmax': vmax3, 'text' : r'$\overline{spread_a}$=' + f'{str(round(np.nanmean(data3), 4))}'},
        'data4': {'cmap': cmocean.cm.balance, 'vmin': vmin4, 'vmax': vmax2, 'text' : f'MAE={str(round(np.nanmean(abs(data4)), 4))}'}
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
        
        # Thickening the axis lines for the second subplot
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)                  # Set axis line width to 2

        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)

        # Add annotation in the upper left corner
        ax.text(0.02, 0.98, params['text'], transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5))

    # Savefig
    fig.savefig(f"{config.figures_dir}subplots_analysis_{config.my_exp}_{config.varnames[vari]}_{config.date}.png", bbox_inches = 'tight', dpi = 300)  

def interpolation_metrics(var, lon, lat) :
    
    from scipy.interpolate import griddata

    # Flatten the latitude and longitude arrays
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()

    # Create target grid coordinates based on the shape of lon and lat
    lon_target, lat_target = np.meshgrid(np.linspace(lon.min(), lon.max(), lon.shape[1]),
                                         np.linspace(lat.min(), lat.max(), lat.shape[0]))

    # Flatten the var array and create corresponding x, y coordinates
    var_flat = var.ravel()
    var_lon = np.linspace(lon.min(), lon.max(), var.shape[1])
    var_lat = np.linspace(lat.min(), lat.max(), var.shape[0])
    var_lon, var_lat = np.meshgrid(var_lon, var_lat)

    # Interpolate var onto the target grid
    var_interpolated = griddata(
        (var_lon.ravel(), var_lat.ravel()), 
        var_flat, 
        (lon_target, lat_target), 
        method='linear'
    )
    
    return var_interpolated

def plot_metrics() :
    
    new_mask = read_mask()
    lon, lat = lon_lat()
    
    for diag_file in glob.glob(f'{config.storage_dir}/enkf_files/*diag*nc'):
        print("\nDo we need to modify any EnKF parameters in the obstypes.prm file?")
        update_srf, update_dfs = check_dfs_srf(config.Nens, diag_file)
    
    data = netCDF4.Dataset(diag_file, mode = 'r')

    list_metrics = ['dfs', 'srf']

    # Making subplots
    fig, axes = plt.subplots(1, 2, figsize = (10, 5), subplot_kw = {'projection': ccrs.NorthPolarStereo()})
    for ax, metric in zip(axes.flatten(), list_metrics):
        # Set the map extent
        ax.set_extent([-180, 180, 45, 90], crs = ccrs.PlateCarree())
        
        # Add land with light gray color
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor = 'face',
                                            facecolor = 'lightgray')
        ax.add_feature(land, zorder = 1)

        var_interp = interpolation_metrics(data[metric][:], lon, lat)
        var_interp_masked = np.ma.masked_array(var_interp, mask = new_mask) 
        # Plot the data with the appropriate transform
        mesh = ax.pcolormesh(lon, lat, var_interp_masked, transform = ccrs.PlateCarree(), zorder = 2,
                             cmap = cmocean.cm.rain, vmin = 0, vmax = 3) # Greens

        # Add coastlines and gridlines
        ax.coastlines(resolution = '50m', color = 'gray', linewidth = 0.5, zorder = 3)
        ax.gridlines()

        # Add a title to each subplot
        ax.set_title(metric.upper(), fontsize = 16)
        
        # Thickening the axis lines for the second subplot
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)                  # Set axis line width to 2

        # Add a colorbar for each subplot
        cbar = plt.colorbar(mesh, ax = ax, orientation = 'horizontal', fraction = 0.046, pad = 0.04)
        cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
        cbar.ax.tick_params(labelsize = 14)

        # Add annotation in the upper left corner
        vmin, vmax = np.nanmin(var_interp_masked), np.nanmax(var_interp_masked)
        vmean = np.nanmean(var_interp_masked)
        ax.text(0.02, 0.98, fr'$\overline{{{metric.upper()}}}$ = {vmean:.3f}', transform = ax.transAxes,
            fontsize = 12, verticalalignment = 'top', bbox = dict(facecolor = 'white', alpha = 0.5)) 
    ratio = data['dfs'][:]/data['srf'][:];
    print('Mean ratio dfs/srf: ', np.nanmean(ratio))
    print('Mean dfs:', np.nanmean(data['dfs'][:]))
    print('Mean srf: ', np.nanmean(data['srf'][:]))
    # Suptitle
    if 'sic' in config.assim : suptitle_text = 'EnKF metrics (assimilation of SIC)'
    elif 'tb' in config.assim : suptitle_text = 'EnKF metrics (assimilation of Tb)'
    plt.suptitle(suptitle_text, fontsize = 16, y = 1.)

    plt.subplots_adjust(wspace=0.1)
    
    # Savefig
    fig.savefig(f"{config.figures_dir}subplots_metrics_{config.my_exp}_{config.date}.png", bbox_inches = 'tight', dpi = 300)
