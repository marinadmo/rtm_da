# Figures and diagnostics before and after DA
import matplotlib.pyplot as plt
import cmocean
#
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .reading_data import *

def pretty_cbar(fig, cs, cbar_ax, cbar_label = "") :    
    cbar = fig.colorbar(cs, cax = cbar_ax, orientation = 'vertical', pad = 0.001)
    cbar.ax.tick_params(labelsize = 10)
    cbar.outline.set_linewidth(1.5)  # Set colorbar border thickness
    cbar.ax.tick_params(labelsize = 14)
    
    if cbar_label != "" :
        cbar.set_label(cbar_label, fontsize = 12, labelpad = 20)
        cbar.ax.yaxis.label.set_rotation(270)

def subregion_plot(ax) :
    ax.set_extent([-180, 180, 45, 90], crs = ccrs.PlateCarree())

    # Add land with light gray color
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor = 'face',
                                            facecolor = 'lightgray')
    ax.add_feature(land, zorder = 1)

    # Add coastlines and gridlines
    ax.coastlines(resolution = '50m', color = 'gray', linewidth = 0.5, zorder = 3)
    ax.gridlines()

    # Thickening the axis lines for the second subplot
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)     
            
def subplots_rtm_tbs() :    
    new_mask = read_mask()

    fig, axes = plt.subplots(2, 4, figsize = (12, 6), subplot_kw = {'projection': ccrs.NorthPolarStereo()})
    for ich in range(0, len(config.channels)) :
        tb_model_ens = read_model_matrix_tb(ich)
        tb_sat = read_obs_tb(ich)
    
        mean = np.ma.masked_array(np.nanmean(list(tb_model_ens.values()), axis = 0), mask = new_mask)
        diff = np.nanmean(list(tb_model_ens.values()), axis = 0) - tb_sat
    
        cs1 = axes[0, ich].pcolormesh(lon_lat()[0], lon_lat()[1], mean, transform = ccrs.PlateCarree(), zorder = 2, cmap = cmocean.cm.thermal, vmin = 120, vmax = 260)
        cs2 = axes[1, ich].pcolormesh(lon_lat()[0], lon_lat()[1], diff, transform = ccrs.PlateCarree(), zorder = 2, cmap = cmocean.cm.balance, vmin = -20, vmax = 20)

        subregion_plot(axes[0, ich]); subregion_plot(axes[1, ich])
        
        axes[0, ich].set_title(config.channels[ich][2::], fontsize = 12)
        # Model spread
        #data4 = np.ma.masked_array(np.nanstd(list(sic_model_ens.values()), axis = 0), mask = new_mask)
    # Add colorbars
    # Create an axis for the common colorbar
    cbar_ax1 = fig.add_axes([0.93, 0.535, 0.015, 0.335]); pretty_cbar(fig, cs1, cbar_ax1, cbar_label = "Tb$_{RTM}$ (K)")
    cbar_ax2 = fig.add_axes([0.93, 0.115, 0.015, 0.335]); pretty_cbar(fig, cs2, cbar_ax2, cbar_label = "Tb$_{RTM}$ - Tb$_{AMSR2}$ (K)")
    
    plt.subplots_adjust(hspace = 0.1, wspace = 0.2)
    
    # Savefig
    fig.savefig(f"{config.figures_dir}subplots_rtm_{config.date}.png", bbox_inches = 'tight', dpi = 300)

def hist_fct(data, ich, axb, thecolor, label) :
    rmin = (170, 80, 180, 120)
    rmax = (270, 255, 270, 250)
    h1 = axb.hist(data.flatten(), bins = 100, color = thecolor, range = [rmin[ich], rmax[ich]], alpha = 0.6, edgecolor = 'k', label = label)  
    axb.tick_params(axis = 'both', labelsize = 16)
    for axis in ['top', 'bottom', 'left', 'right']: 
        axb.spines[axis].set_linewidth(2)
        
def plot_histograms() :    
    lab_title = ("AMSR2", "RTM") 
    
    fig, axes = plt.subplots(4, 2,  sharey = 'row', figsize = (23, 18), constrained_layout = True)
    
    for ich in range(0, len(config.channels)) :
        hist_fct(read_obs_tb(ich), ich, axes[ich, 0], 'k', label = config.channels[ich][2::])
        
        rtm_mean = np.ma.masked_array(np.nanmean(list(read_model_matrix_tb(ich).values()), axis = 0), mask = read_mask())
        hist_fct(rtm_mean, ich, axes[ich, 1], 'green', label = config.channels[ich][2::])
        
        ymax = max(axes[ich, 0].get_ylim()[1], axes[ich, 1].get_ylim()[1]) 
        axes[ich, 0].set_ylim([0, ymax]); axes[ich, 1].set_ylim([0, ymax])
        
        axes[ich, 0].ticklabel_format(axis = 'y', scilimits = [-3, 3]); 
        axes[ich, 0].yaxis.major.formatter._useMathText = True
        axes[ich, 0].set_ylabel(str(config.channels[ich][2::]), fontsize = 22)   
        
    for icol in range(0, 2) :
        axes[0, icol].set_title(lab_title[icol], fontsize = 24);
        axes[3, icol].set_xlabel("Tb (K)", fontsize = 22)
    
def ow_area(sic_data, data_tb_ow, sic_low = 0.05) :
    dpt = 10 # Only every 10 points
    msic = sic_data[0::dpt, 0::dpt].flatten()
    data_tb_ow[msic > sic_low] = np.nan
    return data_tb_ow
    
def ice_area(sic_data, data_tb_ice, sic_high = 0.95) :
    dpt = 10 # Only every 10 points
    msic = sic_data[0::dpt, 0::dpt].flatten()
    data_tb_ice[msic < sic_high] = np.nan
    return data_tb_ice

def matrix_diagrams() :    
    sic_data = read_obs()
    dpt = 10 # Only every 10 points
    ll = len(sic_data[0::dpt, 0::dpt].flatten())
    ow_matrix_obs = np.zeros((len(config.channels), ll)); ice_matrix_obs = np.zeros((len(config.channels), ll))
    ow_matrix_rtm = np.zeros((len(config.channels), ll)); ice_matrix_rtm = np.zeros((len(config.channels), ll))
    for ich in range(0, 4) :
        # Observations
        ow_matrix_obs[ich, :] = ow_area(sic_data, read_obs_tb(ich)[0::dpt, 0::dpt].flatten())
        ice_matrix_obs[ich, :] = ice_area(sic_data, read_obs_tb(ich)[0::dpt, 0::dpt].flatten())
        # RTM simulation
        ow_matrix_rtm[ich, :] = ow_area(sic_data, read_obs_tb(ich)[0::dpt, 0::dpt].flatten())
        ice_matrix_rtm[ich, :] = ice_area(sic_data, read_obs_tb(ich)[0::dpt, 0::dpt].flatten())
    
    return ow_matrix_obs, ow_matrix_rtm, ice_matrix_obs, ice_matrix_rtm

def plot_diagrams() :
    sic_low = 0.05; sic_high = 0.95
    ow_matrix_obs, ow_matrix_rtm, ice_matrix_obs, ice_matrix_rtm = matrix_diagrams()        
        
    fig, axes = plt.subplots(1, 2, figsize = (14, 7)) 
    
    # Observations
    axes[0].scatter(ow_matrix_obs[2, :], ow_matrix_obs[0, :], marker = 'o', color = 'royalblue') 
    axes[0].scatter(ice_matrix_obs[2, :], ice_matrix_obs[0, :], marker = 'o', color = 'gray') 
    axes[1].scatter(ow_matrix_obs[2, :], ow_matrix_obs[3, :], marker = 'o', color = 'royalblue') 
    axes[1].scatter(ice_matrix_obs[2, :], ice_matrix_obs[3, :], marker = 'o', color = 'gray') 
    axes[0].set_xlim([160, 280]); axes[1].set_xlim([160, 280])

    # Observed_tb simul_data_obs
    axes[0].scatter(ow_matrix_rtm[2, :], ow_matrix_rtm[0, :], marker = 'o', facecolor = 'powderblue', alpha = 0.1)
    axes[0].scatter(ice_matrix_rtm[2, :], ice_matrix_rtm[0, :], marker = 'o', facecolor = 'silver', alpha = 0.1)
    axes[1].scatter(ow_matrix_rtm[2, :], ow_matrix_rtm[3, :], marker = 'o', facecolor = 'powderblue', alpha = 0.1)
    axes[1].scatter(ice_matrix_rtm[2, :], ice_matrix_rtm[3, :], marker = 'o', facecolor = 'silver', alpha = 0.1)

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    # Define the desired linewidth
    line_width = 6
    # Create custom legend handles with two lines each
    legend_lines = [
        (Line2D([0], [0], color='royalblue', lw=line_width),
         Line2D([0], [0], color='powderblue', lw=line_width)),

        (Line2D([0], [0], color='gray', lw=line_width),
         Line2D([0], [0], color='silver', lw=line_width))
    ]
    legend_lines2 = [
        (Line2D([0], [1], color='royalblue', lw=line_width),
         Line2D([0], [1], color='gray', lw=line_width)),

        (Line2D([0], [1], color='powderblue', lw=line_width),
         Line2D([0], [1], color='silver', lw=line_width))
    ]

    # Adding the legend with the custom handles
    leg = axes[0].legend(
        handles = legend_lines,
        labels = [f'SIC ≤ {sic_low}', f'SIC ≥ {sic_high}'],
        handler_map = {tuple: HandlerTuple(ndivide = None)},
        fontsize = 16,
        loc = 'lower right',
        bbox_to_anchor = (0.45, 0.25),
        ncol = 1
    )
    leg2 = axes[1].legend(
        handles = legend_lines2,
        labels = ['AMSR2', 'RTM'],
        handler_map = {tuple: HandlerTuple(ndivide = None)},
        fontsize = 16,
        loc = 'lower right',
        bbox_to_anchor = (-0.93, 0.100),
        ncol = 1
    )

    chx = 2; chy = 0; # x:tb37v, y:tb19v 2, 0
    chx2 = 2; chy2 = 3 # x:37v, y: 37h 2, 3 
    
    axes[0].set_xlabel(f'Tb{config.channels[chx][2::]} (K)', fontsize = 17, labelpad = 15, fontweight='bold') 
    axes[0].set_ylabel(f'Tb{config.channels[chy][2::]} (K)', fontsize = 17, labelpad = 15, fontweight='bold') 
    axes[1].set_xlabel(f'Tb{config.channels[chx][2::]} (K)', fontsize = 17, labelpad = 15, fontweight='bold') 
    axes[1].set_ylabel(f'Tb{config.channels[chy2][2::]} (K)', fontsize = 17, labelpad = 15, fontweight='bold')
    
    for axis in ['top','bottom','left','right']: axes[0].spines[axis].set_linewidth(2); axes[1].spines[axis].set_linewidth(2)
    axes[0].tick_params(axis = 'both', labelsize = 16); axes[1].tick_params(axis = 'both', labelsize = 16) 
    
    plt.subplots_adjust(wspace = 0.3)

    # Savefig
    fig.savefig(f"{config.figures_dir}2d_diagrams_rtm_{config.date}.png", bbox_inches = 'tight', dpi = 300)