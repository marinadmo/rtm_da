import netCDF4
import numpy as np

def check_dfs_srf(ens_count, diag_file):
    # Check the DFS and SRF values to see if any exceed the recomended values
    # for example if more than 5% of the data has values larger than the limit
    
    file_handle = netCDF4.Dataset(diag_file, mode='r')
    psrf = file_handle.variables['psrf']
    pdfs = file_handle.variables['pdfs']


    srf_lim = 2 
    dfs_lim = ens_count/3 
    
    update_srf = np.zeros(psrf.shape[0])
    update_dfs = np.zeros(psrf.shape[0])
    for i in range(0, psrf.shape[0]) :
        # more than 5% of the data larger than the limit then reduce
        if len(psrf[i,:,:][psrf[i,:,:] > srf_lim])/(psrf.shape[1] * psrf.shape[2]) > 0.05:
            print('Must increase Rfactor for obstype ' + str(i))
            update_srf[i] = 1
        elif len(psrf[i,:,:][psrf[i,:,:] < srf_lim*0.25])/(psrf.shape[1] * psrf.shape[2]) > 0.99:
            print('Must decrease R_factor for obstype ' + str(i))
            update_srf[i] = -1
        else: print('No need to modify R_factor for obstype ' + str(i))
        if len(pdfs[i,:,:][pdfs[i,:,:] > dfs_lim])/(pdfs.shape[1] * pdfs.shape[2]) > 0.05:
            print('Must reduce loc_rad for obstype ' + str(i))
            update_dfs[i] = 1
        elif len(pdfs[i,:,:][pdfs[i,:,:] < srf_lim*0.25])/(pdfs.shape[1] * pdfs.shape[2]) > 0.99:
            print('Can increase loc_rad for obstype ' + str(i))
            update_dfs[i] = -1
        else: print('No need to modify loc_rad for obstype ' + str(i))
    # If any must be updated run update prm with information regarding which obstypes that should be updated
    # and how they should be updated. Consider to change back to default values when the assimilation is finished 
    # with satisfying results
    file_handle.close()
    print(update_srf, update_dfs)
    return update_srf, update_dfs

def update_tuning(tuning_file, update_srf, update_dfs):
# To check each locrad and R_factor obstypes.prm should be searched from top to bottom and each time it passes a name
# this name should be rembered such that the next locrad and rfactor encoutered belongs to this name
# It is also important that each obstypes has a locrad specified and an rfractor specified.
# Must investigate a bit more how this should be done in practice, have a list with 0 and 1 per obs type is probably
# the easiest.

    file1 = open(tuning_file, 'r') 
    Lines = file1.readlines()
    ii = -1
    obs_num = -1

    for l in Lines:
        ii += 1
        
        if l[0:4] == 'NAME':
            print(l[7:])
            current_obs = l[7:]
            obs_num += 1
            print(obs_num)
            if obs_num < len(update_srf):
                if update_dfs[obs_num] == 1 and l[0:7] == 'RFACTOR':
                    rf_old = float(l[10:-1])
                    Lines[ii] = 'RFACTOR = '+str(round(rf_old*1.5))+'\n'
                elif update_dfs[obs_num] == -1 and l[0:7] == 'RFACTOR':
                    rf_old = float(l[10:-1])
                    Lines[ii] = 'RFACTOR = '+str(round(rf_old*0.75))+'\n'
                elif update_dfs[obs_num] == -1 and l[0:6] == 'LOCRAD':
                    lr_old = float(l[9:-1])
                    Lines[ii] = 'RFACTOR = '+str(round(lr_old*0.75))+'\n'
                elif update_dfs[obs_num] == -1 and l[0:6] == 'LOCRAD':
                    lr_old = float(l[9:-1])
                    Lines[ii] = 'RFACTOR = '+str(round(lr_old*0.75))+'\n'
    file1.close()
