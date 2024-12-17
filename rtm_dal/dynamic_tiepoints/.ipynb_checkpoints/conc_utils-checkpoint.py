#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os, sys
import numpy as np
from math import ceil
from netCDF4 import Dataset
import pyresample as pr


def timedelta_total_seconds(td):
    """
    Implements timedelta.total_seconds() as available from Py2.7

    :param td: delta time
    :type td: datetime.timedelta object
    :returns: time difference in total number of seconds

    """
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

def timedelta_total_hours(td):
    """

    :param td: delta time
    :type td: datetime.timedelta object
    :returns: time difference in total number hours

    """
    return timedelta_total_seconds(td) / 60. / 60.

def downsample_index(current_size, target_size):
    """
       returns indexes for accessing exactly 'target_size' elements
       from an array with originally 'current_size' elements
    """
    #print "Start with %d, downsample to %d" % (current_size,target_size,)
    indx = list(range(current_size))

    if ( current_size < target_size or target_size == -1 ):
        return indx

    sel_indx = []
    cpt = 0
    while ((len(sel_indx) != target_size) and (len(indx) > 0)):
        #print "beg i=%d indx=%s" % (cpt, indx,)
        skip = int(ceil(float(len(indx))/(target_size-len(sel_indx))))
        loc_sel_indx = indx[::skip]
        sel_indx.extend(loc_sel_indx)
        indx = list(set(indx) - set(loc_sel_indx))
        #print "end i=%d, skip=%d, sel=%s, indx=%s, sel_len=%d" % (cpt, skip, sel_indx, indx, len(sel_indx))
        cpt += 1

    return sel_indx

def interpolate_1d_linear(x_fit, y_fit, x_eval,):
    """ returns the same as:
            F = scipy.interpolate.interp1d(x_fit, y_fit, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_eval = F(x_eval)
    """
    # get to numpy arrays
    x_fit = np.asarray(x_fit)
    y_fit = np.asarray(y_fit)
    x_eval = np.asarray(x_eval)
    # sanity checks
    if len(x_fit) != len(y_fit):
        raise ValueError("x_fit and y_fit must have the same number of elements (length).")
    if len(x_fit) < 2:
        raise ValueError("x_fit must have at least 2 elements.")
    if not all(x_fit[i] < x_fit[i+1] for i in range(len(x_fit)-1)):
        raise ValueError("x_fit must be strictly increasing without repetition")
    # first, do interpolation for values within the range
    y_eval = np.empty_like(x_eval)
    slopes  = np.zeros(2)
    offsets = np.zeros(2)
    for i in range(len(x_fit)-1):
        x_m, x_M = x_fit[i:i+2]
        indx = (x_eval >= x_m) * (x_eval < x_M)
        if indx.sum():
            y_m, y_M = y_fit[i:i+2]
            slope  = (y_M - y_m) / (x_M - x_m)
            offset = y_m - slope * x_m
            y_eval[indx] = slope * x_eval[indx] + offset
            # store coeffs for first and last segment to ease extrapolation
            if i == 0:
                slopes[0] = slope
                offsets[0] = offset
            elif i == (len(x_fit)-2):
                slopes[1] = slope
                offsets[1] = offset

    # second, do extrapolation
    indx = x_eval < x_fit[0]
    y_eval[indx] = slopes[0] * x_eval[indx] + offsets[0]
    indx = x_eval >= x_fit[-1]
    y_eval[indx] = slopes[1] * x_eval[indx] + offsets[1]
    # return
    return y_eval


def load_grid_defs_from_OSISAF_ncCF_file(ncCF_file,):
    """ Searches for a variable with attribute 'grid_mapping'. Use the value of that attribute
    to load proj string information, and then use xc and yc for defining the extent
    of the pyresample area_def object. """

    def get_area_id(area, projection, resolution):
        reso = int(float(resolution)*10)
        area_id = '{}{}'.format(area,reso,)
        pcs_id = '{}_{}-{}'.format(area, projection, reso)
        return area_id, pcs_id

    with Dataset(ncCF_file,"r") as gf:
        # proj_dict
        crs_name = 'crs'
        for varn in gf.variables.keys():
            try:
                a_var     = gf.variables[varn]
                crs_name  = a_var.grid_mapping
                break
            except AttributeError:
                pass
        if crs_name is None:
            raise ValueError("ERROR: did not find any variable with 'grid_mapping' attribute")
        else:
            print("Info: read grid_mapping information from {} (found in {})".format(crs_name,varn))

        try:
            proj_str  = gf.variables[crs_name].proj4_string
        except KeyError:
            proj_str  = gf.variables['crs'].proj4_string

        proj_dict = dict([s[1:].split('=') for s in proj_str.split( )])
        # xcs
        xcs = gf.variables['xc']
        if xcs.units == 'km':
            xcs = xcs[:] * 1000.
        elif xcs.units == 'm':
            xcs = xcs[:]
        else:
            raise NotImplementedError("Do not know how to handle 'xc' with units {}".format(xcs.units))
        # ycs
        ycs = gf.variables['yc']
        if ycs.units == 'km':
            ycs = ycs[:] * 1000.
        elif ycs.units == 'm':
            ycs = ycs[:]
        else:
            raise NotImplementedError("Do not know how to handle 'yc' with units {}".format(ycs.units))
        # shape
        shape = a_var.shape[::-1]
        if len(shape) == 3:
            shape = (shape[0],shape[1])
        # grid spacing
        xgs = abs(xcs[1]-xcs[0])
        ygs = abs(ycs[1]-ycs[0])
        # extent
        extent = [xcs.min(),ycs.min(),xcs.max(),ycs.max(),]
        extent[0] = extent[0] - 0.5*xgs
        extent[1] = extent[1] - 0.5*ygs
        extent[2] = extent[2] + 0.5*xgs
        extent[3] = extent[3] + 0.5*ygs
        
        # try and guess an area_id from global attributes:
        area_id, pcs_id = crs_name, crs_name
        try:
            area_id, pcs_id  = get_area_id(gf.area,gf.projection,gf.resolution.split(' ')[0])
        except AttributeError as ae:
            print("Missing global attribute {} to create an area ID".format(ae,))
        except Exception as ex:
            print("Got {}".format(ex,))
            pass
        
        area_def  = pr.geometry.AreaDefinition(area_id,crs_name,pcs_id,
                                        proj_dict,
                                        shape[0],shape[1],
                                        extent)
        return area_def

def get_chn_from_algo(algorithm):
    """ Return list of channels needed for algorithm of type ALGO1_corrALGO2
    
        Example:
        N90LIN_SICCI3LF (90v/90v for N90LIN + 19v/37v/37h for SICCI3LF)
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import dynamic_tiepoints as dtp
    
    algo = algorithm.lower().split('_')
    if ( len(algo) != 2 ):
        print('\nInvalid algo format for get_chn_from_algo: {}\n'.format(algorithm))
        return None

    try:
        ct1,_ = dtp.DynIceConcAlgoFactory.get_ct_specs(algo[0])
        ct2,_ = dtp.DynIceConcAlgoFactory.get_ct_specs(algo[1])
        # Don't distinguish between pca3d:best_ow/nD:best_ow/pca3d:best_cice/nD:best_cice/ch1xch2
        # for now. Assume same channels.
        chns1 = ct1[0]['channels']
        chns2 = ct2[0]['channels']
    except NotImplementedError as e:
        return None

    return (chns1, chns2)
