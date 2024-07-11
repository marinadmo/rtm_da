#!/usr/bin/env python3
# _*_ coding: UTF-8 _*_

import logging
LOG = logging.getLogger(__name__)

import os, sys
import psutil
import shutil
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset, num2date
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rtm_smmr
import rtm_ssmi
import rtm_ssmis
import rtm_amsr
import dynamic_tiepoints as dtp

fill_values = {np.int8:-127, np.int16:-32767, np.float32:-1e10}

def copy_split(infile, outfile):
    """
        Split copy process into first a copy to tmp file and then a rename
    """
    outfile_tmp = outfile + '.tmp'
    try:
        shutil.copy2(infile, outfile_tmp)
        shutil.move(outfile_tmp, outfile)
    except IOError as e:
        print('\niError: Cannot copy {} to {}, aborting.\n'.format(infile, outfile))
        # Try removing potential incomplete copy
        try:
            os.remove(outfile_tmp)
        except OSError as e:
            pass
        sys.exit(2)

    # Check that source and target files are the same size, if not then try copying a second time
    if ( os.stat(infile).st_size != os.stat(outfile).st_size ):
        print('\nWarning: Second attempt to copy {}\n'.format(infile))
        os.remove(outfile)
        outfile_tmp = outfile + '.tmp'
        try:
            shutil.copy2(infile, outfile_tmp)
            shutil.move(outfile_tmp, outfile)
        except IOError as e:
            print('\nError: Cannot re-copy {} to {}, aborting.\n'.format(infile, outfile))
            # Try removing potential incomplete copy
            try:
                os.remove(outfile_tmp)
            except OSError as e:
                pass
            sys.exit(2)


class IONetCDFError(Exception):
    pass

class BaseNetCDFHandler(object):
    """

        Handles basic netcdf file operations, reading and writing variables.

        :param nc_file_path: netcdf file path
        :type nc_file_path: string

    """

    limits = {}

    def __init__(self, nc_file_path):
        self.nc_file_path = nc_file_path

    def __str__(self):
        return self.nc_file_path

    def read(self, variable_name):
        try:
            with Dataset(self.nc_file_path, 'r') as nc_file:
                file_variable = self.get_file_variable(variable_name)
                varnc = nc_file.variables[file_variable]
                # set_auto_maskandscale to forcing application of scale_factor when reading
                varnc.set_auto_maskandscale(True)
                data = varnc[:]
                try:
                    # Make sure data read is always a masked array.
                    data.mask
                except AttributeError as e:
                    data = ma.array(data, mask=np.zeros(data.shape).astype(np.bool))
                        
        except IOError as e:
            raise IOError(e, self.nc_file_path)
        except RuntimeError as e:
            raise RuntimeError(e, self.nc_file_path)
        return data

    def read_clean_variables(self, *variable_names):
        var_list, select_idx = self.read_variables_and_valid_idx(*variable_names)
        if not select_idx.all():
            for i, data in enumerate(var_list):
                if np.ma.is_masked(data):
                    var_list[i] = data.data[select_idx]
                else:
                    var_list[i] = data[select_idx]

        return var_list

    def read_variables_and_valid_idx(self, *variable_names):
        var_list = []
        select_idx_list = []
        for variable_name in variable_names:
            data = self.read(variable_name)
            var_list.append(data)
            try:
                vmin, vmax = self.limits[variable_name]
                select_idx = (data >= vmin) & (data <= vmax)
                select_idx_list.append(select_idx)
            except KeyError as e:
                pass

        if len(select_idx_list) > 0:
            select_idx = select_idx_list[0]
            for next_idx in select_idx_list[1:]:
                select_idx = select_idx & next_idx
        else:
            select_idx = np.ones(var_list[0].shape, dtype=np.bool)

        for var in var_list:
            if np.ma.is_masked(var):
                select_idx = select_idx & (~var.mask)

        return var_list, select_idx

    def has_variable(self, variable_name):
        with Dataset(self.nc_file_path, 'r') as nc_file:
            file_variable = self.get_file_variable(variable_name)
            try:
                nc_file.variables[file_variable]
                return True
            except KeyError as e:
                return False

    def write(self, variable_name, data, update_history=True, auto=False):
        with Dataset(self.nc_file_path, 'a') as nc_file:
            if variable_name not in list(nc_file.variables.keys()):
                # Get info from input variable.
                if ( auto ):
                    dshape = data.shape
                    # If dshape dimensions are not in file then abort.
                    nc_dims = {len(nc_file.dimensions[key]):key for key in list(nc_file.dimensions.keys())}
                    for d in dshape:
                        if ( d not in list(nc_dims.keys()) ):
                            raise IONetCDFError('Length {} dim not in netcdf.'.format(d), nc_file)
                    dtype = data.dtype.type
                    try:
                        fill_value = fill_values[dtype]
                    except KeyError as e:
                        fill_value = None
                    # search through the list of variables if there is one with same shape.
                    #    If yes, use its dimensions.
                    dimensions = None
                    for v in list(nc_file.variables.keys()):
                        if dshape == nc_file.variables[v].shape:
                            dimensions = nc_file.variables[v].dimensions
                            break
                    #    If no, try from nc_dims dict()
                    if dimensions is None:
                        dimensions = tuple([nc_dims[s] for s in dshape])
                    nc_file.createVariable(variable_name, dtype, dimensions, fill_value=fill_value, zlib=True)
                    # No attributes or scale factor/offset.
                else:
                    if ( data.ndim == 2 and data.shape[1] > 4 ):
                        dtype, dimensions, fill_value, scale, offset = \
                            self.get_2D_variable_info(variable_name)
                    else:
                        dtype, dimensions, fill_value, scale, offset = \
                            self.get_variable_info(variable_name)
                    nc_file.createVariable(variable_name, dtype, dimensions, fill_value=fill_value, zlib=True)
                    nc_attrs = self.get_nc_attrs(variable_name)
                    for name, value in nc_attrs:
                        setattr(nc_file.variables[variable_name], name, value)
                    if (scale != 1):
                        setattr(nc_file.variables[variable_name], 'scale_factor', scale)
                    if (offset != 0):
                        setattr(nc_file.variables[variable_name], 'add_offset', offset)

            nc_file.variables[variable_name].set_auto_maskandscale(True)
            nc_file.variables[variable_name][:] = data
            if update_history:
                try:
                    prev_history = getattr(nc_file.variables[variable_name], 'history')
                except AttributeError as e:
                    prev_history = ""
                cmd = ' '.join(sys.argv[:])
                pid_starttime = datetime.utcfromtimestamp(psutil.Process(os.getpid()).create_time())
                history = prev_history+"\n{:%Y-%m-%dT%H:%M:%SZ} : {}".format(pid_starttime, cmd)
                try:
                    setattr(nc_file.variables[variable_name], 'history', history)
                except AttributeError as e:
                    print('\nSkipping history update due to AttributeError ({})'.format(e))

    def writeattr(self, var, attr, value):
        with Dataset(self.nc_file_path, 'a') as nc_file:
            if var not in list(nc_file.variables.keys()):
                raise IONetCDFError('Variable {} not in netcdf, cannot add attribute {}.'.format(var, attr),nc_file)
            else:
                setattr(nc_file.variables[var], attr, value)

    def readvarattr(self, var, attr):
        with Dataset(self.nc_file_path, 'r') as nc_file:
            if var not in list(nc_file.variables.keys()):
                raise IONetCDFError('Variable {} not in netcdf, cannot read attribute :{}'.format(var, attr))
            else:
                try:
                    return getattr(nc_file.variables[var], attr)
                except AttributeError as e:
                    raise IONetCDFError('Attribute :{} not found for variable {}'.format(attr, var))

    def readglobalattr(self, attr):
        with Dataset(self.nc_file_path, 'r') as nc_file:
            try:
                return getattr(nc_file, attr)
            except AttributeError as e:
                raise IONetCDFError('Global attribute :{} not present'.format(attr,))

    def writeglobalattr(self, attr, value):
        with Dataset(self.nc_file_path, 'a') as nc_file:
            #setattr(nc_file, attr, value)
            # Workaround for bug in netCDF4 4.1.x (fixed in 4.2)
            nc_file.tmpattr = value
            setattr(nc_file, attr, value)
            del nc_file.tmpattr

    def get_file_variable(self, variable_name):
        #Overwrite if necessary
        return variable_name

    def scale(self, variable_name):
        raise NotImplementedError('Method "scale" is deprecated. Should use get_variable_info() instead.')

    def get_variable_info(self, variable_name):
        raise NotImplementedError('Method "get_variable_info" not '
                                  'implemented in base class')

    def get_nc_attrs(self, variable_name):
        raise NotImplementedError('Method "get_nc_attrs" not '
                                  'implemented in base class')


class OSIBaseNetCDFHandler(BaseNetCDFHandler):
    """

    Class for all PMR swath data (AMSR-E, AMSR2, SSM/I, SSMIS, SMMR,...)

    This class defines some top-level methods, that can be overwritten by the per-instrument
        subclasses (below in this file)

    """

    def get_variable_info(self, variable_name):
        """
           Get dtype, dimensions, fillvalue and scale_factor for a variable
              based on its name.

           The default behaviour is to return a float, with scale_factor 1.
        """
        variable_name = variable_name.lower()
        if (variable_name.startswith('dtb85') or variable_name.startswith('ct_n90_') or \
                variable_name.startswith('stddev_ct_n90')):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            offset = 0
            dimensions = ('ni', 'n85')
        elif (variable_name == 'wf_NASA_n90'):
            dtype = np.int8
            fillvalue = -1
            offset = 0
            scale = 1
            dimensions = ('ni', 'n85')
        elif (variable_name.startswith('ct_') or variable_name.startswith('cmfraq') or \
              variable_name.startswith('dtb') or variable_name.startswith('stddev')):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            offset = 0
            dimensions = ('ni', )
        elif (variable_name in ['tcwv', 'wind_speed', 'air_temp', 'tclw', 'skt']):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            dimensions = ('ni', )
            # Add offset in case air_temp is above 327K
            # tcwv and wind_speed can't be <0 so it won't cause problems for them
            # (and they can't be over 327 + 273 either so int works for them)
            # CHECK IF TCWV can be over 327 + 273 kg / m^2 !!!
            if variable_name in ('air_temp', 'skt'):
                offset = 273
            else:
                offset = 0
        elif (variable_name.startswith('wf_')):
            dtype = np.int8
            fillvalue = -127
            offset = 0
            scale  = 1
            dimensions = ('ni', )
        elif (variable_name.startswith('index')):
            dtype = np.int8
            fillvalue = -127
            offset = 0
            scale = 1
            dimensions = ('ni', )
        else:
            dtype = np.float32
            fillvalue = -1e10
            scale = 1
            offset = 0
            dimensions = ('ni', )
        return dtype, dimensions, fillvalue, scale, offset


    def get_2D_variable_info(self, variable_name):
        variable_name = variable_name.lower()
        if ( variable_name.startswith('dtb85') or ('ct_n90' in variable_name) ):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            offset = np.int16(0)
            dimensions = (self.scnl_h, self.scnp_h)
        elif ( np.array([a in variable_name for a in dtp.HF_ALGOS]).any() ):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            offset = np.int16(0)
            dimensions = (self.scnl_h, self.scnp_h)
        elif (variable_name in ['wf_nasa_n90', 'indexarray_h', 'wf_ucorr@90', 'wf_corrsicci2lf@90']):
            dtype = np.int8
            fillvalue = -127
            offset = np.int8(0)
            scale = np.float32(1)
            dimensions = (self.scnl_h, self.scnp_h)
        elif (variable_name.startswith('tcwv') or variable_name.startswith('wind_speed')
            or variable_name.startswith('air_temp') or variable_name.startswith('tclw')
            or variable_name.startswith('skt') ):
            scale = np.float32(0.01)
            dtype = np.int16
            fillvalue = -32767
            if ( '@tb90' in variable_name ):
                dimensions = (self.scnl_h, self.scnp_h)
            else:
                dimensions = (self.scnl_l, self.scnp_l)
            # Add offset in case air_temp is above 327K
            # tcwv and wind_speed can't be <0 so it won't cause problems for them
            # (and they can't be over 327 + 273 either so int works for them)
            # CHECK IF TCWV can be over 327 + 273 kg / m^2 !!!
            if ( variable_name.startswith('air_temp') or variable_name.startswith('skt') ):
                offset = np.int16(273)
            else:
                offset = np.int16(0)
        elif (variable_name in ['wf_nasa', 'indexarray', 'wf_ucorr', 'wf_corrsicci2lf',]):
            dtype = np.int8
            fillvalue = -127
            offset = np.int8(0)
            scale  = np.float32(1)
            dimensions = (self.scnl_l, self.scnp_l)
        elif ( '@tb37' in variable_name ):
            dtype = np.float32
            fillvalue = -1e10
            offset = np.float32(0)
            scale  = np.float32(1)
            dimensions = (self.scnl_l, self.scnp_l)
        elif ( '@tb90' in variable_name ):
            dtype = np.float32
            fillvalue = -1e10
            offset = np.float32(0)
            scale  = np.float32(1)
            dimensions = (self.scnl_h, self.scnp_h)
        else:
            dtype = np.float32
            fillvalue = -1e10
            scale = np.float32(1)
            offset = np.float32(0)
            dimensions = (self.scnl_l, self.scnp_l)
        return dtype, dimensions, fillvalue, scale, offset


    def get_nc_attrs(self, variable_name):
        if variable_name == 'ct_NASA':
            return [('units', '%'),
                    ('long_name',
                     'Uncorrected total ice concentration using NASA Team')]
        elif variable_name == 'wf_NASA':
            return [('units','1'),
                    ('long_name',
                     'Weather filter from Cavalieri et al. (1992)'),
                    ('comment',
                     '1: Probably OW, 0: Probably ICE')]
        elif variable_name == 'ct_NASA_wWF':
            return [('units', '%'),
                    ('long_name',
                     'Uncorrected total ice concentration using NASA Team, screened by Weather Filter')]
        elif variable_name == 'cmfraq_NASA':
            return [('units', '%'),
                    ('long_name',
                     'Uncorrected total Multi-Year ice fraction using NASA Team')]
        elif variable_name == 'cmfraq_NASA_wWF':
            return [('units', '%'),
                    ('long_name',
                     'Uncorrected total Multi-Year ice fraction using NASA Team, screened by Weather Filter')]
        elif variable_name == 'ct_osi_hybrid':
            return [('units', '%'),
                    ('long_name',
                     'Corrected total ice concentration using OSI SAF fcomiso and Bristol hybrid')]
        elif variable_name == 'tcwv':
            return [('units', 'kg/m2'),
                    ('long_name',
                    'NWP total column water vapour')]
        elif ( variable_name.startswith('tcwv@') ):
            return [('units', 'kg/m2'),
                    ('long_name',
                    'NWP total column water vapour at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name == 'wind_speed':
            return [('units', 'm/s'),
                    ('long_name',
                     'NWP 10m wind speed')]
        elif ( variable_name.startswith('wind_speed@') ):
            return [('units', 'm/s'),
                    ('long_name',
                     'NWP 10m wind speed at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name == 'surf_temp':
            return [('units', 'K'),
                    ('long_name',
                     'NWP surface temperature')]
        elif ( variable_name.startswith('surf_temp@') ):
            return [('units', 'K'),
                    ('long_name',
                     'NWP surface temperature at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name == 'air_temp':
            return [('units', 'K'),
                    ('long_name',
                     'NWP air temperature (at 2m)')]
        elif ( variable_name.startswith('air_temp@') ):
            return [('units', 'K'),
                    ('long_name',
                     'NWP air temperature (at 2m) at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name == 'tclw':
            return [('units', 'kg/m2'),
                    ('long_name',
                    'NWP total column cloud liquid water')]
        elif ( variable_name.startswith('tclw@') ):
            return [('units', 'kg/m2'),
                    ('long_name',
                    'NWP total column cloud liquid water at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name == 'skt':
            return [('units', 'K'),
                    ('long_name',
                     'NWP skin temperature')]
        elif ( variable_name.startswith('skt@') ):
            return [('units', 'K'),
                    ('long_name',
                     'NWP skin temperature at {}GHz resolution'.format(variable_name[-2:]))]
        elif variable_name.startswith('dtb'):
            parts = variable_name.split('_')
            if ( parts[-1] == 'ucorr' ):
                tmp = ''
            else:
                tmp = ' using {}'.format(parts[-1][4:])
            return [('units', 'K'),
                    ('long_name',
                     '{} correction of BT {} GHz{}'.format(parts[1], variable_name[3:6], tmp))]
        elif variable_name.startswith('ct_n90_'):
            parts = variable_name.split('_')
            if (parts[3] == 'ucorr'):
                tmp = 'uncorrected'
            else:
                tmp = variable_name.split('_')[3][4:]
            if parts[-1] == 'wWF':
                WF = ', screened by Weather Filter'
            else:
                WF = ''
            return [('units', '%'),
                    ('long_name',
                     '{} n90 ice concentration using {} brightness temperatures and tie-points{}'.format( \
                     parts[2], tmp, WF))]
        elif variable_name.startswith('ct_'):
            parts = variable_name.split('_')
            if (parts[2] == 'ucorr'):
                tmp = 'uncorrected'
            else:
                tmp = parts[2][4:]
            if parts[-1] == 'wWF':
                WF = ', screened by Weather Filter'
            else:
                WF = ''
            return [('units', '%'),
                    ('long_name',
                     '{} ice concentration using {} brightness temperatures and tie-points{}'.format( \
                     parts[1], tmp, WF))]
        elif variable_name.startswith('wf_'):
            parts = variable_name.split('_')
            if (parts[1] == 'ucorr'):
                tmp = 'uncorrected'
            else:
                tmp = parts[1][4:]
            return [('units', '1'),
                    ('long_name', '{} Weather Filter'.format(parts[1])),
                    ('comment', '1: Probably OW, 0: Probably ICE')]
        elif variable_name.startswith('dal_'):
            return [('units', '1'),
                    ('long_name', 'Distance along the ice line')]
        else:
            return []

    def get_time_range(self):
        """
           Return first and last observation time in a swath file
        """
        with Dataset(self.nc_file_path, 'r') as nc_file:
            datestring_format = "%Y-%m-%dT%H:%M:%SZ"
            try:
                start_datetime = datetime.strptime(nc_file.start_date_and_time,datestring_format)
                stop_datetime  = datetime.strptime(nc_file.end_date_and_time,datestring_format)
            except KeyError:
                raise ValueError('Missing global attributes :start_date_and_time and/or :end_date_and_time from swath file %s' % self.nc_file_path)
            except ValueError:
                raise ValueError('Issue with datetime conversion of %s' % datestring_format)

        return start_datetime, stop_datetime

    def is_HighFreq_var(self, variable):
        """
           Decide if a variable (name) is a "high-frequency" variable
        """
        if ( 'tb90' in variable or 'tb85' in variable or variable.endswith('_h')
            or (np.array([a in variable.lower() for a in dtp.HF_ALGOS]).any() and 'tb' not in variable)
            or 'n90' in variable.lower() ):
            return True
        else:
            return False

    def get_LowFreq_sampling(self, varn, vals):
        """
           Ensure we get a low-frequency sampling of the values
        """
        if self.is_HighFreq_var(varn):
            var_s = vals.shape
            if len(var_s) != 2:
                raise ValueError("Unsupported format: a high-freq variable should be either (ni,4) or (nscanlines,nscanpos)")
            if var_s[-1] == 4:
                # This a 1D swath object:
                ret = vals[:,0]
            elif var_s[-1] == 2*self.nscanpos:
                # A 2D swath object:
                ret = vals[::2,::2]
            elif ( var_s[0] == var_s[1] ):
                # Gridded file so just return original values
                ret = vals
            elif ( var_s[1] == self.nscanpos ):
                # tb85 in tb37 res ?
                ret = vals
            else:
                raise ValueError("Unsupported format: neither 4, nor {} in the 2nd dimension (got {})".format(2*self.nscanpos,var_s[-1]))
        else:
            # if a low-freq variable, no slicing
            ret = vals

        return ret

    def get_HighFreq_sampling(self, varn, vals):
        """
           Ensures we get a high-frequency sampling of the values
        """
        if self.is_HighFreq_var(varn):
            # already a high-freq variable, nothing to do
            ret = vals
        else:
            var_s = vals.shape
            if len(var_s) == 1:
                # This a 1D swath object, duplicate the data and mask if needed
                if isinstance(vals, ma.MaskedArray):
                    ret = ma.column_stack((vals,)*4)
                else:
                    ret = np.column_stack((vals,)*4)
            elif len(var_s) == 2 and var_s[-1] == self.nscanpos:
                # A 2D swath object.
                ret = vals.repeat(2,axis=1).repeat(2,axis=0)
            elif ( var_s[0] == var_s[1] ):
                # Gridded file so just return original values
                ret = vals
            else:
                raise ValueError("Unsupported format: neither 1 nor 2 dimensions (got shape: {})".format(var_s,))

        return ret

    def read(self, variable_name, sampling='nominal'):
        """
           Re-implement the parent's read() for some specific cases like 'times'
              which are difficult to handle in a generic way
        """
        if variable_name == 'times':
            with Dataset(self.nc_file_path, 'r') as nc_file:
                try:
                    # get the reference time of the swath file
                    ncvar = nc_file.variables['time']
                    try:
                        # Use python datetime, not cftime, if option available
                        ref_time = num2date(ncvar[0], ncvar.units, only_use_cftime_datetimes=False)
                    except TypeError as e:
                        ref_time = num2date(ncvar[0], ncvar.units)
                    # access the delta_time of each observation
                    ncvar = nc_file.variables[self.get_file_variable('dtimes')]
                    ncdtime = ncvar[:]
                    # we cannot live with _FillValues in dtime (this happens for some OSISAF v1
                    #    reproc swath files (converted from RSS).
                    if (isinstance(ncdtime, ma.MaskedArray)):
                        if ncdtime.mask.sum() != 0:
                            LOG.warning("Found some _FillValue in variable %s. Set them to 0." % \
                                        (self.get_file_variable('dtimes'),))
                        ncdtime.data[ncdtime.mask] = 0
                        ncdtime = ncdtime.data
                    else:
                        if ncdtime.min() < 0:
                            LOG.warning("Found some negative values in variable %s. Set them to 0." % \
                                        (self.get_file_variable('dtimes'),))
                            ncdtime[ncdtime<0] = 0
                    # special case if dtimes has dimensions 'nscn', then we should re-format it to 'ni'
                    if 'nscn' in ncvar.dimensions:
                        # get the scanline number for each observation ('ni')
                        scanlines  = nc_file.variables[self.get_file_variable('scanline')][:]
                        # uniq the set of scanlines:
                        scanlines_u = np.unique(scanlines)
                        # get the dtime transferred to 'ni'
                        dtimes = np.ones_like(scanlines).astype('int16')
                        for ilu, slu in enumerate(scanlines_u):
                            dtimes[scanlines == slu] = ncdtime[ilu]
                        # finally, compute 'times' as sum ref_time + dtimes
                        times = np.array([ref_time + timedelta(seconds=int(dt)) for dt in dtimes])
                    elif ( self.scnl_l in ncvar.dimensions ):
                        scanline = nc_file.variables[self.get_file_variable('scanline')][:]
                        times = np.array([], dtype=datetime)
                        for scanline_time in ncdtime:
                            times = np.append(times, np.repeat(ref_time + \
                                timedelta(seconds=int(scanline_time)), scanline.shape[1]))
                        times = times.reshape(scanline.shape)
                    elif ( self.scnl_h in ncvar.dimensions ):
                        scanline = nc_file.variables[self.get_file_variable('scanline_h')][:]
                        times = np.array([], dtype=datetime)
                        for scanline_time in ncdtime:
                            times = np.append(times, np.repeat(ref_time + \
                                timedelta(seconds=int(scanline_time)), scanline.shape[1]))
                        times = times.reshape(scanline.shape)
                    else:
                        dtimes = ncdtime[:]
                        # finally, compute 'times' as sum ref_time + dtimes
                        times = np.array([ref_time + timedelta(seconds=int(dt)) for dt in dtimes])
                except KeyError as k:
                    raise ValueError("Cannot find variable/attribute %s in %s" % (k,self.nc_file_path,))
                except Exception as e:
                    raise ValueError("Error accessing 'times' in %s (%s)" % (self.nc_file_path,e,))
            ret = times

        elif ( variable_name.startswith('dtimes') ):
            """
                Return a time for each scanpoint in swath
            """
            # Used for reading dtime (_l/_h) variables with era as unit, letting us
            # skip combining the dtime with the reference time as above.
            with Dataset(self.nc_file_path, 'r') as nc_file:
                varname = self.get_file_variable(variable_name)
                dtimes = nc_file.variables[varname][:]
                dtimes_unit = nc_file.variables[varname].units
                if ( dtimes_unit == 'second' ):
                    # Try reading variable as 'times' (old method taking into account 1D files) ?
                    ret = self.read('times', sampling=sampling)
                else:
                    # Set masked values to 0
                    try:
                        dtimes[dtimes.mask] = 0
                        dtimes = dtimes.data
                    except AttributeError:
                        pass
                    try:
                        # Use python datetime, not cftime, if option available
                        dtimes = num2date(dtimes, dtimes_unit, only_use_cftime_datetimes=False)
                    except TypeError as e:
                        dtimes = num2date(dtimes, dtimes_unit)
                    if ( dtimes.ndim == 2 ):
                        ret = dtimes
                    else:
                        times = np.array([], dtype=datetime)
                        if ( variable_name == 'dtimes' ):
                            n_pos = len(nc_file.dimensions[self.scnp_l])
                        elif ( variable_name == 'dtimes_h' ):
                            n_pos = len(nc_file.dimensions[self.scnp_h])
                        for time in dtimes:
                            times = np.append(times, np.repeat(time, n_pos))
                        ret = times.reshape(len(dtimes), n_pos)

        elif ( variable_name.startswith('tb') and len(variable_name.split('_')) > 2 ):
            varname_parts = variable_name.split('_')
            # Add dtb to tb for variables of the type tb19v_OSISAF_ucorr, tb19v_corrNASA etc
            with Dataset(self.nc_file_path, 'r') as nc_file:
                # Read tbxxx
                try:
                    if ( varname_parts[1].startswith('lc')):
                        tb_variable = self.get_file_variable(varname_parts[0]+'_lc')
                    else:
                        tb_variable = self.get_file_variable(varname_parts[0])
                    nc_file.variables[tb_variable].set_auto_maskandscale(True)
                    tb = nc_file.variables[tb_variable][:]
                    try:
                        tb.mask
                    except AttributeError as e:
                        tb = ma.array(tb, mask=np.zeros(tb.shape).astype(np.bool))
                except KeyError as k:
                    raise ValueError("Cannot find variable %s in %s" % (k, self.nc_file_path))

                # Read dtbxxx
                try:
                    dtb_variable = self.get_file_variable(('d'+variable_name).replace('_lc', ''))
                    dtb_variable = dtb_variable.replace('@tb37', '')
                    dtb_variable = dtb_variable.replace('@tb90', '')
                    nc_file.variables[dtb_variable].set_auto_maskandscale(True)
                    dtb = nc_file.variables[dtb_variable][:]
                    try:
                        dtb.mask
                    except AttributeError as e:
                        dtb = ma.array(dtb, mask=np.zeros(dtb.shape).astype(np.bool))
                except KeyError as k:
                    raise ValueError("Cannot find variable %s in %s" % (k, self.nc_file_path))

                if ( tb.shape != dtb.shape ):
                    raise ValueError('{}: tb and dtb shapes differ {} {}'.format(tb_variable,tb.shape,dtb.shape))
            ##print '\tCombining:', tb_variable, dtb_variable
            ret = tb + dtb

        else:
            # use parent's read
            ret = super(OSIBaseNetCDFHandler, self).read(variable_name)

        if 'low' in sampling:
            ret = self.get_LowFreq_sampling(variable_name, ret)
        elif 'high' in sampling:
            ret = self.get_HighFreq_sampling(variable_name, ret)

        return ret



class SSMINetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to SSMI netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000 # In meters
    theta = 53.1 # Incidence angle
    altitude = 850 # Nominal height over Geoid (in km) [F08=856, F10=790, F11=853, F14=852]
    rtm_channels = ('19v', '19h', '22v', '37v', '37h', '90v', '90h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    #nasa_tp = {'nh': (185.04, 117.16, 208.72, 252.79, 238.20, 244.68, 223.64, 206.46, 190.14),
    #           'sh': (185.02, 118.00, 209.59, 259.92, 244.57, 254.39, 246.27, 221.95, 226.46)}
    # New TP
    nasa_tp = {'nh': (184.94, 116.82, 208.39, 254.37, 238.70, 247.84, 227.54, 210.33, 195.86),
        'sh': (184.64, 116.90, 208.93, 258.06, 239.99, 252.69, 245.42, 222.11, 225.34)}
    nscanpos = 64
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime_h'
        elif variable == 'tb22v':
            return 'tb22v'
        elif variable == 'tb22h':
            return 'tb22h'
        if variable == 'lons_h':
            return 'lon_h'
        elif variable == 'lats_h':
            return 'lat_h'
        elif ( variable.startswith('tb90') or variable.startswith('dtb90') ):
            if ( '@tb37' in variable ):
                return variable
            variable = variable.replace('tb90', 'tb85')
            if ( '@tb85' in variable ):
                variable = variable.replace('@tb85', '@tb90')
            return variable
        else:
            return variable


class SSMISNetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to SSMI/S netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000 # In meters
    theta = 53.1 # Incidence angle
    altitude = 850 # Nominal height over Geoid (in km)
    rtm_channels = ('19v', '19h', '22v', '37v', '37h', '90v', '90h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    #nasa_tp = {'nh': (185.04, 117.16, 208.72, 252.79, 238.20, 244.68, 223.64, 206.46, 190.14),
    #           'sh': (185.02, 118.00, 209.59, 259.92, 244.57, 254.39, 246.27, 221.95, 226.46)}
    # New TP
    nasa_tp = {'nh': (187.22, 117.40, 209.98, 258.21, 241.94, 252.46, 230.03, 208.97, 197.01),
        'sh': (186.62, 117.24, 210.16, 260.73, 241.09, 256.05, 247.87, 220.95, 226.31)}
    nscanpos = 90
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime_h'
        elif variable == 'tb22v':
            return 'tb22v'
        elif variable == 'tb22h':
            return 'tb22h'
        if variable == 'lons_h':
            return 'lon_h'
        elif variable == 'lats_h':
            return 'lat_h'
        elif ( variable.startswith('tb90') or variable.startswith('dtb90') ):
            if ( '@tb37' in variable ):
                return variable
            variable = variable.replace('tb90', 'tb85')
            if ( '@tb85' in variable ):
                variable = variable.replace('@tb85', '@tb90')
            return variable
        else:
            return variable


class AMSRENetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to AMSR netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000 # In meters
    theta = 55.0 # Incidence angle
    altitude = 705 # Nominal height over Geoid (in km)
    rtm_channels = ('06v', '06h', '19v', '19h', '22v', '37v', '37h', '90v', '90h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    #nasa_tp = {'nh': (183.72, 108.46, 209.81, 252.15, 237.54, 247.13, 226.26, 207.78, 196.91),
    #           'sh': (185.34, 110.83, 212.57, 258.58, 242.80, 253.84, 246.10, 217.65, 226.51)}
    # New TP
    nasa_tp = {'nh': (185.68, 111.12, 213.59, 253.57, 236.46, 250.50, 228.63, 208.83, 200.26),
        'sh': (185.65, 111.54, 213.72, 257.29, 237.75, 255.42, 245.72, 221.50, 228.27)}
    nscanpos = 243
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime_h'
        elif variable == 'tb22v':
            return 'tb22v_R19'
        elif variable == 'tb22h':
            return 'tb22h_R19'
        elif variable == 'tb37v':
            return 'tb37v_R19'
        elif variable == 'tb37h':
            return 'tb37h_R19'
        if variable == 'lons_h':
            return 'lon_h'
        elif variable == 'lats_h':
            return 'lat_h'
        elif ( variable.startswith('tb90') or variable.startswith('dtb90') ):
            if ( '@tb37' in variable ):
                return variable
            variable = variable.replace('tb90', 'tb85')
            return variable
        else:
            return variable


class AMSR2NetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to AMSR2 netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000 # In meters
    theta = 55.0 # Incidence angle
    altitude = 700 # Nominal height over Geoid (in km)
    rtm_channels = ('06v', '06h', '19v', '19h', '22v', '37v', '37h', '90v', '90h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    #nasa_tp = {'nh': (183.72, 108.46, 209.81, 252.15, 237.54, 247.13, 226.26, 207.78, 196.91),
    #           'sh': (185.34, 110.83, 212.57, 258.58, 242.80, 253.84, 246.10, 217.65, 226.51)}
    # New TP
    nasa_tp = {'nh': (190.67, 113.87, 215.62, 259.17, 241.28, 255.09, 233.00, 209.31, 200.11),
        'sh': (190.09, 114.12, 215.26, 261.99, 240.49, 259.17, 249.83, 219.89, 228.56)}
    nscanpos = 243
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime_h'
        elif variable == 'tb22':
            return 'tb22v'
        if variable == 'lons_h':
            return 'lon_h'
        elif variable == 'lats_h':
            return 'lat_h'
        elif ( variable.startswith('tb90') or variable.startswith('dtb90') ):
            if ( '@tb37' in variable ):
                return variable
            variable = variable.replace('tb90', 'tb85')
            if ( '@tb85' in variable ):
                variable = variable.replace('@tb85', '@tb90')
            return variable
        else:
            return variable


class SMMRNetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to SMMR netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000 # In meters
    theta = 50.2 # Incidence angle
    altitude = 947 # Nominal height over Geoid (in km)
    rtm_channels = ('19v', '19h', '22v', '37v', '37h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    nasa_tp = {'nh': (176.99, 111.45, 207.48, 252.15, 237.54, 247.13, 226.26, 207.78, 196.91),
               'sh': (175.39, 110.67, 207.57, 258.58, 242.80, 253.84, 246.10, 217.65, 226.51)}
    nscanpos = 47
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'tb22':
            return 'tb22v'
        else:
            return variable


class MWRINetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to MWRI netcdf files. Handles file io and instrument specific
        parameters.

    """
    def __init__(self, nc_file_path):
        self.nc_file_path = nc_file_path
        if ( 'fy3d' in os.path.basename(nc_file_path) ):
            self.nscanpos = 266
        else:
            self.nscanpos = 254
        if ( 'fy3a' in os.path.basename(nc_file_path) ):
            self.altitude = 834
        else:
            self.altitude = 836    

    nwp_fwhm_footprint = 56000 # In meters
    theta = 53.1 # Incidence angle
    rtm_channels = ('19v', '19h', '22v', '37v', '37h', '90v', '90h') #Which channels to run atm_corr for
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    nasa_tp = {'nh': (185.04, 117.16, 208.72, 252.79, 238.20, 244.68, 223.64, 206.46, 190.14),
               'sh': (185.02, 118.00, 209.59, 259.92, 244.57, 254.39, 246.27, 221.95, 226.46)}
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp'

    def get_file_variable(self, variable):
        if ( variable in ['lons', 'lon_l', 'lons_l'] ):
            return 'lon'
        elif ( variable in ['lats', 'lat_l', 'lats_l'] ):
            return 'lat'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime'
        elif variable == 'tb22v':
            return 'tb22v'
        elif variable == 'tb22h':
            return 'tb22h'
        elif ( variable in ['lon_h', 'lons_h'] ):
            return 'lon'
        elif ( variable in ['lat_h', 'lats_h'] ):
            return 'lat'
        elif ( variable.startswith('tb90') or variable.startswith('dtb90') ):
            variable = variable.replace('tb90', 'tb85')
            if ( '@tb37' in variable ):
                return variable.replace('@tb37', '')
            elif ( '@tb85' in variable ):
                return variable.replace('@tb85', '')
            else:
                return variable
        else:
            return variable


class MWINetCDFHandler(OSIBaseNetCDFHandler):
    """
        Interface to MWI netcdf files. Handles file io and instrument specific
        parameters.

    """
    nwp_fwhm_footprint = 56000
    theta = 53.1
    altitude = 835
    rtm_channels = ('19v', '19h', '22v', '37v', '37h', '90v', '90h')
    limits = {'tb19v' : [150, 295], 'tb19h' : [75, 295], 'tb37v' : [150, 295],
              'tb37h' : [100, 295], 'lons' : [-180, 180],
              'lats' : [-90, 90]}
    nasa_tp = {'nh': (190.67, 113.87, 215.62, 259.17, 241.28, 255.09, 233.00, 209.31, 200.11),
        'sh': (190.09, 114.12, 215.26, 261.99, 240.49, 259.17, 249.83, 219.89, 228.56)}
    nscanpos = 99
    scnl_l = 'n_scanl'
    scnl_h = 'n_scanl_h'
    scnp_l = 'n_scanp'
    scnp_h = 'n_scanp_h'

    def get_file_variable(self, variable):
        if variable == 'lons':
            return 'lon_l'
        elif variable == 'lats':
            return 'lat_l'
        elif variable == 'dtimes':
            return 'dtime'
        elif variable == 'dtimes_h':
            return 'dtime_h'
        elif variable == 'tb19v':
            return 'tb18v'
        elif variable == 'tb19h':
            return 'tb18h'
        elif variable == 'tb37v':
            return 'tb31v'
        elif variable == 'tb37h':
            return 'tb31h'
        elif variable in ('tb22', 'tb22v'):
            return 'tb23v'
        elif variable in ('tb90v', 'tb85v'):
            return 'tb89v'
        elif variable in ('tb90h', 'tb85h'):
            return 'tb89h'
        if variable == 'lons_h':
            return 'lon_h'
        elif variable == 'lats_h':
            return 'lat_h'
        elif ( variable.startswith('dtb90') ):
            return variable.replace('tb90', 'tb85')
        else:
            return variable


class SwathFactory(object):
    @staticmethod
    def get_swath(sat_id, file_path):
        if 'ssmi' in sat_id.lower():
            if ( int(sat_id[-2:]) <= 15 ):
                return SSMINetCDFHandler(file_path)
            else:
                return SSMISNetCDFHandler(file_path)
        elif 'amsre' in sat_id.lower() or 'amsr_aq' in sat_id.lower():
            return AMSRENetCDFHandler(file_path)
        elif 'amsr_gw1' in sat_id.lower():
            return AMSR2NetCDFHandler(file_path)
        elif 'smmr' in sat_id.lower():
            return SMMRNetCDFHandler(file_path)
        elif 'ssmis' in sat_id.lower():
            return SSMISNetCDFHandler(file_path)
        elif 'mwri' in sat_id.lower():
            return MWRINetCDFHandler(file_path)
        elif 'mwi' in sat_id.lower():
            return MWINetCDFHandler(file_path)
        else:
            raise TypeError('Unknown swath_type: %s' % sat_id)

    @staticmethod
    def get_swath_guess(file_path):
        """ Same as get_swath(), but will guess the sensor from the filename """
        fn = os.path.basename(file_path)
        if fn.startswith('ssmi_'):
            sat = int(fn[len('ssmi_f'):len('ssmi_f')+2])
            if sat <= 15:
                return SSMINetCDFHandler(file_path)
            else:
                return SSMISNetCDFHandler(file_path)
        elif fn.startswith('ssmis'):
            return SSMISNetCDFHandler(file_path)
        elif fn.startswith('amsr_aq'):
            return AMSRENetCDFHandler(file_path)
        elif fn.startswith('amsr_gw1'):
            return AMSR2NetCDFHandler(file_path)
        elif fn.startswith('smmr_n05'):
            return SMMRNetCDFHandler(file_path)
        elif fn.startswith('smmr_ni07'):
            return SMMRNetCDFHandler(file_path)
        elif fn.startswith('mwri'):
            return MWRINetCDFHandler(file_path)
        elif fn.startswith('mwi'):
            return MWINetCDFHandler(file_path)
        else:
            raise TypeError('Unable to guess sensor for file {}'.format(fn,))

if __name__ == '__main__':

    sw = SwathFactory.get_swath_guess('ssmi_f13_200802291033_s.nc')
    sw = SwathFactory.get_swath_guess('ssmi_f17_200802291033_s.nc')
    sw = SwathFactory.get_swath_guess('ssmis_f17_200802291033_s.nc')
    sw = SwathFactory.get_swath_guess('amsr_aq_200802291033_s.nc')
    sw = SwathFactory.get_swath_guess('smmr_n05_200802291033_s.nc')


