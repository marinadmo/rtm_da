"""
    Simple module for handling masks in the EASE projection. The main entrance
    into this module is the factory method:

    MaskFactory.create_mask(mask_type, hemisphere, timestamp)
"""

import os, sys
import logging
import numpy as np
sys.modules['snappy'] = None
from pyresample import geo_filter, utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import io_handler

LOG = logging.getLogger(__name__)
dirname = os.path.dirname(__file__)


class dynMask(geo_filter.GridFilter):
    """
        Class using information from LandOceanLakeMask file (defined in
        subclasses) to specify projection/grid information.

        :param hemisphere: Either 'nh' for Northern hemisphere or 'sh' for Southern hemisphere
        :type hemisphere: string
        :param timestamp: When the mask is valid
        :type timestamp: datetime
        :param cmask: Optional name of mask variable
        :type cmask: string
        :param vals: Optional list of mask values in cmask to be added to mask
        :type vals: list
    """

    def __init__(self, hemisphere, timestamp, cmask=None, vals=None, path=None, newmask=False, nasa95=False):

        if ( path != None ):
            self.file_path = self.get_file_path(hemisphere, timestamp, path)
        else:
            self.file_path = self.get_file_path(hemisphere)
        nc_file = io_handler.BaseNetCDFHandler(self.file_path)
        
        try:
            hemisphere = nc_file.readglobalattr('area')
        except io_handler.IONetCDFError as e:
            # Already given as argument so not sure why this is here?
            pass
        
        try:
            proj_string = nc_file.readvarattr('Lambert_Azimuthal_Grid', 'proj4_string')
        except io_handler.IONetCDFError as e:
            # Try reading crs variable instead
            proj_string = nc_file.readvarattr('crs', 'proj4_string')
        
        try:
            proj = nc_file.readglobalattr('projection').upper()
        except io_handler.IONetCDFError as e:
            # Try reading grid variable instead
            try:
                proj = nc_file.readglobalattr('grid').upper()
            except io_handler.IONetCDFError as e:
                # Just get a name from filename, only used locally anyway
                proj = os.path.basename(self.file_path).split('_')[3]

        if ( cmask != None and vals != None):
            self.mask = self.read_mask(nc_file, timestamp, cmask, vals)
        else:
            if ( newmask ):
                self.mask = self.read_mask(nc_file, timestamp, nasa95)
            else:
                self.mask = self.read_mask(nc_file, timestamp)

        xc, yc = nc_file.read('xc'), nc_file.read('yc')
        shape = len(xc), len(yc)
        #area_extent = (1000*yc[-1], 1000*xc[0], 1000*yc[0], 1000*xc[-1])
        # Since xc/yx are grid centers we need to subtract/add half a grid cell
        # to the LowerLeft/UpperRight corners.
        halfgrid_x = abs(xc[1] - xc[0]) / 2.0
        halfgrid_y = abs(yc[1] - yc[0]) / 2.0
        area_extent = (1000 * (yc[-1] - halfgrid_y), 1000 * (xc[0] - halfgrid_x), \
            1000 * (yc[0] + halfgrid_y), 1000 * (xc[-1] + halfgrid_x))
        del xc, yc

        proj_dict = {}
        for tmp in proj_string.split():
            key,value = tmp.split('=')
            proj_dict[key[1:]] = value

        if self.mask.shape != shape:
            raise ValueError('Unexpected mask shape: ' + str(self.mask.shape))

        area_def = utils.get_area_def('%s %s' %(proj, hemisphere),
                                         '%s %s' %(proj, hemisphere),
                                         '%s %s' %(proj, hemisphere),
                                         proj_dict, shape[0], shape[1],
                                         area_extent)

        super(dynMask, self).__init__(area_def, self.mask)

    def __str__(self):
        return self.file_path

    def get_file_path(self, hemisphere):
        raise NotImplementedError('get_file_path method not implemented')

    def read_mask(self, nc_file, timestamp):
        raise NotImplementedError('get_mask method not implemented')

    def get_shape(self):
        raise NotImplementedError('get_shape method not implemented')



class OSIDynTPMask_met(dynMask):
    """
    Defines a mask for dynamical tiepoints on the 25.0 km EASE projection.

    """

    def get_file_path(self, hemisphere):
        path = os.path.join(dirname, '../../par/LandOceanLakeMask_{}_ease2-250.nc'.format(hemisphere))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp):
        mask1 = nc_file.read('climatology')[timestamp.month - 1]
        mask1 = (mask1 == 10) # climatology = 10 => ow tp area
        mask2 = nc_file.read('coastmask_500')
        mask2 = (mask2 == 0) # coastmask_500 = 0 => Ocean
        # Combine sea mask and ow area.
        mask = np.logical_and(mask1, mask2)
        return mask


class OSIIceMask(dynMask):
    """
    Defines a sea-ice mask on the 25.0 km EASE projection.

    """

    def get_file_path(self, hemisphere):
        path = os.path.join(dirname, '../../par/LandOceanLakeMask_{}_ease2-250.nc'.format(hemisphere))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp):
        mask1 = nc_file.read('climatology')[timestamp.month - 1]
        mask1 = (mask1 == 3) # climatology = 3 => Ice
        mask2 = nc_file.read('coastmask_500')
        mask2 = (mask2 == 0) # coastmask_500 = 0 => Ocean
        # Combine sea mask and ice mask.
        mask = np.logical_and(mask1, mask2)
        return mask


class OSILandNearCoastMask(dynMask):
    """
    Defines a mask of area with land percentage > 0 within 200km of coast.
    """

    def get_file_path(self, hemisphere):
        path = os.path.join(dirname, '../../par/LandOceanLakeMask_{}_ease2-250.nc'.format(hemisphere))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp):
        mask = nc_file.read('nearcoast')
        mask = (mask == 1) # nearcoast = 1 => within 200km of coast.
        return mask


class OSICorrectionMask(dynMask):
    """
    Defines mask positions to be corrected.
    """

    def get_file_path(self, hemisphere):
        path = os.path.join(dirname, '../../par/LandOceanLakeMask_{}_ease2-250.nc'.format(hemisphere))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp, cmask, vals):
        mask = nc_file.read(cmask)
        newmask = np.zeros_like(mask)
        for val in vals:
            newmask = np.logical_or(newmask, mask == val)
        return newmask


class OSI_ice_new(dynMask):
    """
    New ice mask
    """

    def get_file_path(self, hemisphere, timestamp, path):
        path = os.path.join(path, 'tc_osisaf_{}_ease2-250_{}12.nc'.format(hemisphere, timestamp.strftime('%Y%m%d')))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp, nasa95):
        mask = nc_file.read('tp_area')
        mask = (mask == 3)
        if ( nasa95 ):
            nasa = nc_file.read('ct_NASA_wWF')[0]
            nasa_above_95 = (nasa > 95.0)
            mask = np.logical_and(mask, nasa_above_95)
        return mask


class OSI_ow_new(dynMask):
    """
    New ow mask
    """

    def get_file_path(self, hemisphere, timestamp, path):
        path = os.path.join(path, 'tc_osisaf_{}_ease2-250_{}12.nc'.format(hemisphere, timestamp.strftime('%Y%m%d')))
        LOG.debug("reading %s from path : %s" % (self.__class__.__name__, path))
        return path

    def read_mask(self, nc_file, timestamp, nasa95):
        mask = nc_file.read('tp_area')
        mask = (mask == 10)
        return mask


class MaskFactory(object):

    @staticmethod
    def create_mask(mask_type, hemisphere, timestamp, cmask=None, vals=None, path=None, nasa95=False):
        """
        Factory method. Returns a new mask object based on the input parameters.

        FIXME this method should not be wrappe in a class

        Valid mask types are:

        :dyn_tp_water: Mask for dynamical tiepoints
        :dyn_tp_water_met: Mask for dynamical tiepoints on the 25km EASE2 grid
        :land_near_coast: Mask for land area within 200km of coast on the 25km EASE2 grid
        :correction_mask: Mask defined by a combo of several mask values on the 25km EASE2 grid

        :param mask_type: The name of the mask
        :type mast_type: string
        :param hemisphere: Either 'nh' for Northern hemispehre or 'sh' for Southern hemisphere
        :type hemisphere: string
        :param timestamp: When the mask is valid
        :type timestamp: datetime
        :param cmask: Optional name of mask variable
        :type cmask: string
        :param vals: Optional list of mask values in cmask to be added to mask
        :type vals: list
        :param path: Path to files with tp_area
        :type path: string
        :param nasa95: Bool for combining mask with ct_NASA_wWF>95
        :type nasa95: bool
        """

        if mask_type.lower() == 'ice':
            mask = OSIIceMask(hemisphere, timestamp)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        elif mask_type.lower() == 'dyn_tp_water_met':
            mask = OSIDynTPMask_met(hemisphere, timestamp)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        elif mask_type.lower() == 'land_near_coast':
            mask = OSILandNearCoastMask(hemisphere, timestamp)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        elif mask_type.lower() == 'correction_mask':
            mask = OSICorrectionMask(hemisphere, timestamp, cmask=cmask, vals=vals)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        
        elif mask_type.lower() == 'ice_new':
            mask = OSI_ice_new(hemisphere, timestamp, path=path, newmask=True, nasa95=nasa95)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        elif mask_type.lower() == 'ow_new':
            mask = OSI_ow_new(hemisphere, timestamp, path=path, newmask=True)
            LOG.info("using %s from %s" % (mask.__class__.__name__, mask))
            return mask
        
        else:
            raise ValueError('Unknown mask type: %s' % mask_type)

