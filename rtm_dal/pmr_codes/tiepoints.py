import os
import numpy as np
from copy import copy, deepcopy


from scipy.interpolate import splprep

#from . import common
import common

class Tiepoint(object):

    def __init__(self, what, source='rrdp', instr='AMSR2', area='NH', channels='all', **kwargs):

        # check source
        if 'rrdp' in source:
            if source == 'rrdp':
                source = 'rrdp3'
            else:
                if source not in ('rrdp2', 'rrdp3'):
                    raise ValueError("RRDP source should be 'rrdp3' or 'rrdp2'")

        # 1. checks and preparation
        self._store_info(source, instr, area, what)

        # 2. load (from files)
        self._load(channels, **kwargs)

        # finalize the construction
        self._finalize_construct(channels)

    def _store_info(self, source, instr, area, what):

        self.source = source
        self.instr = instr
        self.area  = area.upper()

        if what not in ('cice','ow'):
            raise ValueError('what should be "cice" or "ow" (got {})'.format(what))

        self.what = what

    def transfer_info(self,obj):
        obj.source   = self.source
        obj.instr    = self.instr
        obj.area     = self.area
        obj.what     = self.what
        obj.channels = tuple(self.channels[:])

    def _finalize_construct(self,channels):
        # extract channel names
        self.channels = tuple(sorted([t for t in dir(self) if t.startswith('tb')]))
        if channels != 'all':
            if set(self.channels) != set(channels):
                raise ValueError('loaded {} tb channels, when asked to restrict to {}'.format(len(self.channels),channels))

        # stack the Tbs (in self.all_tbs)
        self._stack_tbs()

    def to_dict(self,):
        return common.to_dict(self,)

    def _extract_tp_and_C(self):
        # mean tie-point
        self.tp = self.all_tbs.mean(axis=1)

        # covariance around mean tie-point
        self.C  = np.cov(self.all_tbs)

    def _stack_tbs(self,):
        # stack the tbs into a single array
        self.all_tbs = np.column_stack(([getattr(self,ch) for ch in self.channels],))

    def _load(self, channels, **kwargs):

        if 'rrdp' in self.source:

            # we need to import the rrdp/ module
            from sirrdp import rrdp_file as rrdp

            # read from RRDP DTUSIC1 and DMISC0 files
            instr = self.instr
            if self.instr == 'AMSRE' or self.instr == 'AMSR':
                instr='AMSR'
                rrdp_years  = range(2007,2012)
            elif self.instr == 'AMSR2' or self.instr == 'CIMR':
                if self.source == 'rrdp2':
                    rrdp_years  = range(2012,2017)
                elif self.source == 'rrdp3':
                    rrdp_years = range(2016,2020)
                else:
                    raise ValueError("Do not know about RRDP version {}".format(self.source))

            if self.what == 'ow':
                rrdp_years = [rrdp_years[0],]

            if self.area.lower() == 'nh':
                rrdp_months = [10, 11, 12, 1, 2, 3, 4]
            elif self.area.lower() == 'sh':
                rrdp_months = [5, 6, 7, 8, 9, 10, 11]
            else:
                raise ValueError('Invalid area {}'.format(self.area,))

            if self.what == 'ow':
                rrdp_months = range(1,13)

            try:
                max_n_lines = kwargs['max_n_lines']
            except KeyError:
                max_n_lines = None

            try:
                rrdpdir = kwargs['srcdir']
            except KeyError as k:
                rrdpdir = os.path.join(os.path.dirname(rrdp.__file__),{'rrdp2':'RRDP_v2.0','rrdp3':'RRDP_v3.0'}[self.source])

            # read from RRDP files
            tbs = rrdp.load_rrdp_samples(rrdpdir,instr,self.area,self.what,
                                     rrdp_years,rrdp_months,channels=channels,
                                     max_sel=10000,with_nwp=True,max_n_lines=max_n_lines)
        else:
            # expect tbs=dict() ready from the call line.
            try:
                tbs = kwargs['tbs']
            except KeyError:
                raise ValueError("self.source is {}, thus kwargs should contain tbs=dict()".format(self.source,))

        # transfer from dict() to attributes of the object
        if channels == 'all':
            channels = [t for t in tbs.keys() if t.startswith('tb')]
        for k in channels:
            setattr(self, k, tbs[k])

    def extract_channels(self,channels):
        # first, check the channels we request are in the original object
        for ch in channels:
            if ch not in self.channels:
                raise ValueError("Channel {} not in list of available channels: {}".format(ch,self.channels))
        # then get a copy of the object
        ret = deepcopy(self)
        # then delete all channels that are not to be kept
        for ch in self.channels:
            if ch not in channels:
                delattr(ret,ch)
        # finalize the construction
        ret._finalize_construct(channels)
        return ret

    def get_samples(self, N="all"):

        ch1 = self.channels[0]
        length_ch1 = len(getattr(self,ch1))

        if N == "all":
            inds = np.arange(length_ch1)
        else:
            # get the indices at random
            inds = np.random.randint(0,length_ch1-1,size=N)

        # extract the Tbs into a dict
        samples = dict()
        for ch in self.channels:
            samples[ch] = getattr(self, ch)[inds]

        return samples

class OWTiepoint(Tiepoint):

    def __init__(self, **kwargs):
        # first the Tiepoint __init_ is called
        super().__init__('ow', **kwargs)
        self._finalize_ow_construct()

    def _finalize_ow_construct(self):
        # next we complement the object with OW pecific attributes
        self._extract_tiepoints()
        # we are done with all_tbs
        del self.all_tbs

    def _extract_tiepoints(self):
        # extract OW tie-point statistics (mean, covar,...)
        super()._extract_tp_and_C()

    def extract_channels(self,channels):
        # return a copy of the object, but keep only some of the Tbs
        ret = super().extract_channels(channels)
        ret._finalize_ow_construct()
        return ret

    def get_ice_conc(self):
        # return 0% SIC
        return np.zeros_like(getattr(self,self.channels[0]))

    def strip(self, copy_first=True):
        keep = ('tp',)
        return common.strip(self, keep, copy_first)

    @classmethod
    def from_dict(cls, dct):
        obj = type('@'+cls.__name__, (object,), dct)()
        return obj

class CICETiepoint(Tiepoint):
    # an CICE Tiepoint object is a Tiepoint object
    #   that knows it is for CICE, has more methods and attributes
    def __init__(self, **kwargs):
        # first the Tiepoint __init_ is called
        super().__init__('cice', **kwargs)
        self._finalize_cice_construct()

    def _finalize_cice_construct(self):
        # next we complement the object with CICE specific attributes
        self._extract_tiepoints()
        # we are done with all_tbs
        del self.all_tbs

    def _pca(self):
        # compute eigen-elements from a covariance matrix
        eigenvals, eigenvectors = np.linalg.eig( self.C )

        # rank the eigenvalue(s)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvectors = eigenvectors[:,idx]

        # return eigen vectors, values, and mean point
        return eigenvectors, eigenvals

    def _extract_tiepoints(self):
        # extract basic CICE tie-point statistics (mean, covar,...)
        super()._extract_tp_and_C()
        # more to be done only if n > 1
        if len(self.channels) > 1:
            # extract additional CICE tie-points statistics (u, matrices,...)
            self.evecs, self.evals = self._pca()

            # check (correct if needed) direction of vector u
            #   (should be from MYI -> FYI)
            if np.dot(self.evecs[:,0],np.ones(len(self.channels))) < 0:
                self.evecs *= -1.

            # ice line direction u
            self.u = self.evecs[:,0]

            # the dal
            self.dals = np.dot(self.u,self.all_tbs)

            # dal of the mean cice point
            dal_cice = np.dot(self.tp, self.u)

            # two extreme points along the ice line
            #   (in-place for MYI and FYI tie-points)
            l,L = np.percentile(self.dals,(5.,95.))
            self.myi_tp = self.tp + (l - dal_cice) * self.u
            self.fyi_tp = self.tp + (L - dal_cice) * self.u
        else:
            self.evecs  = None
            self.evals  = None
            self.u      = None
            self.dals   = None
            self.myi_tp = None
            self.fyi_tp = None

    def extract_channels(self,channels):
        # return a copy of the object, but keep only some of the Tbs
        ret = super().extract_channels(channels)
        ret._finalize_cice_construct()
        return ret

    def get_ice_conc(self):
        # return 100% SIC
        return np.ones_like(getattr(self,self.channels[0]))

    def get_projection_matrices(self):
        # projection matrices derived from the eigen elements
        if len(self.channels) > 1:
            M = np.array(self.evecs[:,1:])
            P = M @ M.T
            return (M, P)
        else:
            raise ValueError("no projection matrices are defined for n<2")

    def strip(self, copy_first=True):
        keep = ('tp','u','fyi_tp','myi_tp')
        return common.strip(self, keep, copy_first)

    @classmethod
    def from_dict(cls, dct):
        obj = type('@'+cls.__name__, (object,), dct)()
        return obj

class CICETiepointDalSegment(CICETiepoint):

    def __init__(self, ci, dal_range=None, min_count = 50,):

        # top-level sanity checks
        if len(ci.channels) == 1:
            raise ValueError("DAL is not defined for n-channel = 1")

        # default is to take all the samples
        if dal_range is None:
            dal_range = (ci.dals.min()-1.e-3, ci.dals.max()+1.e-3)

        # selection mask
        dal_indx = (ci.dals>=dal_range[0])*(ci.dals<dal_range[1])

        if dal_indx.sum() < min_count:
            raise IndexError("Not enough samples ({}) in the requested segment. Change dal_range= or min_count=".format(dal_indx.sum()))

        # the selected tb samples
        all_tbs = np.column_stack(([getattr(ci,ch)[dal_indx] for ch in ci.channels],))
        mean_tb = all_tbs.mean(axis=1)

        # projection matrix
        _,Pci = ci.get_projection_matrices()

        # project the samples
        projs = np.asarray( Pci @ (all_tbs - mean_tb[:,np.newaxis]) + mean_tb[:,np.newaxis])

        # work on covariance C and eigen elements
        C  = np.cov(projs)
        # compute eigen-elements
        eigenvals, eigenvectors = np.linalg.eig( C )

        # we expect the lowest variance to be almost 0 (since
        #    samples are collapsed on a subspace)
        idx_low = np.argmin( eigenvals )
        if abs(eigenvals[idx_low]) > 1.e-3:
            raise ValueError("The smallest eigenval should be 0 after collapsing (got {})".format(eigenvals[idx_low]))
        
        # copy the collapsed eigen elements from original ci
        eigenvals[idx_low] = ci.evals[0]
        eigenvectors[:,idx_low] = ci.evecs[:,0]
        
        # rank the eigenvalue(s)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvectors = eigenvectors[:,idx]
        
        # now finalize/populate the object
        self.tp    = mean_tb
        self.C     = C
        self.evals = eigenvals
        self.evecs = eigenvectors
        
        # check (correct if needed) direction of vector u
        #   (should be from MYI -> FYI)
        if np.dot(self.evecs[:,0],np.ones(len(ci.channels))) < 0:
            self.evecs *= -1.

        # ice line direction u
        # NB! the ice line (vector u) is the same for all segments
        self.u = self.evecs[:,0]

        # the dals (note we use the collapsed tbs, thus should be repeating a single value)
        self.dals = np.dot(self.u,projs)
        
        # we do not define myi and fyi tie-points (we just collapsed this dimension)

        # store/transfer info from original ci
        ci.transfer_info(self)

        # transfer tbs from dict() to attributes of the object
        for ich, ch in enumerate(self.channels,):
            setattr(self, ch, projs[ich,:])

    # due to the way extract_channels is built for CICETiePoint objects, it is not possible
    #    for a CICETiepointDalSegment to do the same. Thus, the CICETiePoint object provided as
    #    input to __init__ must already have the right channels extracted.
    def extract_channels(self,channels):
        raise NotImplementedError("A {} object cannot extract channels.".format(self.__class__.__name__))

class CICETiepointDalSegments(object):

    def __init__(self, ci, nseg=53, percentiles=(1.,99.), min_count=15,):

        # segmented tiepoints require DAL, and this requires at least 2 channels to work.
        if len(ci.channels)  < 2:
            raise ValuerError("Cannot have segmented CICETiepoints with n<2")

        # segment the Tbs by DAL, and return nseg tiepoint objects
        l_dal, h_dal = np.percentile(ci.dals, percentiles)
        dal_bins = np.linspace(-1.3, 1.3, num=nseg) * (h_dal - l_dal) + l_dal
        dal_centers = (dal_bins + 0.5*(dal_bins[1]-dal_bins[0]))[:-1]

        # bin the Tb samples by dal
        self.segments = [None,]*(len(dal_bins)-1)
        for dal_b in range(len(dal_bins)-1):
            m_dal = dal_bins[dal_b]
            M_dal = dal_bins[dal_b+1]
            # store DalSegment objects
            try:
                self.segments[dal_b] = CICETiepointDalSegment(ci, dal_range=(m_dal,M_dal), min_count=min_count,)
            except IndexError:
                pass # None is stored

        ci.transfer_info(self)
        self.dal_percentiles = tuple(percentiles)
        self.dal_limits      = (l_dal, h_dal)
        self.dal_centers     = np.asarray(dal_centers)

        # transfer some information that is common to all segments to the top-level object
        for dal_b in range(len(dal_bins)-1):
            if self.segments[dal_b] is not None:
                self.u = self.segments[dal_b].u
                break

        # prepare for spline interpolation of the CI tie-point:
        self._set_tps()
        # TODO: handle the case when some of the segments are invalid (they cannot enter the spline)
        self.tp_tck = list(splprep(self.tps, u=self.dal_centers, k=3))[0]

    # len() and count(): number of segments, and number of active segments
    def __len__(self):
        return len(self.segments)

    def count(self):
        return len(self.segments) - self.segments.count(None)


    # helper routine to loop over the segments
    def _iterate_on_segments(self, fn, *args, **kwargs):
        for s in self.segments:
            if s is not None:
                getattr(s,fn)(*args, **kwargs)

    def extract_channels(self, channels):
        raise NotImplementedError("A {} object cannot extract channels.".format(self.__class__.__name__))

    def _set_tps(self):
        self.tps = np.ma.empty((len(self.channels),len(self)))
        for si, s in enumerate(self.segments):
            if s is None:
                self.tps[:,si] = np.ma.masked
            else:
                self.tps[:,si] = s.tp


