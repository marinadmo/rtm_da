
import numpy as np
import numpy.random
from copy import deepcopy, copy

from scipy.optimize import minimize
from scipy.interpolate import splev, splprep, splrep, interp1d

from matplotlib import pylab as plt

#from . import common
#from . import image
#from . import tiepoints as tp
#from . import pmr as pm
import common
import image
import tiepoints as tp
import pmr as pm

def uniq_keeporder(seq):
    # order preserving
    #    from https://www.peterbe.com/plog/uniqifiers-benchmark
    noDupes = []
    [noDupes.append(i) for i in seq if not noDupes.count(i)]
    return noDupes

def gen_rand_unit_vector(n):
    """ make a n-dim random unit vector """
    v = np.random.normal(size=(n,))
    v /= np.linalg.norm(v)
    return v

def check_init_params(channels, ow, cice):

    # check types
    #if not isinstance(ow, tp.OWTiepoint):
    #    raise TypeError("Wrong type for the 'ow' parameter (got {}, should be a OWTiepoint object)".format(type(ow)))

    #if not isinstance(cice, CICETiepoint) or not isinstance(cice, CICETiepointDalSegment):
    #    raise TypeError("Wrong type for the 'cice' parameter (got {}, should be a CICETiepoint or a CICETiepointDalSegment object)".format(type(cice)))

    # channels must be a list, not a string
    if isinstance(channels, (str,bytes)):
        channels = (channels,)

    # check the list of channels is unique
    if len(channels) != len(set(channels)):
        raise ValueError("The list of channels contains duplicates!")

    # a common mistake is to drop the "tb" in the name of channels
    channels = ['tb'+ch if not ch.startswith('tb') else ch for ch in channels]

    # check dimensions, sort the channels
    channels = tuple(sorted(channels))
    n        = len(channels)
    if n == 0:
        raise ValueError("Algorithms must have at least 1 channel")

    # check the ow and cice signatures hold the needed channels
    for ch in channels:
        if ch not in ow.channels or ch not in cice.channels:
            raise ValueError("The OW or CICE signature object does not have the requested channels ({} not found)".format(ch,))

    return channels, n


class SICAlgoResult(object):

    def __init__(self, sic, sdev, dal, owf):
        """ just a container class to hold results from SIC algorithm evaluation """
        self.sic  = sic
        self.sdev = sdev
        self.dal  = dal
        self.owf  = owf

    def get(self, what):
        """ extract single or multiple fields """
        if isinstance(what, str):
            what = (what,)
        ret = list([getattr(self, w) for w in what])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def __iter__(self):
        """ make the object iterable, so that we can directly access return fields as a tuple. """
        for k in ('sic', 'sdev', 'dal', 'owf'):
            yield getattr(self,k)

class SICAlgo(object):

    keep = ('channels','v','ow','cice','tuning','n', 'ow_sdev', 'cice_sdev', 'owf_threshold', 'owf_lw', 'owf_hw')
    keep_methods = ('compute_sic','_preprocess_tbs','_postprocess_sic','_compute_sic','_check_v','_compute_dal','_compute_owf',)

    def __init__(self, channels, ow, cice, pmr=None, tuning=None, owf_threshold=0.1 ):

        # various checks and pre-processing
        self.channels, self.n = check_init_params(channels, ow, cice)
        channels = self.channels

        # extract the channels we are interested in from the ow and cice signature
        try:
            if ow.channels != self.channels:
                self.ow   = ow.extract_channels(self.channels)
            else:
                self.ow   = ow
            if cice.channels != self.channels:
                self.cice = cice.extract_channels(self.channels)
            else:
                self.cice   = cice
        except Exception as ex:
            raise ValueError("Unable to extract required channels from the OW or CICE signature ({})".format(ex,))

        # if a pmr object is given, store its nedt matrix
        if pmr is not None:
            self.nedt_C = pmr.get_nedt_C(self.channels)
        else:
            self.nedt_C = np.zeros((self.n,self.n))

        # compute vector v if one- or two-channels algorithms
        self.tuned    = False
        self.v        = None
        if self.n == 1:
            # 1d algorithms cannot be tuned
            self.v = np.array([1.,])
        elif self.n == 2:
            # 2d algorithms cannot be tuned
            self.v = np.array([-self.cice.u[1],self.cice.u[0]])
        else:
            # prepare the projection matrices
            self._update_matrices()
            # algorithms with more dimensions can be tuned now,
            #   or later (by calling tune)
            if tuning is not None:
                self.tune(tuning)

        # if n<3 we assign the uncertainty nodes at 0% and 100% SIC
        #   if n>=3, this is done within "tune()"
        if self.n < 3:
            self.ow_sdev, self.cice_sdev = self._compute_algo_uncert_tiepoints(ret_stddev_percent=True)

        # just to be extra sure.
        self._check_v()

        # prepare the OWF (sets several attributes)
        if self.n > 1:
            self.prepare_owf(owf_threshold, )


    def strip(self, copy_first=True):
        return common.strip(self,self.keep,copy_first=copy_first)

    def to_dict(self,):
        return common.to_dict(self)

    @classmethod
    def from_dict(cls, dct):
        # create the two tiepoints objects from their dicts
        ow   = tp.OWTiepoint.from_dict(dct['ow'])
        cice = tp.CICETiepoint.from_dict(dct['cice'])

        # create SICAlgo object (no tuning)
        # replace the dict by objects
        dct['ow'] = ow
        dct['cice'] = cice
        obj = type('@'+cls.__name__, (object,), dct)()

        # add the few needed methods
        #      the __get__(obj) is to bound the method to the instance
        #      (and thus having 'self' to be passed as 1st parameter)
        for meth in cls.keep_methods:
            setattr(obj, meth, getattr(cls,meth).__get__(obj))

        return obj

    def prepare_owf(self, owf_threshold, ):
        # store OWF configuration
        self.owf_threshold = owf_threshold

        # compute the classic dal of all OW samples, and of the mean OW tie-point
        ow_dals = self._compute_dal(self.ow.get_samples(),dal_type='classic')
        ow_avg_dal = np.dot(self.cice.u, self.ow.tp)

        # place two points (low and high weather) along a line parallel to the ice line but at SIC=0%
        l,L = np.percentile(ow_dals,(5.,95.))
        self.owf_lw = self.ow.tp + (l - ow_avg_dal) * self.cice.u
        self.owf_hw = self.ow.tp + (L - ow_avg_dal) * self.cice.u

    def tune(self, tuning='random'):

        if self.n == 1 or self.n == 2:
            # nothing to do, already tuned in v
            pass
        else:

            # analyse the value of the 'tuning' keyword:
            #   <tuning>[-<theo|stat>]
            # 'theo' : default, tuning uses uncertainty propagation formula
            # 'stat' : uses statistics (requires applying the SIC algorithm)
            try:
                tuning, tuning_on = tuning.split('-')
            except ValueError:
                # not enough values to unpack, only tuning was given
                tuning_on = 'theo'
                pass

            # do tuning
            if tuning == 'random':
                # v is along a random direction perpendicular to u
                z = gen_rand_unit_vector(self.n - 1)
                # then put z into n-dim, in the space orthogonal to u
                self.v = self._get_v_from_z(z)
            elif tuning == 'ow' or tuning == 'cice':
                # optimize for best accuracy at either end.
                self._tune_accuracy_tiepoints(tuning, tuning_on=tuning_on)
            else:
                raise NotImplementedError("Tuning '{}' is not implemented".format(tuning))

            self.tuned    = tuning
            self.tuned_on = tuning_on

            # store the uncertainty at 0% and 100% SIC
            _uncert_func = {'theo':self._compute_algo_uncert_tiepoints,
                            'stat':self._compute_algo_uncert_tiepoints_stat}[tuning_on]
            self.ow_sdev, self.cice_sdev = _uncert_func(ret_stddev_percent=True)

        # just to be extra sure.
        self._check_v()

    def _tune_accuracy_tiepoints(self, tuning, tuning_on='theo'):

        if self.n == 1 or self.n == 2:
            return

        _indx = {'ow':0,'cice':1}[tuning]
        _uncert_func = {'theo':self._compute_algo_uncert_tiepoints,'stat':self._compute_algo_uncert_tiepoints_stat}[tuning_on]
        if self.n == 3:
            # for n=3 we do a brute force optimization (evaluate all the 1d axis)
            x_eval = np.linspace(-1.,1.,num=51)
            y_eval = np.zeros_like(x_eval)
            for ie, xe in enumerate(x_eval,):
                # generate a v vector from xe:
                self.v = self._get_v_from_z(np.array([xe,]))
                y_eval[ie] = _uncert_func(indx=_indx)

            #print(tuning, 100.*np.sqrt(y_eval))
            #print(tuning, 100.*sqrt(y_eval.min()))
            # record v allowing best accuracy (minimum variance)
            self.v = self._get_v_from_z(np.array([x_eval[y_eval.argmin()],]))
        else:
            # for n>3 we rely on a descent algorithm in n-2 dimensions.

            def _compute_algo_uncert_from_z(z, extras,):
                # routine to be optimized
                alg  = extras['alg']
                indx = extras['indx']
                alg.v = alg._get_v_from_z(z)
                sdev = _uncert_func(indx=indx,ret_stddev_percent=True)
                return sdev

            def conf(z):
                # constrain in the n-2 space: |z|<1
                return 1. - np.linalg.norm(z)

            extras = dict()
            extras['alg'] = self
            extras['indx'] = _indx
            con = {'type':'ineq', 'fun': conf}

            failed_iter = 0
            good_iter = 0

            max_good = 3
            max_failed = 3
            while good_iter < max_good:

                z_fg = np.array([0.,]*(self.n-2))
                if failed_iter > 0:
                    z_fg  = gen_rand_unit_vector(self.n - 2)
                    z_fg *= np.random.uniform()

                res = minimize( _compute_algo_uncert_from_z, z_fg, args=(extras,), constraints=con )

                if not res.success:
                    print("FAILED minimization with :", res)
                    # rerun in case of failure
                    failed_iter += 1
                    if failed_iter >= max_failed:
                        break
                    continue
                else:
                    #print(good_iter, x, res.fun)
                    #print("SUCCESS:", res)
                    good_iter += 1

            zopt = res.x
            vopt = self._get_v_from_z(zopt)
            self.v = vopt

    def get_channels_str(self,no_tb=False, delim='+', bandnames=False, no_pol=False):
        
        # format in strings
        if bandnames:
            ret = [ pm.tb_dict[ch[:-1]]+'('+ch[-1]+')' for ch in self.channels ]
        else:
            if no_tb:
                ret = [ ch[2:] for ch in self.channels ]
            else:     
                ret = self.channels[:]
        
        # remove polarization information
        if no_pol:
            polstrs = ('(v)','(h)','v','h')
            for polstr in polstrs:
                ret = [ ch.replace(polstr,'') for ch in ret ]
            # compact (remove duplicate frequencies)
            ret = uniq_keeporder(ret)

        # join into a string
        ret = delim.join(ret)

        return ret

    def __str__(self):
        tune_str = 'no'
        if self.tuned:
            tune_str = self.tuned
        return "{} {} tuning".format('+'.join(self.channels),tune_str)

    def _update_matrices(self):
        self.M, self.P = self.cice.get_projection_matrices()

    def _preproc_z(self,z):
        
        #if np.isnan(z).any():
        #    raise ValueError("vector z has nans. Abort.")

        if len(z) != self.n-1 and len(z) != self.n-2:
            raise ValueError("Dimension problem with z={}. It should have length n-1 or n-2 (n={})".format(z,self.n))

        if len(z) == self.n-2:
            z1 = np.sqrt( 1. - np.dot(z,z) )
            z = np.append([z1,],z)

        return z

    def _get_v_from_z(self,z):

        z = self._preproc_z(z)
        v = np.squeeze(np.asarray(z @ self.M.T))

        return v

    def _get_z_from_v(self, v):
        z = np.squeeze(np.asarray(v @ self.M))
        return z

    def _check_v(self):
        if self.v is not None and len(self.v) > 1:
            if len(self.v) != self.n:
                raise ValueError("WARNING! v is not of correct length !")
            #if np.isnan(self.v).any():
            #    raise ValueError("ERROR! v has NaNs !")
            if abs(np.linalg.norm(self.v)) - 1. > 1.e-3:
                raise ValueError("WARNING! v is not a unit vector !")
            if abs(np.dot(self.v, self.cice.u)) > 1.e-3:
                raise ValueError("WARNING! v is not perpendicular to u !")

    def compute_sic(self, tbs, **kwargs):

        # pre-process original tbs to a dict with compressed/masked tbs
        __tbs, _mask, _shape, ffield = self._preprocess_tbs(tbs)

        # compute sic
        rets = self._compute_sic(__tbs, **kwargs)

        #post-process
        return self._postprocess_sic(rets, _mask, _shape, ffield,)

    def _preprocess_tbs(self, tbs):

        # access tbs as a dict
        if hasattr(tbs,'keys'):
            tbs_as_dict = tbs
        elif isinstance(tbs,(tp.OWTiepoint,tp.CICETiepoint,)):
            tbs_as_dict = tbs.get_samples()
        else:
            raise ValueError("The Tb data is not stored in a dict() nor a Tiepoint object, don't know what to do with {}".format(type(tbs)))

        ffield = tbs_as_dict[self.channels[0]]

        # pre-process: the input Tb data
        _shape = ffield.shape
        _tbs = dict()
        if isinstance(ffield,(image.Image, image.BaseImage,)):
            for k in self.channels:
                _tbs[k] = deepcopy(tbs_as_dict[k].img).reshape(-1)
        elif isinstance(ffield,numpy.ndarray):
            for k in self.channels:
                _tbs[k] = deepcopy(tbs_as_dict[k]).reshape(-1)
        else:
            raise NotImplementedError('do not know how to pre-process Tb data')

        # generate and apply a common mask
        _mask  = np.zeros(_shape).astype('bool').reshape(-1)
        for k in self.channels:
            if hasattr(_tbs[k],'_mask'):
                _mask = np.logical_or(_mask,_tbs[k]._mask)
        __tbs = dict()
        for k in self.channels:
            if hasattr(_tbs[k],'_mask'):
                __tbs[k] = _tbs[k]._data[~_mask]
            else:
                __tbs[k] = _tbs[k]

        return __tbs, _mask, _shape, ffield

    def _postprocess_sic(self, rets, _mask, _shape, ffield, ):

        def _place_rets_in_cont(rets, what, _shape, _mask, ffield):
            _res = getattr(rets,what)
            if _res is None:
                _dtype = type(1.0)
            else:
                _dtype = _res.dtype
            _arr = np.ma.masked_array(np.zeros(_shape),mask=_mask,dtype=_dtype).reshape(-1)
            _arr[~_mask] = _res
            _arr = _arr.reshape(_shape)
            if isinstance(ffield,(image.Image, image.BaseImage,)):
                _arr = image.Image(_arr,ffield.step)
            return(_arr)

        # place result(s) of _compute_sic() in new containers
        _sic  = _place_rets_in_cont(rets, 'sic', _shape, _mask, ffield,)
        _dal  = _place_rets_in_cont(rets, 'dal', _shape, _mask, ffield,)
        _sdev = _place_rets_in_cont(rets, 'sdev', _shape, _mask, ffield,)
        _owf  = _place_rets_in_cont(rets, 'owf', _shape, _mask, ffield,)

        # return
        rets = SICAlgoResult(_sic, _sdev, _dal, _owf)
        return rets

    def _compute_sic(self, tbs, dal_type='classic'):

        if self.v is None:
            raise ValueError("This algorithm has not been tuned yet!")

        # check all the required channels are given as input
        for ch in self.channels:
            if ch not in tbs.keys():
                raise ValueError("Tb channel {} is missing as input".format(ch))

        # stack the Tbs
        Tbs   = np.column_stack(([tbs[ch] for ch in self.channels],))

        # dimensions (n = nbchannels, N=nbTbs)
        n,N = Tbs.shape
        if n != self.n:
            raise ValueError("Dimensionality problem! Got {} Tb channels, but the algo is {}-dim".format(n,self.n))

        # prepare the Water and Ice signatures
        W = self.ow.tp[:,np.newaxis]
        W = W.repeat(N,axis=1)
        #W = np.squeeze(W)

        I_needs_repeat = True
        if (I_needs_repeat):
            I = self.cice.tp[:,np.newaxis]
            I = I.repeat(N,axis=1)
            #I = np.squeeze(I)

        # prepare unit vector v
        v = self.v

        # just to be extra sure.
        self._check_v()

        if self.n != len(v):
            raise ValueError("Dimensionality problem! Got {} Tb channels, but v has length {}".format(n,len(v)))

        # expand v by repeat to match shape of Tbs
        V = v.reshape(n,1)
        V = V.repeat(N,axis=1)

        # compute SIC with the dot product (using einsum notation)
        numer = (Tbs - W)
        denom = (I - W)

        Cn=np.einsum("ji,ji->i", V, numer)
        Cd=np.einsum("ji,ji->i", V, denom)

        C  = Cn / Cd

        # Compute algorithm uncertainties, if possible.
        if hasattr(self,'ow_sdev') and hasattr(self,'cice_sdev'):
            # the algorithm is already tuned. So we know the ow_sdev and cice_sdev and can compute sdev
            # 'Banana-shaped' uncertainty blending, with mirroring at 0 and 1.
            concc = np.clip(C, -0.99, 1.99) # make sure iceconc doesn't reach -1 or 2
            sqrt_tmp = np.sqrt( ((1-concc)*self.ow_sdev)**2 + (concc*self.cice_sdev)**2 ) # Just calculate this once.
            conc_sdev = sqrt_tmp \
                + np.floor(np.abs(concc)) * ( np.sqrt( ((concc - 1.)*self.ow_sdev)**2 \
                + ((2. - concc)*self.cice_sdev)**2 ) - sqrt_tmp ) \
                + np.abs(np.floor(concc/2.)) * ( np.sqrt( ((1.-np.abs(concc))*self.ow_sdev)**2 \
                + (np.abs(concc)*self.cice_sdev)**2 ) - sqrt_tmp )
        else:
            conc_sdev = None

        # Compute dal as well.
        dal = self._compute_dal(tbs, dal_type=dal_type, sics=C)

        # now compute the OWF mask (True: most-probably open water)
        owf = self._compute_owf(tbs, C)

        # return in an object
        ret = SICAlgoResult(C, conc_sdev, dal, owf)
        return ret

    def _compute_dal(self, tbs, dal_type='classic', sics=None):

        # check the type of dal
        if dal_type not in ('classic','normalized', 'owf'):
            raise ValueError("Do not know that type of dal: {})".format(dal_type))

        if dal_type != 'classic' and sics is None:
            raise ValueError("dal_type is not 'classic' so we need sics=")

        # check all the required channels are given as input
        for ch in self.channels:
            if ch not in tbs.keys():
                raise ValueError("Tb channel {} is missing as input".format(ch))

        # stack the Tbs
        Tbs   = np.column_stack(([tbs[ch] for ch in self.channels],))

        # dimensions (n = nbchannels, N=nbTbs)
        n,N = Tbs.shape
        if n != self.n:
            raise ValueError("Dimensionality problem! Got {} Tb channels, but the algo is {}-dim".format(n,self.n))

        # classic DAL: u.Tbs
        U = self.cice.u.reshape(n,1)
        U = U.repeat(N,axis=1)
        dal = np.einsum("ji,ji->i", U, Tbs)
        # normalized DAL: modify the classic DAL with SIC
        if dal_type == 'normalized' or dal_type == 'owf':
            dal_cice = np.dot(self.cice.u,self.cice.fyi_tp)
            if dal_type == 'normalized':
                dal_ow   = np.dot(self.cice.u,self.ow.tp)
            elif dal_type == 'owf':
                dal_ow   = np.dot(self.cice.u,self.owf_lw)
            
            dal_ref = (1. - sics) * dal_ow + sics * dal_cice
            dal -= dal_ref

        return dal

    def _compute_owf(self, tbs, sics):
        # we need "owf" type of DAL
        dal = self._compute_dal(tbs, dal_type='owf', sics=sics)

        # by definition dal of owf_lw is 0. get dal of owf_hw
        dal_hw = np.dot(self.cice.u,self.owf_hw)

        # first part, flat SIC threshold (has effect for dal<0)
        owf_1 = (sics < self.owf_threshold)
        # second part, increasing SIC threshold (has effect for dal>0)
        sic_limit = self.owf_threshold + dal * (0.5 - self.owf_threshold) / dal_hw
        owf_2 = (sics < sic_limit)

        return (owf_1 + owf_2)

    """ UNCERTAINTIES """
    @staticmethod
    def _get_stddev_percent(var):
        #transform variance in (0-1) range to stddevs in (0-100%) range
        return 100. * np.sqrt(var)

    def _compute_algo_uncert_tiepoints(self, ret_stddev_percent=False, indx=None):

        # compute variance of the algorithm uncertainty at 0% and 100%,
        #    *** using theoretical uncertainty propagation formula ***
        #    by default this returns both ow, and cice values (in that order), but
        #    one can use indx=0,1 to return only one of these.

        if self.v is None:
            raise ValueError("v is None... algorithm is not tuned!")

        if indx is not None:
            if indx not in (0,1,):
                raise ValueError("if indx= is specified, it should be 0:ow or 1:cice")

        # prepare unit vector v
        v = self.v

        # just to be extra sure.
        self._check_v()

        # OW accuracy
        Sow = np.dot(np.dot(v.T,self.ow.C),v)

        # CI accuracy
        Sci = np.dot(np.dot(v.T,self.cice.C),v)

        # normalization
        W = self.ow.tp
        I = self.cice.tp
        norm = (np.dot(v,I - W)**2)

        #return
        ret = np.array([Sow, Sci,]) / norm

        if ret_stddev_percent:
            ret = self._get_stddev_percent(ret)

        # implement the indx
        if indx is not None:
            ret = ret[indx]

        return ret

    def _compute_algo_uncert_tiepoints_stat(self, ret_stddev_percent=False, indx=None):

        # compute variance of the algorithm uncertainty at 0% and 100%,
        #    *** using statistical evaluation of the algorithm ***
        #    by default this returns both ow, and cice values (in that order), but
        #    one can use indx=0,1 to return only one of these.

        if self.v is None:
            raise ValueError("v is None... algorithm is not tuned!")

        if indx is not None:
            if indx not in (0,1,):
                raise ValueError("if indx= is specified, it should be 0:ow or 1:cice")

        # prepare unit vector v
        v = self.v

        # just to be extra sure.
        self._check_v()

        # collect the input Tbs from the tie-point signatures
        ow_dict = dict()
        ci_dict = dict()
        for ch in self.channels:
            ow_dict[ch] = getattr(self.ow,ch)
            ci_dict[ch] = getattr(self.cice,ch)

        # statistical variances
        var_stat_ow = 0
        var_stat_ci = 0

        if indx is None or indx == 0:
            var_stat_ow = self.compute_sic(ow_dict).get('sic').var()

        if indx is None or indx == 1:
            var_stat_ci = self.compute_sic(ci_dict).get('sic').var()

        #return
        ret = np.array([var_stat_ow, var_stat_ci,])

        if ret_stddev_percent:
            ret = self._get_stddev_percent(ret)

        # implement the indx
        if indx is not None:
            ret = ret[indx]

        return ret

    def _check_algo_uncert_tiepoints(self,):
        # compute the uncertainty both theoretically and statistically and
        #   compare the results.

        # theoretical variances
        var_theo_ow, var_theo_ci = self._compute_algo_uncert_tiepoints()

        # statistical variances
        var_stat_ow, var_stat_ci = self._compute_algo_uncert_tiepoints_stat()

        check_ok = True
        if ( abs(var_theo_ow - var_stat_ow) > 1.e-3 ):
            check_ok = False
            print("WARNING: OW theoretical variance differs from statistical variance: {} {}".format(var_theo_ow,var_stat_ow))

        if ( abs(var_theo_ci - var_stat_ci) > 1.e-3 ):
            check_ok = False
            print("WARNING: CI theoretical variance differs from statistical variance: {} {}".format(var_theo_ow,var_stat_ow))

        if not check_ok:
            raise ValueError("Issue with theo vs stats variance.")

    def _compute_radiometric_uncert(self, ret_stddev_percent=False):
        # compute variance of the uncertainty term induced by NeDT,
        #    using uncertainty propagation formula

        # prepare unit vector v
        v = self.v

        # just to be extra sure.
        self._check_v()

        # radiometric noise
        Srad = np.dot(np.dot(v.T,self.nedt_C),v)

        # normalization
        W = self.ow.tp
        I = self.cice.tp
        norm = (np.dot(v,I - W)**2)

        #return
        ret = Srad / norm

        if ret_stddev_percent:
            ret = self._get_stddev_percent(ret)

        return ret

    def get_coeffs(self, expand=False):

        if self.v is None:
            raise ValueError("This algorithm has not been tuned yet!")

        # access Water and Ice signature
        W = self.ow.tp
        I = self.cice.tp

        # prepare unit vector v
        v = self.v

        # compute the n-dim coefficients of the linear combination
        denom = np.dot( v, (I - W) )
        coeffs_mul = v / denom
        coeffs_add = np.dot( v, -W ) / denom
        c = np.append( coeffs_mul, coeffs_add )

        # if requested, expand to all channels of a pmr (provided in dict())
        if expand:
            raise NotImplementedError("expand=True not yet implemented")

        return c

    def code(self,fmt='pseudo'):

        c = self.get_coeffs()
        s = ''

        if fmt == 'pseudo':
            s += 'sic = '
            for ich, ch in enumerate(self.channels):
                sign = ''
                spac = ''
                if (ich > 0):
                    sign = '+'
                    spac = ' '

                if c[ich] < 0.:
                    sign = '-'
                s += '{}{}{} * {} '.format(sign, spac, abs(c[ich]), ch)

            sign = '+'
            if c[-1] < 0.:
                sign = '-'
            s += '{} {}'.format(sign,abs(c[-1]))
        elif fmt == 'python':
            s += 'def sic_code({},): '.format(','.join(self.channels))
            s += 'return'
            for ich, ch in enumerate(self.channels):
                s += ' {:+.12f} * {}'.format(c[ich],ch)
            s += ' {:+.12f}'.format(c[-1])
        elif fmt == 'latex':
            s += '$C = '
            fmt = '{:+.6f}'
            for ich, ch in enumerate(self.channels):
                s += ' ' + fmt.format(c[ich]) + ' \\times T_{{{}}}'.format(ch[2:])
            s += ' '+ fmt.format(c[-1]) + ' $'
        else:
            raise NotImplementedError('Code format {} not implemented yet'.format(fmt,))

        return s

    def code_algo_uncert(self,fmt='pseudo',ret_stddev_percent=False):

        v = self._compute_algo_uncert_tiepoints(ret_stddev_percent=False)
        s = ''
        if fmt == 'pseudo':
            raise NotImplementedError("TBD")
        elif fmt == 'python':
            raise NotImplementedError("TBD")
        elif fmt == 'latex':
            symb = '\\Sigma_C'
            if ret_stddev_percent:
                symb = '\\sigma_C'
            s += '$ {} = '.format(symb)
            if ret_stddev_percent:
                s += '100 \\times \\sqrt{'
            fmt = '{:+.6f}'
            s += fmt.format(v[0]) + ' \\times ( 1 - C )^2 ' 
            s += fmt.format(v[1]) + ' + C^2 '
            if ret_stddev_percent:
                s += '}' 
            s += ' $'
        else:
            raise NotImplementedError('Code format {} not implemented yet'.format(fmt,))  

        return s

    def _check_python_code(self, ):
        # collect the input Tbs from the tie-point signatures
        ow_tup = []
        ci_tup = []
        for ch in self.channels:
            ow_tup.append( self.ow.tp[ self.ow.channels.index(ch) ] )
            ci_tup.append( self.cice.tp[ self.cice.channels.index(ch) ] )

        code = self.code(fmt='python')
        exec( code , globals())
        ow_eval = sic_code(*ow_tup)
        ci_eval = sic_code(*ci_tup)

        if ( abs(ow_eval - 0.) > 1.e-3 ):
            raise ValueError("ERROR: the python software returns {:f} at the OW tie-point.".format(ow_eval))
        if ( abs(ci_eval - 1.) > 1.e-3 ):
            raise ValueError("ERROR: the python software returns {:f} at the CICE tie-point.".format(ci_eval))

    def plot(self, ax=None, **kwargs):

        # prepare a new axis if needed
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(1,1,1)

        # handle default values
        if 'x' not in kwargs.keys():
            kwargs['x'] = 'dal'
        if 'y' not in kwargs.keys():
            kwargs['y'] = 'sic'
        if 'what' not in kwargs.keys():
            kwargs['what'] = ('ow','cice')
        else:
            if isinstance(kwargs['what'],str):
                kwargs['what'] = [kwargs['what'],]
        if 'finalize' not in kwargs.keys():
            kwargs['finalize'] = False
        if 'grid' not in kwargs.keys():
            kwargs['grid'] = False
        if 'with_samples' not in kwargs.keys():
            kwargs['with_samples'] = True
        if 'dal' in (kwargs['x'],kwargs['y']):
            if 'dal_type' not in kwargs.keys():
                kwargs['dal_type'] = 'classic'

        xyvals = dict()
        # prepare x and y data points depending on the 'x' and 'y' keywords
        for xy in ('x','y',):
            xyvals[xy] = dict()
            xywhat = kwargs[xy]
            if xywhat.startswith('dal'):
                for w in kwargs['what']:
                    xyvals[xy][w] = self.compute_sic(getattr(self,w), dal_type=kwargs['dal_type']).get('dal')
            elif xywhat == 'sic':
                for w in kwargs['what']:
                    xyvals[xy][w] = self.compute_sic(getattr(self,w),).get('sic')

        # do the plotting of the samples
        if kwargs['with_samples']:
            for w in kwargs['what']:
                ax.plot(xyvals['x'][w],xyvals['y'][w],'.',markersize=2)
                #if w == 'ow':
                #    owf = self.compute_sic(self.ow,).get('owf')
                #    ax.plot(xyvals['x'][w][owf],xyvals['y'][w][owf],'.',markersize=2,color='cyan')

        # further configure the plot
        if kwargs['grid']:
            ax.grid()

        # finalize the plot
        if kwargs['finalize']:
            _axis_label = {'dal':"Distance Along the Line", 'sic':"Sea Ice Concentration"}
            if kwargs['dal_type'] == 'owf':
                _axis_label['dal'] += r' ($d_{OWF}$)'
            else:
                _axis_label['dal'] += r' ($d$)'

            ax.set_xlabel(_axis_label[kwargs['x']])
            ax.set_ylabel(_axis_label[kwargs['y']])
            if kwargs['y'] == 'sic':
                for spanl in (0,1):
                    ax.axhline(y=spanl, lw=1, ls='--', color='k')
            if kwargs['x'] == 'sic':
                for spanl in (0,1):
                    ax.axvline(x=spanl, lw=1, ls='--', color='k')

        return ax


class SICAlgoWithCIL(SICAlgo):

    keep = tuple(list(SICAlgo.keep,) + ['seg_cil','seg_dal',])
    keep_methods = tuple(list(SICAlgo.keep_methods,) + ['_compute_cil',])

    def __init__(self, channels, ow, cice, pmr=None, tuning=None, nseg=53, percentiles=(1,99), min_count=15, owf_threshold=False, ):

        self.channels, self.n = check_init_params(channels, ow, cice)

        # Curvy Ice Line requires DAL, and this requires at least 2 channels to work.
        if len(self.channels)  < 2:
            raise ValuerError("Cannot build a SICAlgosWithCIL with n==1")

        # prepare a SICAlgo object (note: no tuning)
        super().__init__(channels, ow, cice, pmr=pmr, tuning=None, owf_threshold=owf_threshold, )

        # prepare the CIL information
        self._prepare_cil(self.cice, nseg=nseg, percentiles=percentiles, min_count=min_count)

        # tune if requested
        if tuning is not None:
            self.tune(tuning=tuning)

        # store the dal and sic of segments (final, after tuning)
        self.seg_cil, self.seg_dal = super()._compute_sic(self.seg_cice_tps, dal_type='classic').get(('sic','dal'))

    def tune(self, tuning='random'):

        # enforce 'statistical' tuning
        if '-theo' in tuning:
            raise ValueError("A SICAlgoWithCIL cannot be tuned with -theo")

        if tuning in ('ow','cice'):
            tuning += '-stat'

        # tune with super()
        super().tune(tuning)


    def _prepare_cil(self, cice, nseg=10, percentiles=(1,99), min_count=50):

        # segment the cice tiepoint object
        loc_cice = cice.extract_channels( self.channels )
        seg_cice = tp.CICETiepointDalSegments(loc_cice, nseg=nseg, percentiles=percentiles, min_count=min_count)

        # Tbs of the center point in each segment:
        cice_curve = seg_cice.tps

        # some segments will have too few samples, and are invalid. Fix these now, by replacing them
        #    with points along the (straight) ice line
        invalid_segments = seg_cice.tps.mask.any(axis=0)

        # prepare the "straight" ice line
        xs = seg_cice.dal_centers - np.dot(loc_cice.u, loc_cice.tp)
        cice_line = (loc_cice.tp + np.outer ( xs, loc_cice.u )).T

        # merged line (curvy where enough samples, straight where missing samples)
        cice_mix = np.where(invalid_segments, cice_line, cice_curve)

        # record the cice tp from each segment. We want only the valid segments.
        self.seg_cice_tps = dict()
        for ich, ch in enumerate(self.channels):
            self.seg_cice_tps[ch] = cice_mix[ich]
            if len(self.seg_cice_tps[ch].shape) > 1:
                self.seg_cice_tps[ch] = np.squeeze(self.seg_cice_tps[ch])

        # clean-up
        del loc_cice
        del seg_cice

    def _compute_cil(self, dals, interp='linear'):

        if interp not in ('linear','spline'):
            raise ValueError('The CIL can only be prepared with interp="linear" or ="spline"')

        try:
            # algorithm is already tuned, and thus has its final seg_cil and seg_dal
            seg_cil = self.seg_cil
            seg_dal = self.seg_dal
        except AttributeError:
            # we are in the process of tuning... so we must compute temporary seg_cil and seg_dal
            seg_cil, seg_dal = super()._compute_sic(self.seg_cice_tps, dal_type='classic').get(('sic','dal'))

        if interp == 'spline':
            # fit spline (note: 1d spline, thus splrep)
            cil_tck = splrep(seg_dal, seg_cil, k=3)

            # spline interpolation
            cil = splev(dals, cil_tck)
        else:
            # 1d linear interpolation with extrapolation outside the bounds
            F = interp1d(seg_dal, seg_cil, kind='linear', bounds_error=False, fill_value='extrapolate')
            cil = F(dals)

        return cil

    def _compute_sic(self, tbs, dal_type='classic'):

        # compute the 'classic' dal values
        dal = self._compute_dal(tbs, dal_type='classic')

        # use the dals to interpolate the CIL
        cil = self._compute_cil(dal)

        # now call the compute_sic of the SICAlgo.
        #   tbs is already pre-processed to something _compute_sic() can
        #      process, so we skip the compute_sic() step.
        try:
            ret = super()._compute_sic(tbs, dal_type=dal_type)
        except TypeError:
            # this is a bit ugly. when the object is loaded from json, it does not know
            #   its mother-class, so we need to explicitely call SICAlgo._compute_sic()
            ret = SICAlgo._compute_sic(self, tbs, dal_type=dal_type)

        # correct sic for CIL
        ret.sic /= cil

        return ret

class SICAlgoWithSegments(SICAlgo):

    def __init__(self,channels, ow, cice, pmr=None, tuning=None, nseg=53, percentiles=(1.,99.), min_count=15,):

        # various checks and pre-processing
        self.channels, self.n = check_init_params(channels, ow, cice)
        channels = self.channels

        # segmented algos require DAL, and this requires at least 2 channels to work.
        if len(self.channels)  < 2:
            raise ValuerError("Cannot have segmented SICAlgos with n==1")

        # we store only one OW tie-point information (the same for all segments)
        try:
            if ow.channels != self.channels:
                self.ow   = ow.extract_channels(self.channels)
            else:
                self.ow   = ow
        except Exception as ex:
            raise ValueError("Unable to extract required channels from the OW or CICE signature ({})".format(ex,))

        # we segment the cice object if needed
        try:
            # the next statement will fail in case of non-segmented tiepoint
            nseg = len(cice.segments)
            # already a segmented tie-points, check it has the correct channels.
            if set(cice.channels) != set(self.channels):
                msg = "The CI tie-points object is already segmented but does not have the correct channels."
                msg += "\n\tYou must extract_channels() prior to build the algo, or give a non-segmented tie-point object."
                raise ValueError( msg )
        except AttributeError:
            # not already segmented. We must first extract the channels, then segment it for later internal use.
            #   we do extract_channels anyway, to get a copy of the object
            loc_cice = cice.extract_channels( self.channels )
            seg_cice = tp.CICETiepointDalSegments(loc_cice, nseg=nseg, percentiles=percentiles, min_count=min_count)
            # we store the segmented cice object for later use (but most of the work should be done by the segmented algos
            #    that are to be created shortly after
            self.cice = seg_cice

        # TODO: do we really need this call to super() ?
        #    call super() constructor
        #print("call super().__init__()")
        #super().__init__(self.channels, ow_sel, cice, pmr=pmr, tuning=tuning)

        # additional attributes
        self.tuned    = False

        # we create and store a list of SICAlgo objects for the CI part
        self.algos = [None,]*len(self.cice.segments)
        for iseg, seg in enumerate(self.cice.segments,):
            if seg is not None:
                self.algos[iseg] = SICAlgo(self.channels, self.ow, seg, pmr=pmr, tuning=tuning)

        # finalize tuning
        if tuning is not None:
            self.tuned = tuning
            self._finalize_tuning()

    # len() and count(): number of segments, and number of active segments
    def __len__(self):
        return len(self.algos)

    def count(self):
        return len(self.algos) - self.algos.count(None)

    def _iterate_on_algos(self, fn, *args, **kwargs):
        for a in self.algos:
            if a is not None:
                getattr(a,fn)(*args, **kwargs)

    def _check_all_v(self,):
        self._iterate_on_algos('_check_v')

    def _finalize_tuning(self):
        # save the state of tuning
        # prepare spline interpolation of v
        self._set_vs()
        self.v_tck = splprep(self.vs, u=self.dal_centers, k=3)[0]
        self.z_tck = splprep(self.zs, u=self.dal_centers, k=3)[0]

    def tune(self, tuning='random'):
        # tune each segment
        self._iterate_on_algos('tune',tuning=tuning)
        self.tuned = tuning
        # finalize
        self._finalize_tuning()

    def _check_python_code(self):
        self._iterate_on_algos('_check_python_code')

    def _set_vs(self):
        self.vs   = np.ma.empty((len(self.channels),len(self)))
        self.zs   = np.ma.empty((len(self.channels)-2,len(self)))
        self.dal_centers = np.ma.empty((len(self),))
        for si, s in enumerate(self.algos):
            if s is None:
                self.vs[:,si] = np.ma.masked
                self.zs[:,si] = np.ma.masked
                self.dal_centers[si] = np.ma.masked
            else:
                self.vs[:,si] = s.v
                self.zs[:,si] = s._get_z_from_v(s.v)[1:]
                self.dal_centers[si] = s.cice.dals[0]

    def _get_vs_from_zs(self,z):

        z = self._preproc_z(z)
        v = np.squeeze(np.asarray(z @ self.M.T))

        return v

    # re-implement _compute_sic to act on segments
    def _compute_sic(self, tbs, with_dal=False):

        # check all the required channels are given as input
        for ch in self.channels:
            if ch not in tbs.keys():
                raise ValueError("Tb channel {} is missing as input".format(ch))

        # stack the Tbs
        Tbs   = np.column_stack(([tbs[ch] for ch in self.channels],))

        # dimensions (n = nbchannels, N=nbTbs)
        n,N = Tbs.shape
        if n != self.n:
            raise ValueError("Dimensionality problem! Got {} Tb channels, but the algo is {}-dim".format(n,self.n))

        # compute DAL
        U = self.cice.u.reshape(n,1)
        U = U.repeat(N,axis=1)
        dal = np.einsum("ji,ji->i", U, Tbs)

        # prepare the Water and Ice signatures
        W = self.ow.tp[:,np.newaxis]
        W = W.repeat(N,axis=1)

        I = self.cice.tps

        #I = I.repeat(N,axis=1)
        #I = np.squeeze(I)


        # interpolate the CI tie-point <I> (use spline coeffs tck prepared by the
        #    segmented CI tiepoint object)
        I = splev(dal, self.cice.tp_tck)
        I = np.vstack(I)

        # interpolate vector v (use spline coeffs tck prepared in self.tune)
        Z = splev(dal, self.z_tck)
        Z = np.vstack(Z)
        print(Z,Z.shape)

        V = splev(dal, self.v_tck)
        V = np.vstack(V)
        # re-norm V to being a unit vector
        print(np.linalg.norm(V,axis=0))

        print(N,n,W.shape,I.shape,V.shape)
        

