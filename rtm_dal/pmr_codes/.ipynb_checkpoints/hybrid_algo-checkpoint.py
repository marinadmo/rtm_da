
import numpy as np

#from . import common
#from . import algo as algo
#from . import tiepoints as tp
#from . import pmr as pm
import common
import algo as algo
import tiepoints as tp
import pmr as pm

class HybridSICAlgo(object):

    def __init__(self, ow_channels, ow_tiepoint, cice_tiepoint, pmr=None, cice_channels=None, curvy_ice_line=False, 
                 owf_threshold=0.1,):

        # if only ow_channels is provided, then use the same for cice_channels
        if cice_channels is None:
            cice_channels = ow_channels

        # create the OW and CICE algorithms
        bow = algo.SICAlgo(ow_channels, ow_tiepoint, cice_tiepoint, pmr=pmr, tuning='ow', \
                           owf_threshold=owf_threshold,)
        if not curvy_ice_line:
            bci = algo.SICAlgo(cice_channels, ow_tiepoint, cice_tiepoint, pmr=pmr, tuning='cice', \
                               owf_threshold=owf_threshold,)
        else:
            bci = algo.SICAlgoWithCIL(cice_channels, ow_tiepoint, cice_tiepoint, pmr=pmr, tuning='cice', \
                                      owf_threshold=owf_threshold,)

        # store in self:
        self.bow = bow
        self.bci = bci

    def strip(self, copy_first=True):
        keep = ('bow', 'bci', )
        return common.strip(self, keep, copy_first=True)

    def to_dict(self,):
        return common.to_dict(self)

    @classmethod
    def from_dict(cls, dct):
        # use the from_dict() class methods
        bow = algo.SICAlgo.from_dict(dct['bow'])
        if 'seg_cil' in dct['bci'].keys():
            bci = algo.SICAlgoWithCIL.from_dict(dct['bci'])
        else:
            bci = algo.SICAlgo.from_dict(dct['bci'])

        # create HybridSICAlgo object (container, no methods)
        obj = type('@'+cls.__name__, (object,), {'bow':bow, 'bci':bci})()

        # add the few needed methods
        #      the __get__(obj) is to bound the method to the instance
        #      (and thus having 'self' to be passed as 1st parameter)
        for meth in ('compute_sic','_blend',):
            setattr(obj, meth, getattr(cls,meth).__get__(obj))

        return obj

    def _blend(self, sics, devs=None, ):
        bow_conc, bci_conc = sics
        if devs is not None:
            bow_sdev, bci_sdev = devs

        beg_blending = 0.7
        end_blending = 0.9
        len_blending = end_blending - beg_blending
        wCF = np.where((bow_conc >= beg_blending) * (bow_conc <= end_blending),\
            1. - (bow_conc - beg_blending)/len_blending,\
            1.)
        wCF[bow_conc > end_blending] = 0.

        conc = (1 - wCF) * bci_conc + wCF * bow_conc
        ret = [conc,]
        if devs is not None:
            sdev = ((1 - wCF) * bci_sdev**2 + wCF * bow_sdev**2)**0.5
            ret.append(sdev,)

        if len(ret) == 1:
            ret = ret[0]

        return ret

    def compute_sic(self, tbs, dal_type='classic'):

        # call _compute_sic for each of the BOW and BCI
        bow_ret = self.bow.compute_sic(tbs, dal_type=dal_type)
        bci_ret = self.bci.compute_sic(tbs, dal_type=dal_type)

        # now combine SIC with the hybrid merger equation
        hybrid_ct, hybrid_sdev = self._blend((bow_ret.sic,bci_ret.sic),devs=(bow_ret.sdev,bci_ret.sdev))

        # DAL is always that of the bci algorithm
        hybrid_dal = bci_ret.dal

        # OWF is always that of the bow algorithm
        hybrid_owf = bow_ret.owf

        return algo.SICAlgoResult(hybrid_ct, hybrid_sdev, hybrid_dal, hybrid_owf)

    def plot(self,**kwargs):

        # if finalize is asked, we only finalize the last plotting
        finalize_last = False
        if 'finalize' in kwargs.keys() and kwargs['finalize']:
            finalize_last = True
        kwargs.pop('finalize', None)

        ax_provided = 'ax' in kwargs.keys()
        ax = self.bow.plot(what='ow',**kwargs,)

        if not ax_provided:
            kwargs['ax'] = ax

        ax = self.bci.plot( what='cice', **kwargs, finalize=finalize_last)
        return ax

