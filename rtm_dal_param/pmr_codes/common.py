
import numpy as np
from copy import copy

def strip(self, keep, copy_first=True):
    # Strip the SICAlgo object to the bare minimum to be able
    #   to compute SIC (no tuning possible).
    if copy_first:
        obj = copy(self)
    else:
        obj = self

    for k in tuple(vars(obj).keys()):
        if not k in keep:
            delattr(obj, k)
        else:
            if hasattr(getattr(obj,k),'strip'):
                setattr(obj,k,getattr(obj,k).strip(copy_first=copy_first))

    return obj

def to_dict(obj, exclude=None, keep=None):

    # take shallow copy of the dict() version of the object
    ret = vars(obj).copy()

    # prepare defaul list of keeps and excludes
    if keep is not None and exclude is not None:
        raise ValueError("You can only use one of 'keep=' and 'exclude=' when calling to_dict()")

    if keep is None:
        keep = list(ret.keys())
    if exclude is None:
        exclude = ()

    # iterate on the keys
    for kr in tuple(ret.keys()):
        # remove some keys (via exclude= or keep=)
        if kr in exclude or kr not in keep:
            ret.pop(kr,None)
            continue

        # enter the to_dict() methods 
        try:
            ret[kr] = ret[kr].to_dict()
        except AttributeError:
            pass

    # return to caller
    return ret

def jsonify(self):
    if hasattr(self,'to_dict'):
        dct = self.to_dict()
    else:
        dct = self
    for elem in list(dct.keys()):
        if isinstance(dct[elem],np.ndarray):
            dct[elem] = np.around(dct[elem],7).tolist()
        elif isinstance(dct[elem],dict):
            dct[elem] = jsonify(dct[elem])
    return dct

def numpify(dct):
    for elem in list(dct.keys()):
        if isinstance(dct[elem],list):
            dct[elem] = np.asarray(dct[elem])
        elif isinstance(dct[elem],dict):
            dct[elem] = numpify(dct[elem])
    return dct
