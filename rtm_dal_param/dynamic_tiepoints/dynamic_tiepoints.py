#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
    Module for handling dynamical tiepoints.

    Dynamical tiepoints are a the result of a principal component analysis of
    the input brightness temperatues catagorised by an initial processing using
    the Nasa Team ice concentration algorithm.

    The brightness temperatures are catagorized into two groups:
    * Open water points, selected from a open water mask
    * Sea ice points, selected from the nasa team algorithm with probability above 95 %
"""

import logging
import numpy as np
import numpy.ma as ma
import re
from numpy import cos, sin, sqrt
import os, sys
sys.modules['snappy'] = None
import pyresample as pr
import types
from math import ceil, floor
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geo_mask
import conc_utils
import ice_conc_algo as ica

LOG = logging.getLogger(__name__)

ALGOS = {'amsr':['sicci3lf'],
         'smmr':['sicci3lf'],
         'ssmi':['sicci3lf'],
         'mwri':['sicci3lf'],
         'mwi':['sicci3lf']}

VALID_ALGOS = ['comiso', 'osisaf', 'sicci1', 'sicci2lfsil', 'sicci2lf',
    'sicci2hfsil', 'sicci2hf', 'n90linsil', 'sicci2vlfsil', 'sicci2vlf',
    'n90lin', 'sicci3lf', 'sicci3af', 'sicci3k4', 'sicci3hf', 'sicci3vlf']

HF_ALGOS = ['sicci2hfsil', 'sicci2hf', 'n90linsil', 'n90lin',
    'sicci3af', 'sicci3hf']

def getcorr(corr):
    if (corr == None):
        return 'ucorr'
    elif (corr.lower() == 'ucorr'):
        return 'ucorr'
    elif (corr.lower() == 'bristol'):
        return 'corrBri'
    else:
        return 'corr'+corr.upper()

def vnorm(m):
    """norms of a matrix of column vectors.
    """
    return np.sqrt((m**2).sum(0))

def angle3d(v1,v2):
    """ return angle in radians between two (3d) vectors"""
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

class Quaternion(object):

    def __init__(self, scalar, vector):
        self.__x, self.__y, self.__z = vector
        self.__w = scalar

    def rotation_matrix(self):
        x, y, z, w = self.__x, self.__y, self.__z, self.__w
        zero = np.zeros_like(x)
        return np.array(
            ((w**2 + x**2 - y**2 - z**2,
              2*x*y + 2*z*w,
              2*x*z - 2*y*w,
              zero),
             (2*x*y - 2*z*w,
              w**2 - x**2 + y**2 - z**2,
              2*y*z + 2*x*w,
              zero),
             (2*x*z + 2*y*w,
              2*y*z - 2*x*w,
              w**2 - x**2 - y**2 + z**2,
              zero),
             (zero, zero, zero, w**2 + x**2 + y**2 + z**2)))

def qrotate(vector, axis, angle):
    """Rotate *vector* around *axis* by *angle* (in radians).

    *vector* is a matrix of column vectors, as is *axis*.
    This function uses quaternion rotation.
    """

    n_axis = axis / vnorm(axis)
    sin_angle = np.expand_dims(sin(angle/2), 0)
    if np.ndim(n_axis)==1:
        n_axis = np.expand_dims(n_axis, 1)
        p__ = np.dot(n_axis, sin_angle)[:, np.newaxis]
    else:
        p__ = n_axis * sin_angle

    q__ = Quaternion(cos(angle/2), p__)
    return np.einsum("kj, ikj->ij",
                     vector,
                     q__.rotation_matrix()[:3, :3])

def _pca3d(x,y,z):
    """ PCA for 3 dimensional data """
    return _pca(np.vstack((x, y, z)))

def _pca(coords):
    """ Performs a Principal Component Analysis of coords = np.vstack(x,y,z,...) """

    # compute eigen-elements of the covariance matrix
    eigenvals, eigenvectors = np.linalg.eig(np.cov(coords))

    # rank the eigenvalue(s)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvectors = eigenvectors[:,idx]

    # return eigen vectors, values, and mean point
    return eigenvectors, eigenvals, np.mean(coords, axis=1)

def get_tag(method, channels, corr='ucorr'):
    #return '{} ({},{},{}) {}'.format(method,channels[0],channels[1],channels[2],getcorr(corr))
    return '{} ({}) {}'.format(method, ','.join(channels), getcorr(corr))


class ChannelTransform(object):
    """ Class for holding the channel transformation parameters
        to transform 3D measured Tbs into a 2D axes. This generalizes
        what Bristol algorithm does.

        A ChannelTransform object also knows its uncertainties at
        SIC = 0 and SIC = 1 ends
    """
    def __init__(self,channels,a,b,c,d,e,f,multX=1,multY=1,o1=0.,o2=0.,T=0.,Tx=0.,Ty=0.,normalized=False,ow_sdev=None,cice_sdev=None):
        """
           Construct from parameters
        """

        # record the channels used to define the ChannelTransform
        if len(channels) != 3:
            raise ValueError("All ChannelTransforms shall be 3D (received {} channels)".format(channels,))
        self.channels = channels

        # parameters to X=
        self.a  = a
        self.b  = b
        self.c  = c
        # parameters to Y=
        self.d  = d
        self.e  = e
        self.f  = f
        # rotation in (X,Y) space (of angle T, and center Tx,Ty)
        self.T   = T
        self.Tx  = Tx
        self.Ty  = Ty
        # scaling of axis
        self.multX = multX
        self.multY = multY
        # shift of origin
        self.o1 = o1
        self.o2 = o2

        # Setting uncertainties at OW and CICE (by default not known)
        self.ow_sdev   = ow_sdev
        self.cice_sdev = cice_sdev

        # Setting the normalization state (by default False)
        self.normalized = normalized

    def __str__(self):
        """ String representation  of a ChannelTransform """
        ret  = "ChannelTransform ({},{},{}) -> (X,Y)\n".format((self.channels)[0],(self.channels)[1],(self.channels)[2])
        ret += "X =  ROT({:.2f} deg,{:+0.4f} * Tbch1 {:+0.4f} * Tbch2 {:+0.4f} * Tbch3) {:+0.4f}\n".format(
            self.T,
            self.a, self.b,
            self.c, self.o1)
        ret += "Y =  ROT({:.2f} deg,{:+0.4f} * Tbch1 {:+0.4f} * Tbch2 {:+0.4f} * Tbch3) {:+0.4f}\n".format(
            self.T,
            self.d, self.e,
            self.f, self.o2)
        ret += "SDEV OW: {} CICE: {}".format(self.ow_sdev, self.cice_sdev)

        return ret

    def set_uncertainties(self,ow_sdev,cice_sdev):
        """ initialize the OW and CICE uncertainties """
        self.ow_sdev = ow_sdev
        self.cice_sdev = cice_sdev


    def apply(self, tbch1, tbch2=None, tbch3=None):
        """
           Apply a transformation on a triplet of Tbs.

           tbch1 is either all 3 channels as (:,3) or only ch1
             (in which case tbch2 and tbch3 must be provided)
        """

        if tbch1.ndim == 2 and tbch1.shape[1] == 3:
            if tbch2 is not None or tbch3 is not None:
                raise TypeError("When tbch1 is given as {}, then 2nd and 3rd parameters must not be provided".format(tbch1.shape))
            else:
                tbch2 = tbch1[:,1]
                tbch3 = tbch1[:,2]
                tbch1 = tbch1[:,0]
        elif tbch1.ndim > 2 or (tbch1.ndim == 2 and tbch1.shape[1] != 3):
            raise TypeError("tbch1 must either be (N,) or (N,3) (got {})".format(tbch1.shape))
        elif tbch1.ndim == 1 and (tbch2 is None or tbch3 is None):
            raise TypeError("tbch1 is (N,) but we have not tbch2= or tbch3=")

        # apply the transformation
        B1 =  self.a * tbch1 + self.b * tbch2 + self.c * tbch3
        B2 =  self.d * tbch1 + self.e * tbch2 + self.f * tbch3

        theta = np.deg2rad(self.T)
        B1 -= self.Tx; B2 -= self.Ty
        Bp1 = B1 * np.cos(theta) - B2 * np.sin(theta)
        Bp2 = B2 * np.cos(theta) + B1 * np.sin(theta)
        Bp1 += self.Tx; Bp2 += self.Ty

        Bp1 *= self.multX
        Bp2 *= self.multY

        Bp1  += self.o1
        Bp2  += self.o2

        return Bp1, Bp2

    def compute_sic(self, tbch1, tbch2=None, tbch3=None, with_uncert=False, with_dal=False):
        """
           Compute Sea Ice Concentration for the triplet (tbch1, tbch2, tbch3)

           tbch1 is either all 3 channels as (:,3) or only ch1
             (in which case tbch2 and tbch3 must be provided)

           The ChannelTransform object must be normalized before-hand.
        """
        if not self.normalized:
            raise ValueError("Can only compute_sic() from a Normalized ChannelTransform.")

        if tbch1.ndim == 2 and tbch1.shape[1] == 3:
            if tbch2 is not None or tbch3 is not None:
                raise TypeError("When tbch1 is given as {}, then 2nd and 3rd parameters must not be provided".format(tbch1.shape))
            else:
                tbch2 = tbch1[:,1]
                tbch3 = tbch1[:,2]
                tbch1 = tbch1[:,0]
        elif tbch1.ndim > 2 and tbch1.shape[1] != 3:
            raise TypeError("tbch1 must either be (N,) or (N,3) (got {})".format(tbch1.shape))
        elif tbch1.ndim == 1 and (tbch2 is None or tbch3 is None):
            raise TypeError("tbch1 is (N,) but we have not tbch2= or tbch3=")

        # in a normalized ChannelTranform, ice concentration is simply the Y coordinate
        dal, conc = self.apply(tbch1,tbch2,tbch3)

        # if the ChannelTransform know its uncertainties at low and high ends,
        #    we also compute sdev(SIC) and return it.
        if (with_uncert and self.ow_sdev and self.cice_sdev):
            # 'Banana-shaped' uncertainty blending, with mirroring at 0 and 1.
            concc = np.clip(conc, -0.99, 1.99) # make sure iceconc doesn't reach -1 or 2
            sqrt_tmp = np.sqrt( ((1-concc)*self.ow_sdev)**2 + (concc*self.cice_sdev)**2 ) # Just calculate this once.
            conc_sdev = sqrt_tmp \
                + np.floor(np.abs(concc)) * ( np.sqrt( ((concc - 1.)*self.ow_sdev)**2 \
                + ((2. - concc)*self.cice_sdev)**2 ) - sqrt_tmp ) \
                + np.abs(np.floor(concc/2.)) * ( np.sqrt( ((1.-np.abs(concc))*self.ow_sdev)**2 \
                + (np.abs(concc)*self.cice_sdev)**2 ) - sqrt_tmp )
            if with_dal:
                ret = (conc, conc_sdev, dal)
            else:
                ret = (conc, conc_sdev,)
        else:
            if with_dal:
                ret = (conc, dal)
            else:
                ret = conc

        return ret

    def normalize(self,ow_tp,fyi_tp,myi_tp):
        """
           Normalize the axes transformation so that:
               1) the ice line is aligned with X axis;
               2) the ow point is at Y = 0
               3) the ice line is at Y = 1
               4) the myi point is at X = -1
               5) the fyi point is at X = +1
        """
        ow_tp    = np.expand_dims(ow_tp,0)
        fyi_tp   = np.expand_dims(fyi_tp,0)
        myi_tp   = np.expand_dims(myi_tp,0)

        if not self.normalized:
            # STEP 0: flip the axes direction
            ow_b1, ow_b2 = self.apply(ow_tp)
            fyi_b1, fyi_b2 = self.apply(fyi_tp)
            myi_b1, myi_b2 = self.apply(myi_tp)
            # we want the fyi to be on the 'right', and myi on the 'left'
            if fyi_b1[0] < myi_b1[0]:
                self.multX *= -1.

            # we want the ice points to be on 'top' and water to be on 'bottom'
            ice_b2 = 0.5 * (fyi_b2 + myi_b2)
            if ice_b2[0] < ow_b2[0]:
                self.multY *= -1.

            # STEP 1: rotation: we want the ice line to be parallel to the xaxis
            ow_b1, ow_b2 = self.apply(ow_tp)
            fyi_b1, fyi_b2 = self.apply(fyi_tp)
            myi_b1, myi_b2 = self.apply(myi_tp)
            slope = (myi_b2[0] - fyi_b2[0]) / (myi_b1[0] - fyi_b1[0])
            theta = -np.rad2deg(np.arctan(slope))
            self.T  = theta
            self.Tx = (fyi_b1+myi_b1)*0.5
            self.Ty = (fyi_b2+myi_b2)*0.5

            # STEP 2: normalize SIC (Y-axis) to a unit length, and DAL (X-axis) to a length of 2
            ow_b1, ow_b2 = self.apply(ow_tp)
            fyi_b1, fyi_b2 = self.apply(fyi_tp)
            myi_b1, myi_b2 = self.apply(myi_tp)
            # we want the ice_b2 to be at ow_b2 + 1
            ice_b2 = 0.5 * (fyi_b2[0] + myi_b2[0])
            self.multY  *= 1. / (ice_b2 - ow_b2[0])
            # we want the fyi_b2 to be at myi_b2 + 2
            self.multX  *= 2. / (fyi_b1[0] - myi_b1[0])

            # STEP3 : we want the (ow_b1,ow_b2) to be at Y = 0
            #    (hence the ice line at Y=1, since Y-axis has length unity)
            _, ow_b2 = self.apply(ow_tp)
            self.o2 = -ow_b2[0]

            # STEP4 : we want myi_b1 = -1 (hence fyi_b1 at +1)
            myi_b1, _ = self.apply(myi_tp)
            self.o1 = -myi_b1[0]-1

            self.normalized = True

        # check normalization:
        self._is_normalized(ow_tp,fyi_tp,myi_tp,verbose=True)

    def _is_normalized(self,ow_tp,fyi_tp,myi_tp,verbose=False):
        """ Check if the ChannelTransform is truely normalized """
        if not self.normalized:
            if verbose:
                print("self.normalized is set to False")
            return False
        # test the normalization
        ow_b1, ow_b2 = self.apply(ow_tp)
        fyi_b1, fyi_b2 = self.apply(fyi_tp)
        myi_b1, myi_b2 = self.apply(myi_tp)
        ice_b1 = 0.5 * (fyi_b1 + myi_b1)
        ice_b2 = 0.5 * (fyi_b2 + myi_b2)
        # check that the ice line is aligned at B2 = 1
        prec = 1.e-3
        if abs(myi_b2 - fyi_b2) > prec:
            if verbose:
                print("the ice line is not parallel to X axis (fyi_b2 = {} and myi_b2 = {})".format(fyi_b2,myi_b2))
            return False
        if abs(ice_b2 - 1.) > prec:
            if verbose:
                print("the ice line is not at Y = 1 (Y = {})".format(ice_b2,))
            return False
        if abs(ow_b2) > prec:
            if verbose:
                print("the water point is not at Y = 0.")
            return False
        if abs(myi_b1 + 1.) > prec:
            if verbose:
                print("the myi ice point is not at X = -1. ({})".format(myi_b1,))
            return False
        if abs(fyi_b1 - 1.) > prec:
            if verbose:
                print("the fyi ice point is not at X = +1. ({})".format(fyi_b1,))
            return False

    @classmethod
    def from_json(cls,method,channels,corr,cjson):
        """
           Constructor (hence the @classmethod) from json

           Can be called as ct = ChannelTransform.from_json(method,channels,corr,cjson)

           'cjson' can either be the json dictionnary or the content of a json file
           as a string.
        """
        # Parse the json string, and search for the name of the transform (with the right channels)
        tag = get_tag(method,channels,corr)
        if isinstance(cjson,str):
            cjson = json.loads(cjson)
        ctd = cjson[tag]['transf']
        return cls(channels,ctd['a'],ctd['b'],ctd['c'],ctd['d'],ctd['e'],ctd['f'],
                                multX=ctd['multX'],multY=ctd['multY'],
                                o1=ctd['o1'],o2=ctd['o2'],
                                T=ctd['T'],Tx=ctd['Tx'],Ty=ctd['Ty'],
                                normalized=ctd['normalized'],
                                ow_sdev=ctd['ow_sdev'],cice_sdev=ctd['cice_sdev'])


class PCA3D_DynamicTiepts(object):
    """ Class for dynamic tie-points from a PCA approach, that generalizes
       fcomiso, pcomiso, bristol, n90_lin, etc...
    """
    def __init__(self,
                method,
                channels,
                corr,
                cice_tbs,
                ow_tbs,
                fyi_tbs=None, myi_tbs=None,
                fix_the_ice_line=True,
                fit_curvy_ice_line=False):

        # TODO : the current setup is far from optimal, because most of the computations
        # (including the PCAs) will be the same for the best_ow and best_cice algorithms.
        # It would be better to be able to optimize both best_ow and best_cice in one go,
        # and return two ChannelTransforms instead of one.

        # take a safety copy of cice_tbs, since we will modify it internally
        cice_tbs = cice_tbs.copy()

        # check some type, lengths and dimensions
        try:
            fe = channels[0]
            if isinstance(channels,(str,)):
                raise TypeError("'channels' is a string")
        except TypeError as ex:
            raise TypeError("'channels' must be an array of strings ({})".format(ex,))

        if len(channels) != 3:
            raise ValueError("Expect 3 channels (got {})".format(len(channels,)))

        if not isinstance(cice_tbs, np.ndarray) or not isinstance(ow_tbs, np.ndarray):
            raise TypeError("Expect that cice_tbs and ow_tbs are both numpy arrays")

        if cice_tbs.ndim != 2 or ow_tbs.ndim != 2:
            raise ValueError("Expect that cice_tbs and ow_tbs have only 2 dimensions")

        if fyi_tbs is not None:
            if not isinstance(fyi_tbs, np.ndarray):
                raise TypeError("Expect that fyi_tbs (if not None) is a numpy array")
            if fyi_tbs.ndim != 2:
                raise ValueError("Expect that fyi_tbs (if not None) as 2 dimensions")
            if fyi_tbs.shape[1] != len(channels):
                fyi_tbs = fyi_tbs.T

        if myi_tbs is not None:
            if not isinstance(myi_tbs, np.ndarray):
                raise TypeError("Expect that myi_tbs (if not None) is a numpy array")
            if myi_tbs.ndim != 2:
                raise ValueError("Expect that myi_tbs (if not None) as 2 dimensions")
            if myi_tbs.shape[1] != len(channels):
                myi_tbs = myi_tbs.T

        # transpose the input Tbs if needed (we want tbs[samples,3])
        if cice_tbs.shape[1] != len(channels):
            cice_tbs = cice_tbs.T

        if ow_tbs.shape[1] != len(channels):
            ow_tbs = ow_tbs.T

        # "fix the ice line":
        if fix_the_ice_line:
            # Normalize the occurence of cice_tbs along the ice line, so that subsequent tuning of the
            #    algorithms is not skewed by amount of FYI vs MYI in the training sample.
            # -> pca to find the equation of the line
            eigenvecs_cice, eigenvals_cice, mean_cice = _pca(cice_tbs.T)
            if np.dot(eigenvecs_cice[:,0],np.ones(3)) < 0:
                eigenvecs_cice *= -1.
            # -> normalization of occurence along the line
            u = eigenvecs_cice[:,0]
            dal = np.dot(u,cice_tbs.T)
            mdal = np.dot(u,mean_cice)
            dal -= mdal
            nb = 51
            hist, bin_edges = np.histogram(dal, bins=nb, range=np.percentile(dal,(0.5,99.5,)))
            n = min(hist[hist>=int(ceil(0.5 * (float(dal.size)/(nb-1))))])
            normalized_cice_index = np.ones((n*len(hist)),dtype='int') * -1
            for ih, h in enumerate(hist):
                indx_h = np.where((dal>bin_edges[ih])*(dal<=bin_edges[ih+1]))[0]
                if len(indx_h) >= n:
                    n_index_h = conc_utils.downsample_index(len(indx_h),n)
                else:
                    n_index_h = list(range(len(indx_h)))
                normalized_cice_index[ih*n:ih*n+len(n_index_h)] = indx_h[n_index_h]

            normalized_cice_index = normalized_cice_index[normalized_cice_index>=0]
            cice_tbs = cice_tbs[normalized_cice_index,:]

        # Tune the CICE line:
        if fyi_tbs is None:
            # Use the cice_tbs to find the equation of the ice line
            eigenvecs_cice, eigenvals_cice, mean_cice = _pca(cice_tbs.T)
            # Align the direction of the ice line (MYI->FYI) with increasing Tbs (because Eice_myi < Eice_fyi)
            if np.dot(eigenvecs_cice[:,0],np.ones(3)) < 0:
                eigenvecs_cice *= -1.

            # compute the distance along the line
            u = eigenvecs_cice[:,0]
            dal = np.dot(u,cice_tbs.T)
            # set dal = 0 at the mean cice point
            mdal = np.dot(u,mean_cice)
            dal -= mdal
            # Use dal to place the FYI and MYI at percentiles of dal
            #adal = np.abs(dal)
            #dal_L = np.percentile(adal,85)
            #dal_Lmean = (adal[adal>dal_L]).mean()
            #fyi_tp = mean_cice + dal_Lmean * eigenvecs_cice[:,0]
            #myi_tp = mean_cice - dal_Lmean * eigenvecs_cice[:,0]
            dal_LM,dal_LF  = np.percentile(dal,(1.,99.))
            myi_tp = mean_cice + dal_LM * eigenvecs_cice[:,0]
            fyi_tp = mean_cice + dal_LF * eigenvecs_cice[:,0]
        else:
            # Use the fyi_tbs and myi_tbs to define the ice line
            # TODO : this is not exactly what we want. We want the ice line to be
            #    tuned by PCA, and the fyi_ and myi_TPs to be the projected averaged points
            #    on the ice line.
            fyi_tp = fyi_tbs.mean(axis=0)
            myi_tp = myi_tbs.mean(axis=0)
            # add point3 and point4 to numerically stabilize the PCA
            point3 = 0.5*(fyi_tp+myi_tp)
            point3[0] += 0.005
            point4 = 0.5*(fyi_tp+myi_tp)
            point4[1] += 0.001
            cice_tbs_icel = np.row_stack((fyi_tp,myi_tp,point3,point4))
            eigenvecs_cice, eigenvals_cice, mean_cice = _pca3d(cice_tbs_icel[:,0],cice_tbs_icel[:,1],cice_tbs_icel[:,2])
            mean_cice = 0.5*(fyi_tp+myi_tp)

        # For open water TP, take the average (but also compute the "weather line" through PCA)
        #print type(ow_tbs[:,0]), np.mean(ow_tbs,axis=0), np.min(ow_tbs,axis=0)
        eigenvecs_ow, eigenvals_ow, ow_tp = _pca(ow_tbs.T)

        (ow_ch1, ow_ch2, ow_ch3) = ow_tp
        (fyi_ch1, fyi_ch2, fyi_ch3) = fyi_tp
        (myi_ch1, myi_ch2, myi_ch3) = myi_tp

        # check the methods parameter (it must be a string )
        if not isinstance(method,(str,)):
            raise ValueError("The method must be a string (got {})".format(method,))

        # arrays used for some methods
        dal_bins = np.linspace(-1.3,1.3,num=53)
        dal_centers = (dal_bins + 0.5*(dal_bins[1]-dal_bins[0]))[:-1]
        sic_binned = np.empty_like(dal_centers)

        # define u
        u = eigenvecs_cice[:,0]
        # define v that is in the ch1xch2 plane, and perpendicular to u
        v = np.zeros_like(u)
        v[1] = (1. / (1. + (u[1]/u[0])**2))**0.5
        v[0] = -u[1]/u[0] * v[1]
        # define w that is in the ch2xch3 plane, and perpendicular to u
        w = np.zeros_like(u)
        w[2] = (1. / (1. + (u[2]/u[1])**2))**0.5
        w[1] = -u[2]/u[1] * w[2]

        # compute the parameters to the 3D->2D axis transformation (generalisation of the Bristol transform)
        if method == 'hanna':
            ### Dynamic method from Smith (1996), and Hanna and Bamber (2001)
            a1 = 1.0
            # a2 = gradient of ice line in polarization plane (change in Ch3)/(change in Ch2)
            a2 = (myi_ch3 - fyi_ch3) / (myi_ch2 - fyi_ch2)
            # a3 = gradient of ice line in frequency plane (change in Ch1)/(change in Ch2)
            a3 = (myi_ch1 - fyi_ch1) / (myi_ch2 - fyi_ch2)

            ahch2 = fyi_ch2 - ow_ch2
            b1 = 1.0
            b2 = (fyi_ch3 - ow_ch3) / ahch2
            b3 = (fyi_ch1 - ow_ch1) / ahch2

            # plane orthogonal to (fyi, myi, ow)
            c = - (a1*a1 + a2*a2 + a3*a3) / (a1*b1 + a2*b2 + a3*b3)
            ahy = -(a1+c)
            y1 = -1.0
            y2 = (a2+b2*c) / ahy
            y3 = (a3+b3*c) / ahy

            ct = ChannelTransform(channels,a3,+1.0,a2,y3,-1.0,y2)
        elif method == 'smith' or method == 'bristol':
            # This is the transform as published in Smith (1996), and used
            #    in OSISAF and SICCI1
            method = 'bristol'
            ct = ChannelTransform(channels,0.5250,+1.0,1.0450,0.9164,-1.0,0.4965)
        elif method == 'fcomiso':
            # Special case for Comiso Frequency mode
            ct = ChannelTransform(channels,0.,1.,0.,1.,0.,0.)
        elif method == 'ch1xch2':
            # exactly same SIC as fcomiso, but the ChannelTransform is aligned with the ice line
            ct = ChannelTransform(channels,
                                  u[0],u[1],u[2],
                                  v[0],v[1],v[2])
        elif method == 'pcomiso':
            # Special case for Comiso Polarization mode
            ct = ChannelTransform(channels,0.,1.,0.,0.,0.,1.)
        elif method == 'ch2xch3':
            # exactly same SIC as pcomiso, but the ChannelTransform is aligned with the ice line
            ct = ChannelTransform(channels,
                                  u[0],u[1],u[2],
                                  w[0],w[1],w[2])
        elif 'pca3d' in method:
            #    for "best_ow" and "best_cice"
            #    we must search for the optimal rotation angle for minimizing
            #    uncertainties at OW or CICE cases

            # Rotation: the axis is along the ice line, and the angle is departure
            #    from the direction of v
            qvec  = np.expand_dims(-v,1)
            qaxis = np.expand_dims(u,1)
            if 'best_ow' in method:
                what = 'ow'
                if fit_curvy_ice_line:
                    print("WARNING: fit_curvy_ice_line=True does not make sense for pca3d:best_ow. I'll now set it to False.")
                    fit_curvy_ice_line = False
            elif 'best_cice' in method:
                what = 'cice'
            else:
                raise ValueError("ERROR: Unknown PCA3D method {}".format(method))

            angles = np.linspace(-90.,+90.,180)
            best_sdev = 10000.
            ct = None
            for rot in angles:
                # create a ChannelTransform with the given rotation angle
                vdir  = qrotate(qvec,qaxis,np.deg2rad(rot))[:,0]
                transf1 = u
                transf2 = vdir
                nct = ChannelTransform(channels,transf1[0],transf1[1],transf1[2],transf2[0],transf2[1],transf2[2])
                nct.normalize(ow_tp, fyi_tp,  myi_tp)
                # use it to compute sic (and dal)
                sic0,dal0 = nct.compute_sic(eval("{}_tbs".format(what,)),with_uncert=False,with_dal=True)
                # for 'best_cice' algorithm, do extra steps before computing the standard deviation of sic0
                if fit_curvy_ice_line:
                    our_dal_centers = dal_centers.copy()
                    our_sic_binned = np.empty_like(our_dal_centers)

                    # bin the sic values by dal, and obtain a 'curvy' ice line.
                    for dal_b in range(len(dal_bins)-1):
                        m_dal = dal_bins[dal_b]
                        M_dal = dal_bins[dal_b+1]
                        dal_indx = (dal0>=m_dal)*(dal0<M_dal)
                        if dal_indx.sum() > 15:
                            our_sic_binned[dal_b] = sic0[dal_indx].mean()
                        else:
                            our_sic_binned[dal_b] = -99.

                    our_sic_binned[our_sic_binned == -99.]  = 1.

                    # correct sic for the 'curvy' ice line.
                    interpolated_sics = conc_utils.interpolate_1d_linear(our_dal_centers,our_sic_binned,dal0)
                    sic0 = sic0 / interpolated_sics.clip(0.9,1.1)

                # compute standard deviation, and record the channel transform if it improves on previous ones
                sdev = sic0.std()
                if sdev < best_sdev:
                    best_sdev = sdev
                    ct = nct

        else:
            raise ValueError("ERROR: Unknown method {}".format(method))

        # normalize the transformation
        ct.normalize(ow_tp, fyi_tp,  myi_tp)
        method_tag = get_tag(method,channels,corr)

        # compute (and store) a piecewise linear fit of the (curvy) ice line
        if fit_curvy_ice_line:
            # use the channel transform to compute sic and dal for the 100% ice samples
            sic0, dal0 = ct.compute_sic(cice_tbs,with_uncert=False,with_dal=True)
            # bin the sic values by dal, and obtain a 'curvy' ice line.
            for dal_b in range(len(dal_bins)-1):
                m_dal = dal_bins[dal_b]
                M_dal = dal_bins[dal_b+1]
                dal_indx = (dal0>=m_dal)*(dal0<M_dal)
                if dal_indx.sum() > 15:
                    sic_binned[dal_b] = (sic0[dal_indx]).mean()
                else:
                    sic_binned[dal_b] = -99.

            # remove intervals we have no data for
            sic_binned[sic_binned == -99.] = 1.

            # store the ice line
            self.dal_centers = dal_centers
            self.sic_binned  = sic_binned

        # store in the DynamicTiepoint object
        self.channels = channels
        self.corr     = getcorr(corr)
        self.ow_N     = ow_tbs.shape[0]
        self.cice_N   = cice_tbs.shape[0]

        # some parameters to the call
        self.fit_curvy_ice_line = fit_curvy_ice_line
        self.fix_the_ice_line   = fix_the_ice_line

        # the 3D values
        self.cice_eigenvals  = eigenvals_cice
        self.cice_eigenvecs  = eigenvecs_cice
        self.fyi_tp          = fyi_tp
        self.myi_tp          = myi_tp
        self.ow_eigenvals    = eigenvals_ow
        self.ow_eigenvecs    = eigenvecs_ow
        self.ow_tp           = ow_tp

        # the axes transformations
        self.transf          = ct
        # the associated departure angle from v
        ctvec = np.array((ct.d,ct.e,ct.f))
        self.angle = np.rad2deg(angle3d(ctvec,-v)) * np.sign( np.dot( np.cross(ctvec,-v), u) )

        # the average and standard deviation of ice concentration computed for OW and CICE Tbs
        ow_sic,ow_dal     = ct.compute_sic(ow_tbs,with_uncert=False,with_dal=True)
        cice_sic,cice_dal = ct.compute_sic(cice_tbs,with_uncert=False,with_dal=True)
        if fit_curvy_ice_line:
            interpolated_sics = conc_utils.interpolate_1d_linear(dal_centers,sic_binned,cice_dal)
            cice_sic = cice_sic / interpolated_sics.clip(0.9,1.1)
            interpolated_sics = conc_utils.interpolate_1d_linear(dal_centers,sic_binned,ow_dal)
            ow_sic   = ow_sic / interpolated_sics.clip(0.9,1.1)

        ow_avg         = ow_sic.mean()
        ow_std         = ow_sic.std()
        cice_avg       = cice_sic.mean()
        cice_std       = cice_sic.std()
        # the ChannelTransforms know their uncertainties for later re-use
        ct.set_uncertainties(ow_std,cice_std)

        self.bias = {'ow': ow_avg, 'cice': cice_avg}
        self.std = {'ow': ow_std, 'cice': cice_std}

    def _to_dict(self):
        """ Return a dict() version of the object """
        ret = self.__dict__
        for elem in list(ret.keys()):
            if elem == 'transf':
                if isinstance(ret[elem],ChannelTransform):
                    ret[elem] = ret[elem].__dict__
        return ret

    def to_json(self):
        """ Return a dict() version of the object, that can be dumped to json """
        ret = self._to_dict()
        for elem in list(ret.keys()):
            if isinstance(ret[elem],np.ndarray):
                ret[elem] = ret[elem].tolist()
            elif isinstance(ret[elem],dict):
                for k in list(ret[elem].keys()):
                    if type(ret[elem][k]).__module__ == np.__name__:
                        ret[elem][k] = np.asscalar(ret[elem][k])
        return ret


    @classmethod
    def from_json(self,json_obj):
        """ create an object from a representation in a JSON object """
        # modify the content of the json_obj to match what
        #    we want in the object
        raise NotImplementedError("Sorry, this is on my TODO list")
        adict = dict()
        for elem in list(json_obj.keys()):
            pass

        return self(json_obj)

    def __str__(self):
        return self._to_dict().__str__()


class DynSeaIceConcAlgo(object):
    """ Class to define Sea Ice Concentration algorithms defined with
       dynamic tie-points and PCA3D ChannelTransform
    """
    def __init__(self, cts, blend_func=None, cils=None):
        """
           cils = curvy_ice_lines. as many as cts.
        """

        # Some sanity checks
        if len(cts) > 1 and blend_func is None:
            raise ValueError("You must provide a blend_func when creating an DynIceConcAlgo from more than 1 ChannelTransform")

        if cils is not None:
            if len(cils) != len(cts):
                raise ValueError("The cils (curvy_ice_lines) parameter must be as many as cts (channel_transforms)")
            for cil in cils:
                if cil is not None:
                    if len(cil) != 2:
                        raise ValueError("The cils (curvy_ice_lines) must all be a tuple of length 2 (xs,ys) if provided")
                    if len(cil[0]) != len(cil[1]):
                        raise ValueError("The xs and ys of the cils (curvy ice lines) must have the same length")

        for ct in cts:
            if not ct.normalized:
                raise ValueError("Cannot create a DynIceConcAlgo from a ChannelTransform which is not normalized")
            if not ct.ow_sdev or not ct.cice_sdev:
                raise ValueError("Cannot create a DynIceConcAlgo from a ChannelTransform that misses the uncertainties")

        self.cts   = cts
        self.blend = blend_func
        self.cils  = cils

    def __str__(self):
        ret = ''
        for i, c in enumerate(self.cts):
            ret += '\t{}\n'.format(self.cts[i].__str__())
            if self.cils is not None and self.cils[i] is not None:
                ret += '\t{}\n'.format("With curvy ice line ({},{})".format(self.cils[i][0],self.cils[i][1]))
        return ret

    def compute_sic(self, tbs, with_dal = False):
        """
           Compute Sea Ice Concentration for the channels in dict() tbs
           All tbs must have the same shape.
        """

        if self.cts is None:
            raise ValueError("This DynSeaIceAlgo object is not initialized (its cts are None)")

        # Only check algorithm's required channels for matching shape.
        shapes = [tbs[tb].shape for tb in self.cts[0].channels]
        if ( not all([x==shapes[0] for x in shapes]) ):
            raise ValueError("The provided tbs do not have identical shapes")

        # compute and store the SIC from the ChannelTransforms that constitute the blended algorithm
        sics = list(self.cts)
        devs = list(self.cts)
        dals = list(self.cts)
        for ict, ct in enumerate(self.cts):
            chns = ct.channels
            try:
                # compute sic from the channel transform
                ret = ct.compute_sic(tbs[chns[0]], tbs[chns[1]], tbs[chns[2]], with_uncert=True, with_dal=with_dal)
                if not with_dal:
                    sics[ict], devs[ict] = ret
                    dals[ict] = np.zeros_like(sics[ict])
                    if isinstance(sics[ict],ma.core.MaskedArray):
                        dals[ict] = ma.array(dals[ict],mask=sics[ict].mask)
                else:
                    sics[ict], devs[ict], dals[ict] = ret

                # apply curvy ice line correction
                if self.cils is not None and self.cils[ict] is not None:
                    interpolated_sics = conc_utils.interpolate_1d_linear(self.cils[ict][0],self.cils[ict][1],dals[ict])
                    sics[ict] = sics[ict] / interpolated_sics.clip(0.9,1.1)
            except KeyError as k:
                raise ValueError("Tb channel {} is missing from input dict() for computing SIC".format(k,))

        # combine sic and dev using the blend function
        #    also combine the dal with the same blending (could be refined)
        if len(self.cts) > 1:
            blend_sic, blend_sdev, blend_dal = self.blend(sics, devs, dals=dals)
        else:
            blend_sic, blend_sdev, blend_dal = sics[0], devs[0], dals[0]

        ret = (blend_sic, blend_sdev)
        if with_dal:
            ret = (blend_sic, blend_sdev, blend_dal)

        return ret

    @classmethod
    def from_json(cls, cjson, ct_specs, blend_func=None, corr='ucorr',):
        if isinstance(cjson,str):
            cjson = json.loads(cjson)

        cts = list(range(len(ct_specs)))
        cils = list(range(len(ct_specs)))
        for i, ctn in enumerate(ct_specs,):
            # load the ChannelTransforms
            cts[i] = ChannelTransform.from_json(ctn['id'],ctn['channels'],corr,cjson)
            # load the curvy ice line (if defined for this algorithm)
            if ctn['fit_curvy_ice_line']:
                try:
                    tag = get_tag(ctn['id'],ctn['channels'],corr)
                    curvy_ice_line_x = cjson[tag]['dal_centers']
                    curvy_ice_line_y = cjson[tag]['sic_binned']
                    cils[i] = (curvy_ice_line_x,curvy_ice_line_y)
                except KeyError as k:
                    if (corr == 'ucorr' ):
                        cils[i] = None
                    else:
                        raise ValueError("Cannot load curvy ice line for {} {} (missing {})".format(ctn['id'],ctn['channels'],k,))
            else:
                cils[i] = None

        return cls(cts, blend_func=blend_func, cils=cils)

class DynIceConcAlgoFactory(object):

    @staticmethod
    def get_ct_specs(algo_n):
        if algo_n == 'comiso':
            cts = [{'id': 'fcomiso', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'pcomiso', 'channels': ['tb19v','tb37v','tb37h']}]
            blendf = ica.comiso_blend
        elif algo_n == 'osisaf':
            cts = [{'id': 'fcomiso', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'bristol', 'channels': ['tb19v','tb37v','tb37h']}]
            blendf = ica.osisaf_blend
        elif algo_n == 'sicci1':
            cts = [{'id': 'fcomiso', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'bristol', 'channels': ['tb19v','tb37v','tb37h']}]
            blendf = ica.sicci1_blend
        elif algo_n == 'sicci2lfsil':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb19v','tb37v','tb37h']}]
            blendf = ica.sicci1_blend
        elif algo_n == 'sicci2lf':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb19v','tb37v','tb37h'],'fit_curvy_ice_line': True}]
            blendf = ica.sicci1_blend
        elif algo_n == 'sicci2hfsil':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb19v','tb90v','tb90h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb19v','tb90v','tb90h']}]
            blendf = ica.sicci1_blend
        elif algo_n == 'sicci2hf':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb19v','tb90v','tb90h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb19v','tb90v','tb90h'],'fit_curvy_ice_line': True}]
            blendf = ica.sicci1_blend
        elif algo_n == 'n90linsil':
            cts = [{'id': 'ch1xch2', 'channels': ['tb90v','tb90h','tb90h']},]
            blendf = None
        elif algo_n == 'sicci2vlfsil':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb06v','tb37v','tb37h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb06v','tb37v','tb37h']}]
            blendf = ica.sicci1_blend
        elif algo_n == 'sicci2vlf':
            cts = [{'id': 'pca3d:best_ow', 'channels': ['tb06v','tb37v','tb37h']},
                   {'id': 'pca3d:best_cice', 'channels': ['tb06v','tb37v','tb37h'],'fit_curvy_ice_line': True}]
            blendf = ica.sicci1_blend
        #elif algo_n == 'n90lin':
        #    cts = [{'id': 'ch1xch2', 'channels': ['tb90v','tb90h','tb90h'],'fit_curvy_ice_line': True}]
        #    blendf = None
        elif algo_n == 'n90lin':
            cts = [{'id': 'ch1xch2', 'channels': ['tb90v','tb90h'],'fit_curvy_ice_line': True}]
            blendf = None
        elif algo_n == 'sicci3lf':
            cts = [{'id': 'nD:best_ow', 'channels': ['tb19v','tb37v','tb37h']},
                   {'id': 'nD:best_cice', 'channels': ['tb19v','tb37v','tb37h'],
                    'fit_curvy_ice_line': True}]
            blendf = None
        elif algo_n == 'sicci3af':
            cts = [{'id': 'nD:best_ow', 'channels': ['tb19v','tb37v','tb37h','tb90v','tb90h']},
                   {'id': 'nD:best_cice', 'channels': ['tb19v','tb37v','tb37h','tb90v','tb90h'],
                    'fit_curvy_ice_line': True}]
            blendf = None
        elif algo_n == 'sicci3k4':
            cts = [{'id': 'nD:best_ow', 'channels': ['tb19v','tb19h','tb37v','tb37h']},
                   {'id': 'nD:best_cice', 'channels': ['tb19v','tb19h','tb37v','tb37h'],
                    'fit_curvy_ice_line': True}]
            blendf = None
        elif algo_n == 'sicci3hf':
            cts = [{'id': 'nD:best_ow', 'channels': ['tb19v','tb90v','tb90h']},
                   {'id': 'nD:best_cice', 'channels': ['tb19v','tb90v','tb90h'],
                    'fit_curvy_ice_line': True}]
            blendf = None
        elif algo_n == 'sicci3vlf':
            cts = [{'id': 'nD:best_ow', 'channels': ['tb06v','tb37v','tb37h']},
                   {'id': 'nD:best_cice', 'channels': ['tb06v','tb37v','tb37h'],
                    'fit_curvy_ice_line': True}]
            blendf = None
        else:
            raise NotImplementedError("Algo {} is not known".format(algo_n,))

        # fit_curvy_ice_line to False unless otherwise stated in the definition above
        for ct in cts:
            try:
                fcil = ct['fit_curvy_ice_line']
            except KeyError:
                ct['fit_curvy_ice_line'] = False

        return cts, blendf

    @staticmethod
    def load_algo_from_json(algo_n, corr, cjson):
        cts, blendf = DynIceConcAlgoFactory.get_ct_specs(algo_n)
        return DynSeaIceConcAlgo.from_json(cjson, cts, blend_func=blendf, corr=corr)

class DynamicWF(object):
    """ Small container for the dynamically tuned Weather Filters """
    def __init__(self, gr3719v_threshold, sic_threshold):
        self.gr3719v_threshold = gr3719v_threshold
        self.sic_threshold     = sic_threshold

    @classmethod
    def from_json(wf, cjson):
        """
           Constructor (hence the @classmethod) from json

           Can be called as wf = DynamicWF.from_json(cjson)

           'cjson' can either be the json dictionnary or the content of a json file
           as a string.
        """
        # Parse the json string, and search for the name of the transform (with the right channels)
        if isinstance(cjson,str):
            cjson = json.loads(cjson)

        # construct
        return wf(cjson['wf']['gr3719v_threshold'],cjson['wf']['sic_threshold'])

    def to_json(self):
        """ Return a dict() version of the object, that can be dumped to json """
        return self._to_dict()

def calc_tiepts_get_samples(swath_file_list, corr, dt, what=['tb19v', 'tb37v', 'tb37h'],
    samples=None, force_indexarray=False, gridded_tp_path=None, nasa95=False):
    """
        Return selected tie-point samples (Tbs, lat, and lons) for up to 4 classes (ow, cice, MY/FY_cice),
            from a list of swath files.
    """

    # vars_get are vars needed for indexarray generation
    vars_get = ['lons', 'lats', 'ct_NASA', 'cmfraq_NASA', 'wf_NASA']
    # vars_samples are to be saved to sample file along with vars from what input argument.
    vars_samples = ['lons', 'lats', 'ct_NASA', 'cmfraq_NASA']
    read90 = False
    if ( len(corr.split('_')) == 2 ):
        a1, a2 = corr.split('_')
        if ( a1 in HF_ALGOS ):
            vars_samples = ['lons', 'lats']
            read90 = True
    else:
        if ( corr.lower() in HF_ALGOS ):
            # Skip ct_NASA samples for second iteration highfreq algos
            vars_samples = ['lons', 'lats']
            read90 = True

    # Create samples dictionary that will hold the channel data
    # along with lat/lon + ct_NASA for the TPs.
    samples_dict = {'nh': dict(), 'sh': dict()}

    # Create lists for the required variables.
    for var in vars_samples:
        for hs in ['nh', 'sh']:
            samples_dict[hs]['{}_ow_list'.format(var)] = []
            samples_dict[hs]['{}_ice_list'.format(var)] = []
            samples_dict[hs]['{}_my_ice_list'.format(var)] = []
            samples_dict[hs]['{}_fy_ice_list'.format(var)] = []

    # Create lists for the input variables.
    for var in what:
        if (var not in vars_get):
            vars_get.append(var)
            vars_samples.append(var)
            for hs in ['nh', 'sh']:
                samples_dict[hs][var + '_ice_list'] = []
                samples_dict[hs][var + '_ow_list'] = []
                samples_dict[hs][var + '_my_ice_list'] = []
                samples_dict[hs][var + '_fy_ice_list'] = []

    # Create small dictionary for seaice/openwater mask to avoid reading for every swath file.
    maskdict = {'seaice': {'nh': dict(), 'sh': dict()},
        'openwater': {'nh': dict(), 'sh': dict()}}

    # Tiny dict for writing to and reading from indexarray.
    tmp_add = {'nh':0, 'sh':4}

    if ( corr == None ):
        corr = 'ucorr'

    # Read from swath files, adding date from each to list.
    for nr, swath in enumerate(swath_file_list):
        LOG.info("Adding file %s" % swath)

        # Dictionary for storing data from swath file.
        tmpdict = dict()

        # Read channel data + lat/lon/ct_NASA/wf_NASA from swath file.
        try:
            for var in vars_get:
                if ( var.startswith('tb') ):
                    vp = var.split('_')
                    if ( read90 ):
                        at = '@tb90'
                    else:
                        at = '@tb37'
                    if ( var.startswith('tb19') ):
                        if ( len(vp) == 4 and vp[1] == 'lc' ):
                            # tb on form tbxxx_lc_algo_corralgo -> tbxxx_lc@yy_algo_corralgo
                            tmpdict[var] = swath.read('{}_lc{}_{}_{}'.format(vp[0], at, vp[2], vp[3]))
                        elif ( len(vp) == 3 ):
                            # tb on form tbxxx_algo_corralgo -> tbxxx@yy_algo_corralgo
                            tmpdict[var] = swath.read('{}{}_{}_{}'.format(vp[0], at, vp[1], vp[2]))
                        else:
                            tmpdict[var] = swath.read('{}{}'.format(var, at))
                    elif ( var.startswith('tb37') ):
                        if ( read90 ):
                            if ( len(vp) == 4 and vp[1] == 'lc' ):
                                tmpdict[var] = swath.read('{}_lc{}_{}_{}'.format(vp[0], at, vp[2], vp[3]))
                            elif ( len(vp) == 3 ):
                                tmpdict[var] = swath.read('{}{}_{}_{}'.format(vp[0], at, vp[1], vp[2]))
                            else:
                                tmpdict[var] = swath.read('{}{}'.format(var, at))
                        else:
                            tmpdict[var] = swath.read('{}'.format(var))
                    elif ( var.startswith('tb06') or var.startswith('tb10') or var.startswith('tb22') ):
                        # no @tb37/@tb90 for tb06/tb10/tb22 for now
                        tmpdict[var] = swath.read('{}'.format(var))
                    elif ( var.startswith('tb90') ):
                        if (  at == '@tb37' ):
                            if ( len(vp) == 4 and vp[1] == 'lc' ):
                                # tb on form tbxxx_lc_algo_corralgo -> tbxxx_lc@yy_algo_corralgo
                                tmpdict[var] = swath.read('{}_lc{}_{}_{}'.format(vp[0], at, vp[2], vp[3]))
                            elif ( len(vp) == 3 ):
                                # tb on form tbxxx_algo_corralgo -> tbxxx@yy_algo_corralgo
                                tmpdict[var] = swath.read('{}{}_{}_{}'.format(vp[0], at, vp[1], vp[2]))
                            else:
                                tmpdict[var] = swath.read('{}{}'.format(var, at))
                        else:
                            if ( len(vp) == 4 and vp[1] == 'lc' ):
                                # tb on form tbxxx_lc_algo_corralgo -> tbxxx_lc@yy_algo_corralgo
                                tmpdict[var] = swath.read('{}_lc_{}_{}'.format(vp[0], vp[2], vp[3]))
                            elif ( len(vp) == 3 ):
                                # tb on form tbxxx_algo_corralgo -> tbxxx@yy_algo_corralgo
                                tmpdict[var] = swath.read('{}_{}_{}'.format(vp[0], vp[1], vp[2]))
                            else:
                                tmpdict[var] = swath.read('{}'.format(var))
                    else:
                        tmpdict[var] = swath.read('{}{}'.format(var, at))
                elif ( var in ('lons', 'lats') ):
                    if ( read90 ):
                        try:
                            tmpdict[var] = swath.read('{}90'.format(var[:3]))
                        except KeyError as e:
                            tmpdict[var] = swath.read('{}_h'.format(var))
                    else:
                        try:
                            tmpdict[var] = swath.read('{}37'.format(var[:3]))
                        except KeyError as e:
                            tmpdict[var] = swath.read('{}'.format(var))
                elif ( var in ['air_temp', 'wind_speed', 'tcwv', 'tclw', 'skt'] ):
                    if ( read90 ):
                        tmpdict[var] = swath.read(var + '@tb90')
                    else:
                        tmpdict[var] = swath.read(var + '@tb37')
                else:
                    tmpdict[var] = swath.read('{}'.format(var))
        except KeyError as e:
            raise ValueError('KeyError: {} doesn\'t exist in {}'.format(e, swath))

        # Check here if swath file already has indices of ow/ice TPs for nh/sh.
        try:
            if ( read90 ):
                indexarray = swath.read('indexarray_h')
                if force_indexarray:
                    raise KeyError('indexarray_h')
            else:
                indexarray = swath.read('indexarray')
                if force_indexarray:
                    raise KeyError('indexarray')
        except KeyError:
            # No indexarray variable in swath file so go through pyresample
            # and geo_mask to create indexarray and write it swath file.

            # Create array for tp indices for writing to swath file.
            indexarray = np.zeros((tmpdict['lons'].shape)).astype(np.int8)
            """
                indexarray values:
                    0:  unselected for tp
                    1:  nh ice tp
                    2:  nh ow tp
                    3:  nh multi-year ice tp
                    4:  nh first-year ice tp
                    5:  sh ice tp
                    6:  sh ow tp
                    7:  sh multi-year ice tp
                    8:  sh first-year ice tp
            """
            indexarray_values = '0: unselected for tp, 1: nh ice tp, 2: nh ow tp, ' + \
                '3: nh multi-year ice tp, 4: nh first-year ice tp, 5: sh ice tp, ' + \
                '6: sh ow tp, 7: sh multi-year ice tp, 8: sh first-year ice tp'

            swath_def = pr.geometry.SwathDefinition(tmpdict['lons'], tmpdict['lats'])

            # if highres then resample ct_NASA, wf_NASA, cmfraq_NASA to tb90
            if ( read90 ):
                swath_def_37 = pr.geometry.SwathDefinition(swath.read('lons'), swath.read('lats'))
                radius_37 = 25000
                in_data = np.concatenate((tmpdict['ct_NASA'][:,:,np.newaxis],
                    tmpdict['wf_NASA'][:,:,np.newaxis],
                    tmpdict['cmfraq_NASA'][:,:,np.newaxis]), axis=2)
                valid_input_index, valid_output_index, index_array, distance_array = \
                    pr.kd_tree.get_neighbour_info(swath_def_37, swath_def,
                        radius_of_influence=radius_37, neighbours=1,
                        epsilon=0, reduce_data=True, nprocs=1, segments=None)
                output_data = pr.kd_tree.get_sample_from_neighbour_info('nn', swath_def.shape,
                    in_data, valid_input_index, valid_output_index, index_array,
                    distance_array=distance_array, weight_funcs=None,
                    fill_value=None, with_uncert=False)
                tmpdict['ct_NASA'] = output_data[:, :, 0]
                tmpdict['wf_NASA'] = output_data[:, :, 1]
                tmpdict['cmfraq_NASA'] = output_data[:, :, 2]                

            # Go through hemispheres, creating a sea/ow mask for each and
            # adding resulting indices to indexarray.
            for hs in ['nh', 'sh']:

                try:
                    seaice_mask = maskdict['seaice'][hs][dt[nr].month - 1]
                except KeyError:
                    if ( gridded_tp_path != None ):
                        if ( nasa95 ):
                            seaice_mask = geo_mask.MaskFactory.create_mask('ice_new', hs, dt[nr], path=gridded_tp_path, nasa95=True)
                        else:
                            seaice_mask = geo_mask.MaskFactory.create_mask('ice_new', hs, dt[nr], path=gridded_tp_path)
                    else:
                        seaice_mask = geo_mask.MaskFactory.create_mask('ice', hs, dt[nr])
                    maskdict['seaice'][hs][dt[nr].month - 1] = seaice_mask
                try:
                    open_water_mask = maskdict['openwater'][hs][dt[nr].month - 1]
                except KeyError:
                    if ( gridded_tp_path != None ):
                        open_water_mask = geo_mask.MaskFactory.create_mask('ow_new', hs, dt[nr], path=gridded_tp_path)
                    else:
                        open_water_mask = geo_mask.MaskFactory.create_mask('dyn_tp_water_met', hs, dt[nr])
                    maskdict['openwater'][hs][dt[nr].month - 1] = open_water_mask

                LOG.info("Calculating tiepoints using seaice mask %s and open_water_mask %s" \
                    %(seaice_mask, open_water_mask))

                # seaice_mask.get_valid_index(swath_def) gives bool array same
                # size as the swath data. Get subset of swath data using this.

                # Consolidated ice
                a = seaice_mask.get_valid_index(swath_def)
                if ( nasa95 ):
                    idx_nasa_above_95 = (tmpdict['wf_NASA'] == 0.0)
                else:
                    idx_nasa_above_95 = ma.logical_and(tmpdict['ct_NASA'] > 95.0, tmpdict['wf_NASA'] == 0.0)
                b = np.where(ma.logical_and(a, idx_nasa_above_95))
                indexarray[b] = tmp_add[hs] + 1

                # MY & FY below will replace consolidated ice where they exist.

                # Multi-year ice
                idx_nasa_my_ice = ma.logical_and(idx_nasa_above_95, tmpdict['cmfraq_NASA'] > 90.0)
                b = np.where(ma.logical_and(a, idx_nasa_my_ice))
                indexarray[b] = tmp_add[hs] + 3

                # First-year ice
                idx_nasa_fy_ice = ma.logical_and(idx_nasa_above_95, tmpdict['cmfraq_NASA'] < 10.0)
                b = np.where(ma.logical_and(a, idx_nasa_fy_ice))
                indexarray[b] = tmp_add[hs] + 4

                # Open water
                a = open_water_mask.get_valid_index(swath_def)
                # In cases where tb22v is completly missing (some SMMR period), the
                #    WF can still be used, although it was computed only on gr3718v.
                idx_wf_NASA_ow = (tmpdict['wf_NASA'] == 1.0)
                if ( hs == 'nh' ):
                    idx_wf_NASA_ow = ma.logical_and(idx_wf_NASA_ow,
                        tmpdict['lats'] > 50.0)
                b = np.where(ma.logical_and(a, idx_wf_NASA_ow))
                indexarray[b] = tmp_add[hs] + 2

            # Write indexarray to swath file.
            try:
                indexarray[indexarray.mask] = 0
                indexarray = indexarray.data
                print("WARNING... indexarray was a MaskedArray! We set its masked values to 0 and continue")
            except AttributeError:
                pass
            if ( read90 ):
                swath.write('indexarray_h', indexarray)
                swath.writeattr('indexarray_h', 'values', indexarray_values)
            else:
                swath.write('indexarray', indexarray)
                swath.writeattr('indexarray', 'values', indexarray_values)
        
        # Some of the variables we want to return might be masked. We want to compute the
        #   largest mask and apply is to all variables
        # EXCEPT fully masked tb22v for SMMR
        common_mask = np.zeros(indexarray.shape).astype('bool')
        for var in vars_samples:
            if ( os.path.basename(swath.__str__()).startswith('smmr') and var == 'tb22v' ):
                continue
            if isinstance(tmpdict[var], ma.MaskedArray):
                try:
                    common_mask = np.logical_or(common_mask, tmpdict[var].mask)
                except ValueError as e:
                    print('generate_daily_dyn_tp: {} in {} has the wrong shape [{}], should be [{}]'.format(var,
                        swath, tmpdict[var].mask.shape, common_mask.shape))
                    raise ValueError

        # apply the common mask
        for var in vars_samples:
            if isinstance(tmpdict[var], ma.MaskedArray):
                tmpdict[var] = ma.array(tmpdict[var].data, mask=common_mask)
            else:
                tmpdict[var] = ma.array(tmpdict[var], mask=common_mask)

        # Use indexarray to add separate nh/sh ow/ice TPs to samples_dict.
        # Using .compressed() to make sure that masked values are gone
        for hs in ['nh', 'sh']:
            # Ice samples should be MY + FY + the rest
            ice_index = np.logical_or(indexarray == (tmp_add[hs]+3), indexarray == (tmp_add[hs]+4))
            ice_index = np.logical_or(ice_index, indexarray == (tmp_add[hs]+1))
            ow_index = (indexarray == (tmp_add[hs]+2))
            my_ice_index = (indexarray == (tmp_add[hs]+3))
            fy_ice_index = (indexarray == (tmp_add[hs]+4))
            for i,var in enumerate(vars_samples):
                samples_dict[hs][var + '_ice_list'].append((tmpdict[var][ice_index]).compressed())
                samples_dict[hs][var + '_ow_list'].append((tmpdict[var][ow_index]).compressed())
                samples_dict[hs][var + '_my_ice_list'].append((tmpdict[var][my_ice_index]).compressed())
                samples_dict[hs][var + '_fy_ice_list'].append((tmpdict[var][fy_ice_index]).compressed())


    # Concatenate values from each file into '_ow' and '_ice' variables and add
    # variable names to a ow_ice_list for creating dict for netcdf later.
    for hs in ['nh', 'sh']:
        for var in vars_samples:
            samples_dict[hs][var + '_ow'] = np.concatenate(samples_dict[hs][var + '_ow_list'])
            del samples_dict[hs][var + '_ow_list']
            samples_dict[hs][var + '_ice'] = np.concatenate(samples_dict[hs][var + '_ice_list'])
            del samples_dict[hs][var + '_ice_list']
            samples_dict[hs][var + '_my_ice'] = np.concatenate(samples_dict[hs][var + '_my_ice_list'])
            del samples_dict[hs][var + '_my_ice_list']
            samples_dict[hs][var + '_fy_ice'] = np.concatenate(samples_dict[hs][var + '_fy_ice_list'])
            del samples_dict[hs][var + '_fy_ice_list']

    # try to reduce the size of the output samples by
    # only storing every 'n' values
    if samples is not None:
        for hs in ['nh', 'sh']:
            ice_index = conc_utils.downsample_index(samples_dict[hs]['lons_ice'].size, samples)
            ow_index = conc_utils.downsample_index(samples_dict[hs]['lons_ow'].size, samples)
            my_index = conc_utils.downsample_index(samples_dict[hs]['lons_my_ice'].size, samples)
            fy_index = conc_utils.downsample_index(samples_dict[hs]['lons_fy_ice'].size, samples)
            for var in vars_samples:
                samples_dict[hs][var + '_ice'] = samples_dict[hs][var + '_ice'][ice_index]
                samples_dict[hs][var + '_ow'] = samples_dict[hs][var + '_ow'][ow_index]
                samples_dict[hs][var + '_my_ice'] = samples_dict[hs][var + '_my_ice'][my_index]
                samples_dict[hs][var + '_fy_ice'] = samples_dict[hs][var + '_fy_ice'][fy_index]

    # If we used tbxxx_OSISAF_ucorr, tbxxx_corrNASA,... etc as arguments then only retain the tbxxx part.
    for hs in ['nh', 'sh']:
        for key in list(samples_dict[hs].keys()):
            keyparts = key.split('_')
            if ( key.startswith('tb') and len(keyparts) >= 3 ):
                if ( keyparts[-2] == 'my' or keyparts[-2] == 'fy' ):
                    samples_dict[hs][keyparts[0] + '_' + keyparts[-2] + '_' + keyparts[-1]] = samples_dict[hs].pop(key)
                else:
                    samples_dict[hs][keyparts[0] + '_' + keyparts[-1]] = samples_dict[hs].pop(key)

    return samples_dict

