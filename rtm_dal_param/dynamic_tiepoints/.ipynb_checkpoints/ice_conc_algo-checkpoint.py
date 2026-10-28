
"""
    Collection of ice concentation algorithms.

    Main entrance into this module is the calc_osi_conc and calc_nasa_conc functions.

"""

import logging
import numpy as np
import numpy.ma as ma

LOG = logging.getLogger(__name__)

def gr(tbA,tbB):
    """ Gradient Ratio
        gr3718v = gr(18v,37v)
        gr2218v = gr(18v,22v)
    """
    return (tbB - tbA) / (tbB + tbA)

def WF_Cavalieri(tb18v, tb22v, tb37v, gr3718v_threshold=0.05, gr2218v_threshold=0.045):
    """
        Weather filter from Cavalieri et al. (1995)

        The thresholds in gr3718v and gr2218v can be provided as keyword arguments
        (gr3718v_threshold=0.05, gr2218v_threshold=0.045).

        If gr2218v_threshold is None, the gr2218v test is not applied and the WF is
        computed only on the gr3718v test.

        :param tb18v: Brightness temperatues 18 Ghz V polarized
        :type tb18v: numpy array
        :param tb22v: Brightness temperatues 22 Ghz V polarized
        :type tb22v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
        :type tb37v: numpy array
        :returns: Boolean array: True if masked by WF
    """
    gr3718v = gr(tb18v,tb37v)
    wf   = gr3718v > gr3718v_threshold
    if gr2218v_threshold is not None:
        gr2218v = gr(tb18v,tb22v)
        wf   = np.logical_or(wf, (gr2218v > gr2218v_threshold))
    return wf

def nasa(tb18v, tb18h, tb37v, nasa_tp, area='nh'):
    """
        NASA-Team ice concentration algorithm

        Static tiepoint algorithm.

        :param tb18v: Brightness temperatues 18 Ghz V polarized
        :type tb18v: numpy array
        :param tb18h: Brightness temperatues 18 Ghz H polarized
        :type tb18h: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
        :type tb37v: numpy array
        :param nasa_tp: NH/SH NASA tie-points
        :type nasa_tp: dictionary
        :param area: Area, Northern 'nh' or Southern 'sh' hemisphere
        :type area: string
        :returns: Ice concentation

    """

    LOG.info("Calculating NASA Team Ice conc for area %s" % area)

    try:
        tiepts = nasa_tp[area]
    except KeyError as TypeError:
        print('tiepts not a dictionary or key {} not present.\nUsing old tie-points.'.format(area))
        if area.lower() == 'nh':
            ##tiepts = (177.10, 100.80, 201.7, 258.20, 242.80, 252.80, 223.20, 203.90, 186.30)
            # New TPs for non-atmospheric corrected TBs (from PVASR/RRDP work).
            tiepts = (185.04, 117.16, 208.72, 252.79, 238.20, 244.68, 223.64, 206.46, 190.14)
        elif area.lower() == 'sh':
            ##tiepts = (176.60, 100.30, 200.5, 249.80, 237.80, 243.3, 221.60, 193.70, 190.30)
            # New TPs for non-atmospheric corrected TBs (from PVASR/RRDP work).
            tiepts = (185.02, 118.00, 209.59, 259.92, 244.57, 254.39, 246.27, 221.95, 226.46)
        else:
            raise ValueError('NASA Team undefined area: %s', area)

    (tb18v_ow, tb18h_ow, tb37v_ow, tb18v_fy, tb18h_fy,
     tb37v_fy, tb18v_my, tb18h_my, tb37v_my) = tiepts

    a0 = - tb18v_ow + tb18h_ow
    a1 =   tb18v_ow + tb18h_ow
    a2 =   tb18v_my - tb18h_my - tb18v_ow + tb18h_ow
    a3 = - tb18v_my - tb18h_my + tb18v_ow + tb18h_ow
    a4 =   tb18v_fy - tb18h_fy - tb18v_ow + tb18h_ow
    a5 = - tb18v_fy - tb18h_fy + tb18v_ow + tb18h_ow

    b0 = - tb37v_ow + tb18v_ow
    b1 =   tb37v_ow + tb18v_ow
    b2 =   tb37v_my - tb18v_my - tb37v_ow + tb18v_ow
    b3 = - tb37v_my - tb18v_my + tb37v_ow + tb18v_ow
    b4 =   tb37v_fy - tb18v_fy - tb37v_ow + tb18v_ow
    b5 = - tb37v_fy - tb18v_fy + tb37v_ow + tb18v_ow

    gr = (tb37v - tb18v) / (tb37v + tb18v)
    pr = (tb18v - tb18h) / (tb18v + tb18h)

    d0 = (-a2 * b4) + (a4 * b2)
    d1 = (-a3 * b4) + (a5 * b2)
    d2 = (-a2 * b5) + (a4 * b3)
    d3 = (-a3 * b5) + (a5 * b3)

    dd = d0 + (d1 * pr) + (d2 * gr) + (d3 * pr * gr)

    f0 = (a0 * b2) - (a2 * b0)
    f1 = (a1 * b2) - (a3 * b0)
    f2 = (a0 * b3) - (a2 * b1)
    f3 = (a1 * b3) - (a3 * b1)
    m0 = (-a0 * b4) + (a4 * b0)
    m1 = (-a1 * b4) + (a5 * b0)
    m2 = (-a0 * b5) + (a4 * b1)
    m3 = (-a1 * b5) + (a5 * b1)

    cf = (f0 + (f1 * pr) + (f2 * gr) + (f3 * pr * gr)) / dd
    cm = (m0 + (m1 * pr) + (m2 * gr) + (m3 * pr * gr)) / dd

    cf = cf
    cm = cm
    ct = cm + cf
    return ct, cm

def nasa_with_WF(tb18v, tb18h, tb22v, tb37v, nasa_tp, area='nh'):
    """
        NASA-Team ice concentration algorithm with WF screening

        Static tiepoint algorithm.

        :param tb18v: Brightness temperatues 18 Ghz V polarized
        :type tb18v: numpy array
        :param tb18h: Brightness temperatues 18 Ghz H polarized
        :type tb18h: numpy array
        :param tb22v: Brightness temperatues 22 Ghz V polarized
        :type tb22v: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
        :type tb37v: numpy array
        :param nasa_tp: NH/SH NASA tie-points
        :type nasa_tp: dictionary
        :param area: Area, Northern 'nh' or Southern 'sh' hemisphere
        :type area: string
        :returns: Ice concentation

    """
    ct_NASA, cm_NASA = nasa(tb18v, tb18h, tb37v, nasa_tp, area=area)
    wf_NASA = WF_Cavalieri(tb18v, tb22v, tb37v)
    ct_NASA[wf_NASA] = 0.0
    cm_NASA[wf_NASA] = 0.0
    return ct_NASA, cm_NASA

def calc_nasa_conc(lats, tb19v, tb19h, tb37v, nasa_tp, wf_tb22v=None ):
    """

        Calculates the ice concentation using the NASA Team algorithm.
           Can process both Weater-Filtered and non-weather-filtered versions.
           Default is to not apply Weather Filters

        :param lats: The lalitudes of the pixels
        :type lats: numpy array
        :param tb18v: Brightness temperatues 18 Ghz V polarized
        :type tb18v: numpy array
        :param tb18h: Brightness temperatues 18 Ghz H polarized
        :type tb18h: numpy array
        :param tb37v: Brightness temperatues 37 Ghz V polarized
        :type tb37v: numpy array
        :param nasa_tp: NH/SH NASA tie-points
        :type nasa_tp: dictionary
        :param wf_tb22v: If not None, Weather Filter version
                            of NT algoritm will be applied, using tb22v channel
        :type wf_tb22v: None | numpy array
        :returns: The ice concentration as a numpy array

    """

    nh_idx = (lats > 0)
    sh_idx = (lats <= 0)
    LOG.debug("Swath has %d FoVs in NH and %d in SH" % (nh_idx.sum(),sh_idx.sum()))

    if wf_tb22v is None:
        LOG.debug("Use NT without Weather Filter")
        nasa_nh, cm_nh = nasa(tb19v[nh_idx], tb19h[nh_idx], tb37v[nh_idx], nasa_tp, area='nh')
        nasa_sh, cm_sh = nasa(tb19v[sh_idx], tb19h[sh_idx], tb37v[sh_idx], nasa_tp, area='sh')
    else:
        # WARNING: if tb22v is fully masked (e.g. periods of SMMR data) then the WF will
        #    mask all. It is then advised to call the WF separately with the gr2218v_threshold=None
        LOG.debug("Use NT with full Weather Filter (gr2219v and gr3719v)")
        nasa_nh, cm_nh = nasa_with_WF(tb19v[nh_idx], tb19h[nh_idx], wf_tb22v[nh_idx],
            tb37v[nh_idx], nasa_tp, area='nh')
        nasa_sh, cm_sh = nasa_with_WF(tb19v[sh_idx], tb19h[sh_idx], wf_tb22v[sh_idx],
            tb37v[sh_idx], nasa_tp, area='sh')

    conc = ma.zeros(lats.shape)
    conc[nh_idx] = nasa_nh
    conc[sh_idx] = nasa_sh

    cm = ma.zeros(lats.shape)
    cm[nh_idx] = cm_nh
    cm[sh_idx] = cm_sh

    return conc, cm

def comiso_blend(sics, devs):
    """ The original comiso blending: switch to ComisoP above ComisoF > 95 """
    bcom_sic = np.where(sics[0]<0.95,sics[0],sics[1])
    bcom_dev = np.where(sics[0]<0.95,devs[0],devs[1])

    return bcom_sic, bcom_dev

def osisaf_blend(sics, devs):
    """ The original osisaf blending """
    fcomiso_conc = sics[0]
    bristol_conc = sics[1]
    fcomiso_sdev = devs[0]
    bristol_sdev = devs[1]

    threshold = 0.4

    blend = (np.abs(threshold - fcomiso_conc) + threshold - fcomiso_conc) / \
        (2 * threshold)
    below_mask = (fcomiso_conc < 0)
    bf = (blend * (1 - below_mask)) + below_mask

    conc = (1 - bf) * bristol_conc + bf * fcomiso_conc
    sdev = ((1 - bf) * bristol_sdev**2 + bf * fcomiso_sdev**2)**0.5

    return conc, sdev

def sicci1_blend(sics, devs, dals=None):
    """ The SICCI1 blending """
    fcomiso_conc = sics[0]
    bristol_conc = sics[1]
    fcomiso_sdev = devs[0]
    bristol_sdev = devs[1]

    beg_blending = 0.7
    end_blending = 0.9
    len_blending = end_blending - beg_blending
    wCF = np.where((fcomiso_conc >= beg_blending) * (fcomiso_conc <= end_blending),\
            1. - (fcomiso_conc - beg_blending)/len_blending,\
            1.)
    wCF[fcomiso_conc > end_blending] = 0.

    conc = (1 - wCF) * bristol_conc + wCF * fcomiso_conc
    sdev = ((1 - wCF) * bristol_sdev**2 + wCF * fcomiso_sdev**2)**0.5
    ret = [conc, sdev]
    if dals is not None:
        fcomiso_dal = dals[0]
        bristol_dal = dals[1]
        dal  = (1 - wCF) * bristol_dal + wCF * fcomiso_dal
        ret.append(dal,)

    return ret

