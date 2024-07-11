#!/usr/bin/env python3
import os, sys
import argparse
from datetime import datetime, timedelta
try:
    import simplejson as json
except ImportError:
    import json
from glob import glob
import itertools
import shutil
import logging
import logging.config
import numpy as np
import netCDF4 as nc

sys.path.insert(0, '/home/marinadm/python_scripts/jupyter/acciberg/tpd_files/gitlab/')
#import io_handler
import dynamic_tiepoints as dtp

sys.path.insert(0, '/home/marinadm/pmr_sic/')
from pmr_sic import common, hybrid_algo

LOG = logging.getLogger(__name__)


def read_dyn_algos(sat_id, json_dir, corr, period, dt, centre, algorithms,
    ignore, yesterday):
    """
        read dynamic tiepoints and dynamic ice conc algos from tiepoint directory

        :param sat_id: satellite identifier e.g. ssmi_f15
        :type sat_id: string
        :param json_dir: directory containing dynamical tiepoint json files
        :type json_dir: string
        :param dt: The tiepoint timestamp, optional defaults to yesterday
        :type dt: datetime
        :returns: dictionary with keys 'nh' and 'sh' containing tiepoints for
            northern and southern hemisphere.
    """
    if ( yesterday ):
        time_str = (dt - timedelta(days=1)).strftime('%Y%m%d')
        time_str_2 = (dt - timedelta(days=2)).strftime('%Y%m%d')
    else:
        time_str = dt.strftime('%Y%m%d')

    pstr = 'l{}'.format(period)
    if ( centre ):
        pstr = 'c{}'.format(period)
    
    match_corralgo = False
    try:
        if ( len(corr.split(',')) > 1 ):
            match_corralgo = True
    except AttributeError:
        pass
    
    if ( (corr is None or corr == 'ucorr') ):
        tmp_algos = []
        for alg in algorithms:
            if ( len(alg.split('_')) == 2 ):
                a1, a2 = alg.split('_')
                if ( a1.lower() not in tmp_algos ):
                    tmp_algos.append(a1.lower())
                if ( a2.lower() not in tmp_algos ):
                    tmp_algos.append(a2.lower())
            else:
                if ( alg not in tmp_algos ):
                    tmp_algos.append(alg)
        algorithms = tmp_algos

    dynalgos = dict()
    for alg in algorithms:
        dcorr = corr
        if ( corr == 'algos' or match_corralgo ):
            dcorr = alg
        dynalgos[alg] = dict()
        
        for area in ['nh','sh']:

            # Load the dynamic ice concentration algorithms from the json content
            if ( len(alg.split('_')) == 2 ):
                a1, a2 = alg.split('_')
                fname = 'dynAlgo_{}_{}_{}_corr{}_{}_{}days_avg.json'.format(area, sat_id,
                    a1.upper(), a2.upper(), time_str, pstr)
            else:    
                fname = 'dynAlgo_{}_{}_{}_{}_{}_{}days_avg.json'.format(area, sat_id,
                    alg.upper(), dtp.getcorr(dcorr), time_str, pstr)
            fpath = os.path.join(json_dir, fname)
            
            if ( yesterday ):
                # If yesterday option, then allow using TPA file from an additional
                # day in the past for when script is run early in the night before DM1
                if ( not os.path.isfile(fname) ):
                    fname = fname.replace(time_str, time_str_2)
                    fpath = os.path.join(json_dir, fname)
            
            try:
                with open(fpath) as fp:
                    dyn_tp = json.load(fp)
            except IOError as e:
                if ( dt.year == 1978 and corr == 'algos' ):
                    # Exit gracefully (avoiding exception) if we're in 1978 as we're expecting
                    # periods with no TP avg file for corr=algos.
                    LOG.warning('{} not found. Exiting since in 1978.'.format(fname))
                    sys.exit(0)
                else:
                    if ( ignore ):
                        LOG.warning('Did not find {} in {}'.format(fname, json_dir))
                        sys.exit(0)
                    else:
                        raise Exception('Did not find {} in {}'.format(fname, json_dir))

            algo_dict = common.numpify(dyn_tp)
            dynalgos[alg][area] = hybrid_algo.HybridSICAlgo.from_dict(algo_dict)

    return dynalgos
