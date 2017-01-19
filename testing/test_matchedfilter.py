import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import colors
from astropy.io import fits
import pandas as pd
from importlib import reload
from functools import reduce
from collections import OrderedDict
from IPython.display import clear_output

import unittest

from astropy.io import fits
from astropy import units

sys.path.append("/home/jaguilar/Work/BetaPic/")
import parseALICE

sys.path.append("/home/jaguilar/Work/BetaPic/RDI/")
import RDI
import utils
import MatchedFilter as MF
import NICMOS as NICMOSclass


# set up data
hdulist = fits.open("/home/jaguilar/Work/BetaPic/LP/BetaPicStuff2/referenceCube_1412081740.fits")
hdu = hdulist[0]
header = hdu.header
cube = hdu.data
hdulist.close()
cube_metadata = parseALICE.alice2dataframe(header)
bad_refs = parseALICE.badrefs_betaPic(header)

# pull data and references out of the ALICE cube
# image metadata
hr_metadata = cube_metadata[cube_metadata['TARGNAME'].apply(lambda x: x=='HR8799')].reset_index()
bp_metadata = cube_metadata[cube_metadata['TARGNAME'].apply(lambda x: x=='BETA-PIC-DISK')].reset_index()
badref_metadata = cube_metadata[cube_metadata['TARGNAME'].apply(lambda x: x in bad_refs)].reset_index()
goodref_metadata = cube_metadata[cube_metadata['TARGNAME'].apply(lambda x: x not in bad_refs)].reset_index()
# the cubes themselves
hr_cube = cube[hr_metadata['index']]
bp_cube = cube[bp_metadata['index']]
reference_cube = cube[goodref_metadata['index']]

# normalization factor
hr_flux_scale = 1/(NICMOS.frac_flux * hr_metadata['ALIGNNU'])
bp_flux_scale = 1/(NICMOS.frac_flux * bp_metadata['ALIGNNU'])

# Reference Cube
full_rc = RDI.ReferenceCube(reference_cube, instrument=NICMOS)


# NICMOS instrument
NICMOS = NICMOSclass.NICMOS()
# NICMOS.IWApix = NICMOS.IWApix*2.
psf_size = 20
NICMOS.load_psf(fits.getdata("/home/jaguilar/Work/BetaPic/LP/BetaPicStuff2/Era1_F160W.fits"),
                (45,45), psf_size)

# Test images
test_fluxes = np.array([50,75,100]) # these are the correct answers
test_pos = np.array([45,37])
test_cube = utils.inject_psf(np.zeros(NICMOS.imshape), NICMOS.psf, test_pos, scale_flux = test_fluxes)


# kl basis
full_rc.n_basis = np.arange(20,len(reference_cube), 10)
full_rc.generate_kl_basis()
full_rc.generate_matched_filter(mf_locations=np.expand_dims(test_pos, 0))

mf_throughput_noKL = MF.fmmf_throughput_correction(mf)
mf_throughput = MF.fmmf_throughput_correction(full_rc.matched_filter[:,full_rc.flat_region_ind], 
                                              full_rc.kl_basis, 
                                              full_rc.n_basis)
matched_filter = MF.rc.generate_matched_filter(mf_locations=np.expand_dims(targ_loc,0))


class MyTest(unittest.TestCase):
    """A test for each function"""
    def test_generate_matched_filter(self):
        pass
    def test_generate_matched_filter_noKL(self)

    def test_mf_noKL(self):
        fluxes = test_fluxes
        kl_basis = np.ones(full_rc.kl_basis.shape)
        kl_basis /= kl_basis.mean()
        self.assertAlmostEqual(1,1,places=6)
