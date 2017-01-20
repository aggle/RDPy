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


# NICMOS instrument
NICMOS = NICMOSclass.NICMOS()
# NICMOS.IWApix = NICMOS.IWApix*2.
psf_size = 20
NICMOS.load_psf(fits.getdata("/home/jaguilar/Work/BetaPic/LP/BetaPicStuff2/Era1_F160W.fits"),
                (45,45), psf_size)

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


# Test images
test_fluxes = np.array([50,75,100], dtype=np.float) # these are the correct answers
test_pos = np.array([45,37])
test_cube = utils.inject_psf(np.zeros(NICMOS.imshape), NICMOS.psf, test_pos, scale_flux = test_fluxes)

# Reference Cube
full_rc = RDI.ReferenceCube(reference_cube, instrument=NICMOS)
# kl basis
full_rc.n_basis = np.arange(20,len(reference_cube), 10)
full_rc.generate_kl_basis()
# matched filter
# kl-basis matched filter
full_rc.generate_matched_filter(mf_locations=np.expand_dims(test_pos, 0), mean_subtract=True)
#full_rc.matched_filter * NICMOS.frac_flux
mf_throughput = MF.fmmf_throughput_correction(full_rc.matched_filter[:, full_rc.flat_region_ind], 
                                              full_rc.kl_basis, 
                                              full_rc.n_basis)
# ones only - give back sum of image
ones_matched_filter = np.ones((1, full_rc.npix_region))



class MyTest(unittest.TestCase):
    """Figure out the throughput problem!"""

    def test_apply_ones_filter(self):
        """Test that when you apply a matched filter, you get back the signal you expect."""
        # apply ones filter to the test images
        mf_result = MF.apply_matched_filter(ones_matched_filter, utils.flatten_image_axes(test_cube))
        self.assertEqual(np.all(np.abs(mf_result/test_fluxes-1) < 1e-3), True, "ones_filter does not give back right answer")

    def test_apply_kl_filter_check_magnitude(self):
        """Apply MF to the KL-subtracted image"""
        klsub = RDI.klip_subtract_with_basis(img_flat=utils.flatten_image_axes(test_cube),
                                             kl_basis=full_rc.kl_basis,
                                             n_bases=full_rc.n_basis)
        mf_result = MF.apply_matched_filter(klsub,
                                            full_rc.matched_filter,
                                            throughput_corr = mf_throughput,
                                            scale=1)
        self.assertEqual(np.all(np.abs(mf_result/np.expand_dims(test_fluxes,-1)-1) < 0.05), True, "MF with KL sub values wrong")

    def test_apply_kl_filter_check_slope(self):
        """Check that the flux goes monotonically down"""
        klsub = RDI.klip_subtract_with_basis(img_flat=utils.flatten_image_axes(test_cube),
                                             kl_basis=full_rc.kl_basis,
                                             n_bases=full_rc.n_basis)
        mf_result = MF.apply_matched_filter(klsub,
                                            full_rc.matched_filter,
                                            throughput_corr = mf_throughput,
                                            scale=1)
        ratios = mf_result/np.expand_dims(test_fluxes,-1)
        print(ratios.shape)
        self.assertEqual(np.any(np.diff(ratios[int(len(ratios)/2):]) >= 0), True, "MF slope never flattens out")

        

if __name__=="__main__":

    unittest.main()
