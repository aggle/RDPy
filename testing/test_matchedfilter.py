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

from scipy import signal

import unittest

from astropy.io import fits
from astropy import units

sys.path.append("/home/jaguilar/Work/BetaPic/")
import parseALICE

sys.path.append("/home/jaguilar/Work/BetaPic/RDI/")
import RDI
import RDIklip as RK
import utils
import MatchedFilter as MF
import NICMOS as NICMOSclass



# NICMOS instrument
NICMOS = NICMOSclass.NICMOS()
# NICMOS.IWApix = NICMOS.IWApix*2.
psf_file="/home/jaguilar/Work/BetaPic/LP/BetaPicStuff2/Era1_F160W.fits"
psf_size = 10 # diameter = 21
NICMOS.load_psf(fits.getdata(psf_file), (45,45), psf_size)

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
full_rc.generate_matched_filter(mf_locations=[test_pos], mean_subtract=True)
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
        mf_result = MF.apply_matched_filter(ones_matched_filter,
                                            utils.flatten_image_axes(test_cube),
                                            throughput_corr=1)
        self.assertEqual(np.all(np.abs(mf_result/test_fluxes-1) < 1e-3), True,
                         "ones_filter does not give back right answer")

    
    def test_apply_injetion_random_bgnd(self):
        """Apply MF to an injection on a random background"""
        np.random.seed(1234)
        test_input = test_cube + np.random.normal(0,1,test_cube.shape[1:])
        mf_result = MF.apply_matched_filter(full_rc.matched_filter,
                                            utils.flatten_image_axes(test_input),
                                            throughput_corr=np.linalg.norm(full_rc.matched_filter)**2,
                                            scale=1.)
        test_against = test_fluxes.copy()
        for i in range(mf_result.ndim - test_fluxes.ndim):
            test_against = np.expand_dims(test_against,-1)
        self.assertEqual(np.all(np.abs(mf_result/test_against-1) < 0.05),
                         True,
                         "MF fails on random noise background")

    
    def test_apply_kl_filter_check_magnitude(self):
        """Apply MF to the KL-subtracted 0+PSF image and see if you get the injected fluxes back"""
        test_input = test_cube
        klsub = RDI.klip_subtract_with_basis(img_flat=utils.flatten_image_axes(test_input),
                                             kl_basis=full_rc.kl_basis,
                                             n_bases=full_rc.n_basis)
        mf_result = MF.apply_matched_filter(klsub,
                                            full_rc.matched_filter,
                                            throughput_corr = mf_throughput,
                                            scale=1)
        test_against = test_fluxes.copy()
        for i in range(mf_result.ndim - test_fluxes.ndim):
            test_against = np.expand_dims(test_against,-1)
        self.assertEqual(np.all(np.abs(mf_result/test_against-1) < 0.01),
                         True,
                         "MF with KL sub values wrong")

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
        #print(ratios.shape)
        self.assertEqual(np.all(np.diff(ratios[int(len(ratios)/2):]) > 0), False,
                         "MF slope never flattens out")

    def test_injection_location_even_psf(self):
        """Check that the max pixel of an injected psf is where you told it to be"""
        psf = NICMOS.load_psf(fits.getdata(psf_file), (45,45), 20, return_psf=True)
        inj_loc = np.vstack((np.arange(80), np.arange(80))).T
        injected_img = np.concatenate([[utils.inject_psf(np.zeros(NICMOS.imshape), psf, center=i)]
                                       for i in inj_loc], axis=0)
        self.assertEqual(np.all(inj_loc == np.vstack([np.unravel_index(np.argmax(img), NICMOS.imshape)
                                                      for img in injected_img])
                                ),
                         True,
                         "Even PSF not injected where you want it to be injected.")
    def test_injection_location_odd_psf(self):
        """Check that the max pixel of an injected psf is where you told it to be"""
        psf = NICMOS.load_psf(fits.getdata(psf_file), (45,45), 21, return_psf=True)
        inj_loc = np.vstack((np.arange(80), np.arange(80))).T
        injected_img = np.concatenate([[utils.inject_psf(np.zeros(NICMOS.imshape), psf, center=i)]
                                       for i in inj_loc], axis=0)
        self.assertEqual(np.all(inj_loc == np.vstack([np.unravel_index(np.argmax(img), NICMOS.imshape)
                                                      for img in injected_img])
                                ),
                         True,
                         "Odd PSF not injected where you want it to be injected.")

    def test_scipy_matched_filter_location(self):
        """
        Check that a MF gives you the same location of the maximum as the place 
        you injected the psf
        """
        psf = NICMOS.load_psf(fits.getdata(psf_file), (45,45), 21, return_psf=True)
        inj_loc = np.vstack((np.arange(80), np.arange(80))).T
        injected_img = np.concatenate([[utils.inject_psf(np.zeros(NICMOS.imshape), psf, center=i)]
                                       for i in inj_loc], axis=0)
        mfs = np.concatenate([[signal.correlate2d(img, psf, mode='same')] for img in injected_img])
        self.assertEqual(np.all(inj_loc == np.vstack([np.unravel_index(np.argmax(img), NICMOS.imshape)
                                                      for img in mfs])
                                ),
                         True,
                         "MF location not where you injected it.")

    def test_my_matched_filter_location(self):
        """
        Check that a MF gives you the same location of the maximum as the place 
        you injected the psf
        """
        psf = NICMOS.load_psf(fits.getdata(psf_file), (45,45), 10, return_psf=True)
        inj_loc = np.vstack((np.arange(80), np.arange(80))).T
        injected_img = np.concatenate([[utils.inject_psf(np.zeros(NICMOS.imshape), psf, center=i)]
                                       for i in inj_loc], axis=0)
        mf_template = MF.generate_matched_filter(psf, imshape=NICMOS.imshape, mf_locations=inj_loc)
        mf_result =  MF.apply_matched_filter(utils.flatten_image_axes(injected_img),
                                             mf_template,
                                             throughput_corr=False,
                                             scale=1)
        mf_img = np.zeros(NICMOS.imshape)
        for i,loc in enumerate(inj_loc):
            mf_img[loc[0],loc[1]] = mf_result[i,i]
        # if i put the MF in the right location, the MF result should be 1
        self.assertEqual(np.all(np.abs(mf_img[inj_loc.T[0], inj_loc.T[1]] - 1) < 1e-10),
                         True,
                         "MF location not where you injected it.")

    def test_mf_throughput_alignment(self):
        """
        Check that the MF throughput you calculate corresponds to the correct pixel
        """
        MF.fmmf_throughput_correction(full_rc.matched_filter[:, full_rc.flat_region_ind], 
                                              full_rc.kl_basis, 
                                              full_rc.n_basis)


if __name__=="__main__":

    unittest.main()
