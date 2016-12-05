import sys
import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units

from Instrument import Instrument

# unit definitions for flux conversion
photflam = (units.erg / (units.cm**2 * units.Angstrom * units.count))

class NICMOS(Instrument):
    """
    class defining the properties of the NICMOS instrument
    image shape, pixel scale, IWA
    PSF is assumed to be centered at psf_shape/2
    """
    def __init__(self):
        """
        Set the pixel scale, angular resolution, coronagraph properties
        """
        self.Nx = 80 * units.pixel # redundant
        self.Ny = 80 * units.pixel # redundant
        self.imshape = np.array([80,80])
        self.center = np.array([40,40])
        self.pix_scale = 75 * units.mas/units.pixel # citation needed
        self.IWA = 6 * units.pixel * self.pix_scale #600 * units.mas # citation needed
        self.IWApix = self.IWA/self.pix_scale
        self.IWAmask = self.make_IWA_mask(self.imshape, self.center, self.IWApix.value)


    # photometric conversions
    def photometric_conversion(self, filt_name):
        """Photometric conversions for NICMOS filters"""
        phot_dict = {}
        phot_dict['F160W'] = {'PHOTFLAM':2.36501e-19,
                              'PHOTFLAMERR':0.0070,
                              'PHOTPLAM':16059.9,
                              'PHOTBW':1177.3,
                              'PHOTZPT':-21.10,
                              'PHOTFNU':2.03470e-06,
                              'PHOTFNUERR':0.0070,
                              '<F_NU(VEGA)>':1083.9,
                              'APCOR':1.1877}
        return phot_dict[filt_name.upper()]
                                      

    # Image stuff
    @property
    def center(self):
        """Central pixel coordinates in [row,col]"""
        return self._center
    @center.setter
    def center(self, newval):
        self._center = newval
    @property
    def imshape(self):
        return self._imshape
    @imshape.setter
    def imshape(self, newval):
        self._imshape = newval

    # IWA and mask
    @property
    def IWA(self):
        """FPM diameter, in mas"""
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval
        self.IWApix = self.IWA/self.pix_scale
    @property
    def IWApix(self):
        """FPM diameter, in pixels"""
        return self._IWApix
    @IWApix.setter
    def IWApix(self, newval):
        self._IWApix = newval
        self.IWAmask = self.make_IWA_mask(self.imshape, self.center, self.IWApix.value)
    @property
    def IWAmask(self):
        "1's outside, 0's inside"
        return self._IWAmask
    @IWAmask.setter
    def IWAmask(self, newval):
        self._IWAmask = newval
    # mask generator    
    def make_IWA_mask(self, shape, center, iwa):
        rad = np.linalg.norm(np.indices(shape) - center[:,None,None], axis=0)
        outside = np.where(rad >= iwa/2.)
        mask = np.zeros(shape)
        mask[outside] = 1
        return mask
        
        
    # PSF STUFF
    @property
    def psf(self):
        """
        A normalized model of the instrument PSF, possible a cutout
        used for forward modeling and injection
        """
        return self._psf
    @psf.setter
    def psf(self, new_psf):
        # normalized PSF
        self._psf = new_psf / np.nansum(new_psf)
    
    @property
    def frac_flux(self):
        """
        The fraction of the flux contained in the PSF cutout
        """
        return self._frac_flux
    @frac_flux.setter
    def frac_flux(self, newval):
        self._frac_flux = newval

    def load_psf(self, psf, center, diam=11):
        """
        Get a PSF centered at center and cut out a section of it
        Inputs:
            psf: image of the psf
            center: center in (row, col) of the psf in the image
            diam: box size of the PSF (default 11)
        Returns:
            returns nothing but sets self.psf to a diam x diam cutout of the PSF
        """
        center = np.array(center)
        rad = np.int(np.floor(diam/2.))
        init_flux = np.nansum(psf)
        psf = psf[center[0]-rad:center[0]+rad + diam%2,
                  center[1]-rad:center[1]+rad + diam%2]
        final_flux = np.nansum(psf)
        self.psf = psf
        self.frac_flux = final_flux/np.float(init_flux)
    
