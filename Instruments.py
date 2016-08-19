import abc

import sys
import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units

class NICMOS(object):
    """
    class defining the properties of the NICMOS instrument
    image shape, pixel scale, IWA
    PSF is assumed to be centered at psf_shape/2
    """
    def __init__(self):
        """
        Set the pixel scale, angular resolution, coronagraph properties
        """
        self.Nx = 80 * units.pixel
        self.Ny = 80 * units.pixel
        self.pix_scale = 75 * units.mas/units.pixel # citation needed
        self.IWA = 400 * units.mas # citation needed

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
    
