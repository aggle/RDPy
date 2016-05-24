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
    should it have the PSF?
    """
    def __init__(self):
        """
        Set the pixel scale, angular resolution, coronagraph properties
        """
        self.Nx = 80 * units.pixel
        self.Ny = 80 * units.pixel
        self.pix_scale = 75 * units.mas/units.pixel
        self.IWA = 400 * units.mas

    @property
    def self.psf(self):
        return self._psf
    @psf.setter
    def self.psf(self, new_psf):
        # normalized PSF
        self._psf = new_psf / np.nansum(new_psf)
