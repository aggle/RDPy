import abc

import sys
import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import units

class Instrument(object):
    """
    Abstract Class with the required fields and methods that need to be implemented

    Attributes:
      imshape: (Nrow, Ncol) image dimension, set my the reference cube
      center: (Nrow, Ncol) location fo the image center
      psf: (Nrow < imshape[0], Ncol <= imshape[1]) stamp of the instrument PSF
      frac_flux: float < 1 of the fraction of the total PSF energy contained in the psf stamp
    
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractproperty
    def imshape(self):
        """
        shape of the instrument's images
        """
        return
    @imshape.setter
    def imshape(self, newval):
        return


    @abc.abstractproperty
    def psf(self):
        """
        A normalized model of the instrument PSF, possible a cutout
        used for forward modeling and injection
        """
        return
    @psf.setter
    def psf(self, new_psf):
        # normalized PSF
        return
    
    @abc.abstractproperty
    def frac_flux(self):
        """
        The fraction of the flux contained in the PSF cutout
        """
        return
    @frac_flux.setter
    def frac_flux(self, newval):
        return

    ########################
    ### Required Methods ###
    ########################
    @abc.abstractmethod
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
        return NotImplementedError("Subclass needs to implement this!")
    
    
