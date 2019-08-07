import numpy as np
import pandas as pd
import RDIklip as RK

class MatchedFilter(object):
    """
    An object that contains matched filter information:
    - A method to generate the matched filter using PSF and KLIP
    - A method to apply the matched filter and throughput correction
    - A method to calculate the throughput correction
    - The corresponding pixel coordinates
    - The matched filter array
    """

    def __init__(self, loc, psf=None, kl_basis=None):
        """
        Initialize 
        """
        pass
