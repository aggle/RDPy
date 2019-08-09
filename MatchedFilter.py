import numpy as np
import pandas as pd
from . import RDIklip as RK


def calc_matched_filter_throughput(matched_filter):
    """
    Calculate the throughput of the matched filter
    """
    throughput = np.linalg.norm(matched_filter)**2
    return throughput


def create_matched_filter(template):
    """
    Given a matched-filter template, mean-subtract and then scale by the throughput
    Template: 1-D or 2-D template for the matched filter
    """
    matched_filter = template - np.nanmean(template)
    throughput = calc_matched_filter_throughput(matched_filter)
    matched_filter = matched_filter/throughput
    return matched_filter


def apply_matched_filter(matched_filter, target):
    """
    The matched filter is just a dot product. Inputs can be 2-D or 1-D
    Args:
      matched_filter: the matched filter aligned with the target
      target: the target of the matched filter
    Returns:
      single value: the result of the matched filtering
    """
    return np.dot(matched_filter.ravel(), target.ravel())
    


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
