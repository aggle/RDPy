"""
RDEye.py - "Eye" as in "Private Eye" because it's a detective, get it????
"""


import numpy as np
from scipy.interpolate import interp1d

def frac_above_thresh(data, thresh):
    """
    thresh can be an integer or an array of thresholds
    """
    if np.size(thresh) == 1:
        thresh = np.array([thresh])
    return np.squeeze(np.sum( (data > thresh[:,None]), axis=-1))/np.float(data.size)

def get_roc_curve(data1, data2):
    """
    Get the ROC for testing if the population of data2 is drawn from the population of data1
    data1: null hypothesis
    data2: alternative hypothesis
    Returns:
        fpf: false positive fraction
        tpf: true positive fraction
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    num_data1 = np.float(data1.size)
    num_data2 = np.float(data2.size)
    # FPF thresholds
    fpf = np.linspace(0,1.,101)
    # interpolate the data1 values that correspond to the FPF thresholds
    func1 = interp1d(np.linspace(0,1., data1.size), data1)
    fpf_thresh = func1(fpf)
    # 
    tpf = frac_above_thresh(data2, fpf_thresh[::-1])
    return fpf, tpf
