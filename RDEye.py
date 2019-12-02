"""
RDEye.py - "Eye" as in "Private Eye" because it's a detective, get it????
This has code for making SNR maps and ROC curves
"""
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import sys
#sys.path.append('./')
from . import utils


###################
# Annular SNR map #
###################
def calc_pixel_distances(raveled_coords, shape):
    """
    Given a set of raveled coordinates and a shape, calculate a matrix of the distances between all the coordinates
    Returns:
      dist_mat: a symmetric matrix of distances between pixels in the order given
    """
    npix = raveled_coords.size
    unrav = np.unravel_index(raveled_coords, shape)
    x_dist = np.tile(unrav[0], (unrav[0].size, 1)) - np.tile(unrav[0], (unrav[0].size, 1)).T
    y_dist = np.tile(unrav[1], (unrav[1].size, 1)) - np.tile(unrav[1], (unrav[1].size, 1)).T
    dist = np.linalg.norm(np.stack([x_dist, y_dist], axis=0), axis=0)
    return dist

def find_nearest(array, vals):
    """
    for each val in vals, find the element in the array closest to val
    return the indices into array
    """
    idx = np.array([(np.abs(array-val)).argmin() for val in vals])
    return idx

def get_std_pix(raveled_coords, shape, thresh):
    """
    takes raveled img coordinates, the image shape, and a threshold
    returns a list of pixels that sample the coordinates, threshold distance apart
    """
    dist = calc_pixel_distances(raveled_coords, shape)
    std_pix = []
    for i, d in enumerate(dist):
        # find the pixel closest to each integer multiple of the distance threshold
        sample_pix = find_nearest(d, np.ceil(np.arange(thresh, d.max(), thresh)))
        std_pix.append(sample_pix)
    return std_pix

#def get_sample_pix(targ_pix, std_pix, shape, thresh):
#    """
#    Get the pixels to sample for the noise for a target pixel
#    Args:
#      targ_pix: raveled coordinate of the pixel you want to measure the noise for
#      std_pix: raveled coordinates of thee other pixels to consider
#      shape: shape of the original image
#     thresh: distance threshold to impose
#    """
#    std_unrav = np.unravel_index(std_pix, img_shape)
#    targ_unrav = np.unravel_index(targ_pix, img_shape)
#    x_dist = std_unrav[0] - targ_unrav[0]
#    y_dist = std_unrav[1] - targ_unrav[1]
#    dist_mat = np.linalg.norm(np.stack([x_dist, y_dist], axis=0), axis=0)
#    bool_mat = np.zeros_like(dist_mat, dtype=bool)
#
#    sample_pix = find_nearest(dist, np.arange(fwhm, dist.max(), fwhm))


def get_sample_pix(rad, phi, img_shape, fwhm):
    """
    get samples that won't be closer than a fwhm from each other
    If it crashes, it's probably because rad <~ 1
    """
    img_center = np.floor(np.array(img_shape)/2).astype(np.int)
    d_phi = np.arccos(1-0.5*((fwhm/rad)**2))
    n_samples = np.floor((2*np.pi)/d_phi).astype(np.int)
    phi_samples = np.linspace(d_phi, 2*np.pi-d_phi, n_samples) + phi
    # now get x and y positions
    row = np.round(rad * np.sin(phi_samples)).astype(np.int) + img_center[0]
    col = np.round(rad * np.cos(phi_samples)).astype(np.int) + img_center[1]
    # it's possible to get pixels that are outside the image. remove them
    keep_index = (row < img_shape[0]) & (col < img_shape[1])
    row = row[keep_index]
    col = col[keep_index]
    return row, col

def make_annular_std_map(flat_img, region_pix, img_shape, fwhm, n_rings=None):
    """
    Given a flattened image region of interest, make an SNR map
    Args:
      flat_img: flattened region of interest
      region_pix: the flattened pixel coordinates corresponding to the flat_img
      img_shape: the original shape of the image
      fwhm: PSF fwhm, i.e. the resolution element
      n_rings: number of annnuli. if None, defaults to width of 2*fwhm
    """
    if np.ndim(flat_img) == 1:
        flat_img = flat_img[None, :]
    std_vals = pd.DataFrame(np.zeros_like(flat_img).T, index=region_pix)
    # no idea if this works or not
    img_center = np.floor(np.array(img_shape)/2).astype(np.int)
    rad_pix, phi_pix = utils.get_radial_coordinate(img_shape, img_center)
    rad_pix_region = rad_pix.flat[region_pix] # radial values in the ROI
    phi_pix_region = phi_pix.flat[region_pix] # radial values in the ROI
    ann_range = rad_pix_region.min(), rad_pix_region.max()
    if n_rings is None:
        d_rad = 2*fwhm
    else:
        d_rad = np.array(ann_range)/n_rings

    # for every pixel, find the the std
    for i, pix in enumerate(region_pix):
        # get the radius and angle to sample a circle
        rad = rad_pix_region[i]
        phi = phi_pix_region[i]
        # get samples that won't be closer than a fwhm from each other
        row, col = get_sample_pix(rad, phi, img_shape, fwhm)
        ttest_corr = np.sqrt(1+1./len(row))
        std_vals.loc[pix] = np.nanstd(utils.make_image_from_flat(flat_img, region_pix, img_shape, squeeze=False)[:, row, col])
        std_vals.loc[pix] /= ttest_corr
    return np.squeeze(std_vals.T.values)

def make_annular_snr_map(flat_img, region_pix, img_shape, fwhm):
    """
    Given a flattened image region of interest, make an SNR map
    Args:
      flat_img: flattened region of interest
      region_pix: the flattened pixel coordinates corresponding to the flat_img
      img_shape: the original shape of the image
      fwhm: PSF fwhm, i.e. the resolution element
      n_rings: number of annnuli. if None, defaults to width of 2*fwhm
    """
    std_map = make_annular_std_map(flat_img, region_pix, img_shape, fwhm)
    snr_img = utils.make_image_from_flat(np.squeeze(flat_img/std_map),
                                         region_pix, img_shape)
    return snr_img

##############
# ROC curves #
##############
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

def generate_ROC_dataframe(Nimgs, levels, names):
    """
    Generate a dataframe to store the recovered flux for each set of parameters. The indices are given by levels.
    There is one column for each image in range(Nimgs)
    Args:
      Nimgs: the number of images in the cube whose photometry need to be stored
      levels: list of list-like objects that will be used to index the results eg [['a','b'],[0,1,2]]
      names: list of strings - the name of each level
    """
    index = pd.MultiIndex.from_product([level for level in levels], names=names)
    ref_cols = ['ref{0}'.format(i) for i in range(Nimgs)]
    df = pd.DataFrame(np.nan, index=index, columns=ref_cols)
    return df

