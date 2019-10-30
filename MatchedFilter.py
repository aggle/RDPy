import numpy as np
import pandas as pd
from scipy import signal
from . import RDIklip as RK
from . import utils
from . import NICMOS


def calc_matched_filter_throughput(matched_filter):
    """
    Calculate the throughput of the matched filter
    Args:
      matched_filter: 2-D image with the filter on it
    Returns:
      throughput: the throughput correction for the matched filter
         correct signal = (matched_filter_result) / throughput
    """
    orig_shape = matched_filter.shape
    nan_coords = np.isnan(matched_filter)
    matched_filter[nan_coords] = 0
    throughput = np.linalg.norm(utils.flatten_image_axes(matched_filter), axis=-1)**2
    matched_filter[nan_coords] = np.nan
    return throughput

def calc_matched_filter_throughput_klip(matched_filter,
                                        locations,
                                        kl_basis,
                                        num_basis,
                                        return_stamps=False):
    """
    Calculate the KLIP-corrected matched filter throughput
    Args:
      matched_filter: the matched filter stamp
      locations: Npix x 2 array of (row, col) locations you care about
      kl_basis: the full-image KLIP basis as [N_k, Nrow, Ncol] i.e. not raveled, not just a region)
      num_basis: array of the number of bases
      return_stamp [False]: if True, return the KL-subtracted matched filters for debugging
    Returns:
      N_locations array of throughput corrections
    """
    # first, get rid of nans in the KL basis
    kl_basis[np.isnan(kl_basis)] = 0
    stamp_shape = matched_filter.shape
    img_shape = kl_basis.shape[-2:]
    mf_klsub = np.zeros((len(locations), len(num_basis), stamp_shape[0]*stamp_shape[1]),
                        dtype=np.float)
    for i, loc in enumerate(locations):
        if (i+1)%400 == 0:
            print(f"{i+1} throughputs of f{len(locations)} completed")
        # pull out the KL basis stamp
        loc_ravel = np.int(np.ravel_multi_index(loc, img_shape))
        _, stamp_coords = utils.get_stamp_coordinates(loc, stamp_shape[0], stamp_shape[1], img_shape)
        #klip_stamp = utils.get_stamp_from_cube(kl_basis, stamp_shape, loc_ravel)
        # get rid of nan's for edge stamps
        #klip_stamp[np.isnan(klip_stamp)] = 0
        mf_image = utils.inject_psf(np.zeros(kl_basis.shape[-2:]),
                                    psf = matched_filter + matched_filter.min(),
                                    center = loc,
                                    subtract_mean = True)
        # do klip subtraction with the stamp
        klsub_result = RK.klip_subtract_with_basis(mf_image.ravel(), #matched_filter.ravel(),
                                                   utils.flatten_image_axes(kl_basis),#klip_stamp),
                                                   num_basis)
        # now pull out stamps around the filter
        mf_klsub_stamp = utils.get_stamp_from_cube(utils.make_image_from_region(klsub_result),
                                                   stamp_shape, loc_ravel)
        mf_klsub[i] = utils.flatten_image_axes(mf_klsub_stamp)

    # return KL-subtracted stamp for debugging
    if return_stamps == True:
        return mf_klsub

    throughput = calc_matched_filter_throughput(utils.make_image_from_region(mf_klsub))
    return throughput

def create_matched_filter(template):
    """
    Given a matched-filter template, mean-subtract and then scale by the throughput
    Template: 1-D or 2-D template for the matched filter
    """
    matched_filter = template - np.nanmean(template)
    throughput = calc_matched_filter_throughput(matched_filter)
    #matched_filter = matched_filter/throughput
    return matched_filter


# here for legacy
def high_pass_filter_stamp(stamp, filterwidth, target_image_shape):
    """
    Take special care when HPF-ing a stamp, because the image is usually
    a different shape from the target images and this affects the computation
    of the frequencies.
    The approach we're taking here is to assume the PSF model is smaller than
    the target images, and pad it with 0's before HPF.
    Returns:
      PSF with the high-pass filter applied
    """
    """
    # calculate the padding. If odd, add more padding to the front.
    diff = np.array(target_image_shape) - np.array(stamp.shape)
    lo_pad = np.ceil(diff/2).astype(np.int)
    hi_pad = np.floor(diff/2).astype(np.int)
    pad_stamp = np.pad(stamp, [lo_pad, hi_pad], mode='constant', constant_values=0)
    # Now that the STAMP is padded, apply the high pass filter
    filt_pad_stamp = RK.high_pass_filter(pad_stamp, filterwidth)
    # cut out a stamp of the filtered STAMP in the original shape
    pad_center = np.array(np.unravel_index(stamp.argmax(), stamp.shape)) + lo_pad
    pad_stamp_stamp_coords = utils.get_stamp_coordinates(pad_center,
                                                       *stamp.shape,
                                                       pad_stamp.shape)[0]
    filt_stamp = filt_pad_stamp[pad_stamp_stamp_coords[0],
                            pad_stamp_stamp_coords[1]]
    """
    filt_stamp = utils.high_pass_filter_stamp(stamp, filterwidth, target_image_shape)
    
    return filt_stamp


def apply_matched_filter_dot(target, matched_filter):
    """
    The matched filter is just a dot product. Inputs can be 2-D or 1-D.
    This is only valid for a single location
    Args:
      matched_filter: the matched filter aligned with the target
      target: the target of the matched filter
    Returns:
      single value: the result of the matched filtering
    """
    # make sure to remove nan's from the target
    return np.inner(utils.flatten_image_axes(utils.denan(target)),
                    matched_filter.flat)

# legacy compatibility
apply_matched_filter = apply_matched_filter_dot

def apply_matched_filter_dot_to_image(image, matched_filter, pixel_indices=None):
    """
    Apply the dot-product matched filter to an image. You can specify which
    pixels using pixel_indices - note that these are the *flattened* array
    coordinates. This works by making a stamp for each pixel in the region
    of interest.
    Args:
      image: 2-D image (the full image, not a region).
             Possibly in the future this can be flat, and any missing pixels
             will just be filled in with nan's
      matched_filter: 2-D matched filter
      pixel_indices: the (flattened) coordinates pixels you are interested in
    Returns:
      mf_image: an image with the same shape as the original image, having had
                the matched filter applied
    """
    cube_of_stamps = utils.get_cube_of_stamps_from_image(image,
                                                         matched_filter.shape,
                                                         pixel_indices)
    #mf_result_flat = np.inner(utils.denan(utils.flatten_image_axes(cube_of_stamps),
    #                                      matched_filter.flat))
    mf_result_flat = apply_matched_filter_dot(cube_of_stamps,
                                              matched_filter)
    mf_result = utils.make_image_from_region(mf_result_flat, pixel_indices, image.shape)
    return mf_result


def apply_matched_filter_dot_to_cube(cube, matched_filter, pixel_indices):
    """
    Just like apply_matched_filter_dot_to_image,
    but iterates quickly over a cube
    Args:
      cube: N-D cube, where the last 2 axes are the pixel axes. Must be full images
      matched_filter: 2-D matched filter
      pixel_indices: the (flattened) coordinates pixels you are interested in
    Returns:
      mf_image: an image with the same shape as the original image, having had
                the matched filter applied
    """
    orig_shape = cube.shape
    cube = utils.flatten_leading_axes(cube, -2)
    #mf_result = np.array([apply_matched_filter_fft(c, matched_filter) for c in cube])
    mf_result = np.array(list(map(lambda img: apply_matched_filter_dot_to_image(img,
                                                                                matched_filter,
                                                                                pixel_indices),
                                  cube)))
    mf_result = np.reshape(mf_result, orig_shape)
    return mf_result


def apply_matched_filter_fft(image, matched_filter):
    """
    Use the scipy signals processing library to use FFT to convolve the PSF with the whole image
    Uses scipy.signals.fftconvolve
    Args:
      image: the 2-D image you want to apply the MF to (note: *must* be 2-D)
      matched_filter: the signal you're looking for
    Returns:
      mf_result: the matched filtered image, with the same shape as the original image
    """
    mf_result = signal.fftconvolve(image, matched_filter, mode='same', axes=(-1, -2))
    return mf_result

def apply_matched_filter_fft_to_cube(cube, matched_filter):
    """
    Use the scipy signals processing library to use FFT to convolve the PSF with the whole image
    Uses scipy.signals.fftconvolve
    Args:
      cube: the 3-D cube of images you want to apply the MF to (last 2 axes are image pixels)
      matched_filter: the signal you're looking for
    Returns:
      mf_result: the matched filtered image, with the same shape as the original image
    """
    orig_shape = cube.shape
    cube = utils.flatten_leading_axes(cube, -2)
    #mf_result = np.array([apply_matched_filter_fft(c, matched_filter) for c in cube])
    mf_result = np.array(list(map(lambda img: apply_matched_filter_fft(img, matched_filter), cube)))
    mf_result = np.reshape(mf_result, orig_shape)
    return mf_result



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
