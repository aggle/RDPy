import numpy as np
import pandas as pd
from scipy import signal
from . import RDIklip as RK
from . import utils


def calc_matched_filter_throughput(matched_filter):
    """
    Calculate the throughput of the matched filter
    Args:
      matched_filter: 2-D image with the filter on it
    Returns:
      throughput: the throughput correction for the matched filter
         correct signal = (matched_filter_result) / throughput
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
    mf_result = signal.fftconvolve(image, matched_filter, mode='same')
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
