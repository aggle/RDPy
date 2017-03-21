"""
Matched Filter stuff
"""

import numpy as np
import utils
import RDIklip as RK
from scipy import signal

##############################
# GENERATING MATCHED FILTERS #
##############################
def generate_matched_filter(psf, kl_basis=None, n_bases=None,
                            imshape=None, region_pix=None,
                            mf_locations=None):
    """
    generate a matched filter with a model of the PSF. For now we assume that
    the KL basis covers the entire image.
    You can pass a ReferenceCube objects that has all the required arguments as attributes
    Arguments:
        psf: square postage stamp of the PSF
        kl_basis [None]: karhunen-loeve basis for PC projection
        n_basis [None]: list of number of KL modes to use. Default: all (None defaults to all)
        imshape [None]: Nrows x Ncols
        region_pix: the raveled image pixel indices covered by the KL basis. 
            if None, assumed to cover the whole image
        mf_locations: the pixel coordinates of the locations to apply the matched filter.
            Can be raveled coordinates OR [Nloc x 2] array (then, must provide imshape)
        raveled_ind [False]: if True, treat the mf_location coordinates as for a raveled array
        offset: apply a linear shift to the matched filter (e.g. mean subtraction)
    Returns:
        MF: [Nloc x [Nkl x [Npix]]] cube of flattened matched filters. 
            The first index tells you what pixel in the image it corresponds to, 
            through np.unravel_index(i, imshape)
            if mf_locations is given, this will be 0 except for the slices corresponding to mf_pix
            (slice = np.ravel_multi_index(mf_pix.T, imshape, mode='clip')
    """
    # if mf_locations is not set, assume you want a MF at every pixel in the image
    if mf_locations is None:
        mf_locations = list(range(imshape[0]*imshape[1]))
    if np.ndim(mf_locations) == 0:
        mf_locations = [mf_locations]
    if np.ndim(mf_locations) == 2:
        mf_locations = np.ravel_multi_index(np.array(mf_locations).T, imshape)
    mf_locations = np.array(mf_locations)

    # if region_pix is not set, assume the whole image was used for KLIP
    if region_pix is None:
        region_pix = list(range(imshape[0]*imshape[1]))
    if np.ndim(n_bases) == 0:
        n_bases = np.expand_dims(n_bases,0)

    # this stores the PSFs
    mf_template_cube = np.zeros((np.size(mf_locations), imshape[0], imshape[1]))
    # this is used to pick out the desired PSF region *after* KLIP
    mf_pickout_cube = np.zeros((np.size(mf_locations),imshape[0], imshape[1]))

    # inject the instrument PSFs - this cannot be done on a flattened cube
    # only inject PSFs at the specified positions
    # inject_psf will automatically work across the n_bases axis
    try:
        for i, p in enumerate(mf_locations):
            center = np.unravel_index(p, imshape) # injection location
            mf_template_cube[i] = utils.inject_psf(np.zeros(imshape), psf, center)
            mf_pickout_cube[i]  = utils.inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)
    except TypeError: # case for only one location
        i,p = 0,mf_locations
        center = np.unravel_index(p, imshape) # injection location
        mf_template_cube[i] = utils.inject_psf(np.zeros(imshape), psf, center)
        mf_pickout_cube[i]  = utils.inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)

    # flatten along the pixel axis
    mf_flat_template = utils.flatten_image_axes(mf_template_cube)
    mf_flat_pickout =  utils.flatten_image_axes(mf_pickout_cube)

    MF = mf_flat_template

    # if no KL basis is supplied, return the *UNMODIFIED* psf
    # otherwise, go on to the KLIP subtraction of the MF
    if kl_basis is not None:
        # store the klip-subtracted PSF models here
        # index across locations and KL modes
        mf_flat_template_klipped = np.zeros((len(n_bases), len(mf_locations),
                                             imshape[0], imshape[1]))
        mf_flat_template_klipped = utils.flatten_image_axes(mf_flat_template_klipped)

        nloc = list(range(len(mf_locations)))
        for i in nloc:
            psf_template = mf_flat_template[i, region_pix]
            # don't be intimidated - fancy python crap to match array dims
            tmp = np.tile(RK.klip_subtract_with_basis(psf_template,
                                                      kl_basis,
                                                      n_bases),
                          np.ones_like(mf_flat_template.shape))
            mf_flat_template_klipped[:,i,region_pix] = tmp
            mf_flat_template_klipped *= mf_flat_pickout
            mf_norm_flat = fmmf_throughput_correction(mf_flat_template[:, region_pix],
                                                      kl_basis, n_bases)
            MF = mf_flat_template_klipped/np.expand_dims(mf_norm_flat, -1)
            MF = np.roll_axis(MF, 0, 2) # put the locations axis first

    # we only want the places where the template was injected to contribute
    MF *= mf_flat_pickout
    # remember to subtract the mean of the PSF
    MF = MF - np.expand_dims(np.nanmean(MF, axis=-1), -1)#  * mf_flat_pickout

    return MF



def generate_matched_filter_old(psf, kl_basis=None, n_bases=None,
                            imshape=None,
                            region_pix=None,
                            mf_locations=None):
    """
    generate a matched filter with a model of the PSF. For now we assume that 
    the KL basis covers the entire image.
    You can pass a ReferenceCube objects that has all the required arguments as attributes
    Arguments:
        psf: square postage stamp of the PSF
        kl_basis [None]: karhunen-loeve basis for PC projection
        n_basis [None]: list of number of KL modes to use. Default: all (None defaults to all)
        imshape [None]: Nrows x Ncols
        region_pix: the raveled image pixel indices covered by the KL basis. 
            if None, assumed to cover the whole image
        mf_locations: the pixel coordinates of the locations to apply the matched filter.
            Can be raveled coordinates OR [Nloc x 2] array (then, must provide imshape)
        raveled_ind [False]: if True, treat the mf_location coordinates as for a raveled array
        offset: apply a linear shift to the matched filter (e.g. mean subtraction)
    Returns:
        MF: [Nloc x [Nkl x [Npix]]] cube of flattened matched filters. 
            The first index tells you what pixel in the image it corresponds to, 
            through np.unravel_index(i, imshape)
            if mf_locations is given, this will be 0 except for the slices corresponding to mf_pix
            (slice = np.ravel_multi_index(mf_pix.T, imshape, mode='clip')
    """
    # if mf_locations is not set, assume you want a MF at every pixel in the image
    if mf_locations is None:
        mf_locations = list(range(imshape[0]*imshape[1]))
    if np.ndim(mf_locations) == 0:
        mf_locations = [mf_locations]
    if np.ndim(mf_locations) == 2:
        mf_locations = np.ravel_multi_index(np.array(mf_locations).T, imshape)
    mf_locations = np.array(mf_locations)

    # if region_pix is not set, assume the whole image was used for KLIP
    if region_pix is None:
        region_pix = list(range(imshape[0]*imshape[1]))
    if np.ndim(n_bases) == 0:
        n_bases = np.expand_dims(n_bases,0)

    # this stores the PSFs
    mf_template_cube = np.zeros((np.size(mf_locations), imshape[0], imshape[1]))
    # this is used to pick out the desired PSF region *after* KLIP
    mf_pickout_cube = np.zeros((np.size(mf_locations),imshape[0], imshape[1]))

    # inject the instrument PSFs - this cannot be done on a flattened cube
    # only inject PSFs at the specified positions
    # inject_psf will automatically work across the n_bases axis
    try:
        for i, p in enumerate(mf_locations):
            center = np.unravel_index(p, imshape) # injection location
            mf_template_cube[i] = utils.inject_psf(np.zeros(imshape), psf, center)
            mf_pickout_cube[i]  = utils.inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)
    except TypeError: # case for only one location
        i,p = 0,mf_locations
        center = np.unravel_index(p, imshape) # injection location
        mf_template_cube[i] = utils.inject_psf(np.zeros(imshape), psf, center)
        mf_pickout_cube[i]  = utils.inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)

    # flatten along the pixel axis
    mf_flat_template = utils.flatten_image_axes(mf_template_cube)
    mf_flat_pickout =  utils.flatten_image_axes(mf_pickout_cube)

    # if no KL basis is supplied, return the *UNMODIFIED* psf
    if kl_basis is None:
        MF = mf_flat_template * mf_flat_pickout
    # otherwise, go on to the KLIP subtraction of the MF
    else:
        # store the klip-subtracted PSF models here
        # index across locations and KL modes
        mf_flat_template_klipped = np.zeros((len(n_bases), len(mf_locations),
                                             imshape[0], imshape[1]))
        mf_flat_template_klipped = utils.flatten_image_axes(mf_flat_template_klipped)

        nloc = list(range(len(mf_locations)))
        for i in nloc:
            psf_template = mf_flat_template[i, region_pix]
            # don't be intimidated - fancy python crap to match array dims
            tmp = np.tile(RK.klip_subtract_with_basis(psf_template,
                                                      kl_basis,
                                                      n_bases),
                          np.ones_like(mf_flat_template.shape))
            mf_flat_template_klipped[:,i,region_pix] = tmp
            mf_flat_template_klipped *= mf_flat_pickout
            mf_norm_flat = fmmf_throughput_correction(mf_flat_template[:, region_pix],
                                                      kl_basis, n_bases)
            MF = mf_flat_template_klipped/np.expand_dims(mf_norm_flat, -1)
            MF = np.roll_axis(MF, 0, 2) # put the locations axis first
    # remember to subtract the mean
    MF = MF - np.expand_dims(np.nanmean(MF, axis=-1), -1)

    return MF


############################
# APPLYING MATCHED FILTERS #
############################

def apply_matched_filter(images, matched_filter=None,
                         throughput_corr=False, scale=1):
    """
    Apply a matched filter to an image. It is assumed that the image and
    the matched filter have already been sampled to the same resolution, and
    that the regions correspond to each other.
    Arguments:
        image: 1-D flattened image or N-D array of images where the last axis
          is the flattened region of interest. The image(s) and the matched
          filter must have the last two axes aligned.
          Tip: use utils.flatten_image_axes(images) to call on 2-D images
        matched_filter: Cube of flattened matched filters, one MF per pixel
          where we want to know the flux.
        throughput_corr: throughput correction for the matched filter. If None,
          then defaults to |MF|^2
        scale: any additional scaling you want to apply to the matched filter
    Returns:
        mf_map: (array of) images where the matched filter has been applied at
          each pixel
    """
    orig_shape = np.shape(images)
    if images.ndim == 1:
        images = np.expand_dims(images, 0)

    if np.ndim(throughput_corr) == 1:
        throughput_corr = np.expand_dims(throughput_corr, -1)
    # numpy cannot handle nan's so set all nan's to 0! the effect is the same
    nanpix_img = np.where(np.isnan(images))
    nanpix_mf = np.where(np.isnan(matched_filter))
    images[nanpix_img] = 0
    matched_filter[nanpix_mf] = 0

    # apply the matched filter
    mf_map = np.dot(matched_filter, np.rollaxis(images, -1, -2)).T
    # undo the funky linear algebra reshaping
    mf_map = np.rollaxis(mf_map.T, 0, mf_map.ndim)

    # if throughput corrections are provided,
    # apply them to the matched filter results
    # otherwise, use the matched filter norm
    if isinstance(throughput_corr, bool):
        throughput_corr = np.linalg.norm(matched_filter, axis=-1)**2
    mf_map /= throughput_corr

    # multiply the remaining scale factors
    mf_map *= scale
    return np.squeeze(mf_map)


def correlate_mf_template(image, filter_template, filter_norm=None):
    """
    Use scipy.signal.correlate to apply the matched filter to the data
    Arguments:
      image: [Ni x [Nj x [... x [Nx x Ny]]]] image to correlate with the MF
        the last 2 axes are the image axes (usually NICMOS.imshape)
      filter_template: 2D stamp of the PSF
      filter_norm: [None] normalization applied to the filter result (division)
        shape must be compatible with dividing the image.
        Default is no normalization (i.e. division by 1)
    Returns:
      matched_filter: the result of applying the filter template. Same
        dimensions as input image
    """
    # number of dimensions must match
    for dim in range(np.ndim(image) - np.ndim(filter_template)):
        filter_template = np.expand_dims(filter_template, 0)
    # apply the filter template
    mf_result = signal.correlate(image, filter_template, mode='same')
    if filter_norm is None:
        filter_norm = 1
    # you have the option of applying normalization here
    mf_result /= filter_norm
    return mf_result


def apply_matched_filter_to_image(image, matched_filter=None, locations=None,
                                  throughput_corr=False):
    """
    LEGACY CODE
    Apply a matched filter to an image. It is assumed that the image and the
      matched filter have already been sampled to the same resolution.
    Arguments:
        image: 2-D image
        matched_filter: Cube of flattened matched filters, one MF per pixel
          (flattened along the second axis)
        locations: flattened pixel indices of the pixels to apply the matched
          filter
    Returns:
        mf_map: 2-D image where the matched filter has been applied
        throughput_corr: [False] to apply throughput correction, pass an array
          of normalization factors that matches the shape of the matched filter
    """
    # get the image into the right shape
    orig_shape = np.copy(image.shape)
    im_shape = image.shape[-2:]
    if image.ndim == 2:
        image = image.ravel()
        image = np.expand_dims(image, 0)
    elif image.ndim >= 3:
        image = utils.flatten_image_axes(image)

    # numpy cannot handle nan's so set all nan's to 0! the effect is the same
    nanpix_img = np.where(np.isnan(image))
    nanpix_mf = np.where(np.isnan(matched_filter))
    image[nanpix_img] = 0
    matched_filter[nanpix_mf] = 0

    # set up the location indices
    if locations is None:
        locations = list(range(im_shape[0]*im_shape[1]))

    # instantiate the map
    mf_map = np.zeros_like(image)

    # apply matrix multiplication
    try:
        mf_map[..., locations] = np.array([np.diag(np.dot(mf, np.rollaxis(image, -1, -2)))
                                           for mf in matched_filter]).T
    except ValueError:
        mf_map[..., locations] = np.array([np.dot(mf, np.rollaxis(image, -1, -2))
                                           for mf in matched_filter]).T

    # if throughput corrections are provided, apply them to the matched filter results
    if not isinstance(throughput_corr, bool):
        mf_map /= throughput_corr
    # put the nans back
    matched_filter[nanpix_mf] = np.nan
    mf_map[nanpix_img] = np.nan
    return mf_map.reshape(orig_shape)


def apply_matched_filter_to_images(image, matched_filter=None, locations=None,
                                   throughput_corr=False, im_shape=None,
                                   scale=1):
    """
    LEGACY CODE
    Apply a matched filter to an image. It is assumed that the image and the
    matched filter have already been sampled to the same resolution, and that
    the regions correspond to each other.
    Arguments:
        image: 2-D image or 3-D array of images
            the image(s) and the matched filter must have the last two axes aligned
        matched_filter: Cube of flattened matched filters, one MF per pixel (flattened along the second axis)
        locations: flattened pixel indices of the pixels to apply the matched filter
        throughput_corr: [False] to apply throughput correction, pass an array of normalization factors 
            that matches the shape of the matched filter
        scale: any additional scaling you want to apply to the matched filter

    Returns:
        mf_map: 2-D image where the matched filter has been applied at each pixel
    """
    # get the image into the right shape
    orig_shape = np.copy(image.shape)
    im_shape = image.shape[-2:]
    if image.ndim == 2:
        image = image.ravel()
        image = np.expand_dims(image, 0)
    elif image.ndim >= 3:
        image = utils.flatten_image_axes(image)

    # numpy cannot handle nan's so set all nan's to 0! the effect is the same
    nanpix_img = np.where(np.isnan(image))
    nanpix_mf = np.where(np.isnan(matched_filter))
    image[nanpix_img] = 0
    matched_filter[nanpix_mf] = 0

    # instantiate the map
    if locations is None:
        locations = list(range(im_shape[0]*im_shape[1]))
    locations = np.array(locations)

    mf_map = np.dot(matched_filter, np.rollaxis(image, -1, -2)).T
    # undo the funky linear algebra reshaping
    mf_map = np.rollaxis(mf_map.T, 0, mf_map.ndim)
    # if throughput corrections are provided, apply them to the MF results
    if not isinstance(throughput_corr, bool):
        mf_map /= throughput_corr
    # multiply the remaining scale factors
    mf_map *= scale
    # put the nans back
    matched_filter[nanpix_mf] = np.nan
    # mf_map[nanpix_img] = np.nan

    # restore shape: should be all leading axes + mf locations
    final_shape = list(orig_shape[:-2])+[np.size((locations))]
    return mf_map.reshape(final_shape)


##############
# THROUGHPUT #
##############
def fmmf_throughput_correction(psfs, kl_basis=None, n_bases=None):
    """
    Calculate the normalization aka throughput correction factor for the
    matched filter, to get flux out
    New hotness: the PSF should only include the region of the MF!! not the
    Arguments:
        psfs: the flattened model psfs (Nloc, Region_pix)
        kl_basis: the KL basis for projection (KLmax, Region_pix)
        n_bases [None]: list of KL_max values
    returns:
        [(n_basis x) Nloc] throughput correction, per KLmax per location
    """
    # do KLIP on the PSFs, then take the normalization
    # one throughput for each psf. Final shape: Nloc x N_klmax
    if kl_basis is None:
        psfs_modded = psfs
    else:
        psfs_modded = RK.klip_subtract_with_basis(psfs, kl_basis, n_bases)
        # expand mask along the KL axis
        # mask = np.tile(np.expand_dims(mask, 1), (1, len(n_bases), 1))
    # before you take the norm, make sure you set regions outside the PSF to 0

    # prepare a mask that ensures only the relevant parts of the PSF are used
    mask = np.zeros_like(psfs)
    mask[np.where(psfs != 0)] = 1
    # mask = np.ones_like(psfs)
    normed_psfs = np.linalg.norm(psfs_modded * mask, axis=-1)**2
    # normed_psfs = np.linalg.norm(psfs_modded, axis=-1)**2
    return np.squeeze(normed_psfs.T)  # put the KL axis first, location aka pixel axis last

def fmmf_throughput_correction_old(psfs, kl_basis=None, n_bases=None):
    """
    DEPRECATED
    Calculate the normalization aka throughput correction factor for the matched filter, to get flux out
    Arguments:
        psfs: the flattened model psfs (Nloc, Region_pix)
        kl_basis: the KL basis for projection (KLmax, Region_pix)
        n_bases [None]: list of KL_max values
    returns:
        [(n_basis x) Nloc] throughput correction, per KLmax per location
    """
    if kl_basis is None:
        # no KL basis - just return the norm^2 of the matched filter
        return np.linalg.norm(psfs, axis=-1)**2
    if n_bases is None:
        n_bases = [len(kl_basis)+1]
    orig_shape = list(psfs.shape)
    psfs = utils.flatten_leading_axes(psfs)

    # Norm of the matched filter
    psf_norm = np.linalg.norm(psfs, axis=-1)**2
    # Projection of the matched filter onto the KL basis
    psf_basis_prod = np.dot(psfs, kl_basis.T)**2
    oversub = np.array([np.nansum(psf_basis_prod[...,:n], axis=-1) for n in n_bases])
    # put the throughput back into the original shape so that the indexes match for correcting
    # throughput.
    # we know the first axis is gonna be the KL modes, and the rest is the original shape
    # except for the pixel axis
    # temp comment
    #missing_flux = np.reshape(psf_norm[None,:] - oversub, [len(n_bases)] + orig_shape[:-1])
    missing_flux = np.reshape(psf_norm[None,:] - oversub, [len(n_bases)] + orig_shape[:-1])
    return np.squeeze(missing_flux)
