"""
Matched Filter stuff
"""

import numpy as np


##############################
# GENERATING MATCHED FILTERS #
##############################

def generate_matched_filter(psf, kl_basis=None, n_bases=None,
                            imshape=None, region_pix=None, mf_locations=None):
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
        mf_locations: the *raveled* pixel coordinates of the locations to apply the matched filter 
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
    # if region_pix is not set, assume the whole image was used for KLIP
    if region_pix is None:
        region_pix = list(range(imshape[0]*imshape[1]))
    if np.ndim(n_bases) == 0:
        n_bases = np.expand_dims(n_bases,0)

    # this stores the PSFs
    mf_template_cube = np.zeros((len(mf_locations), imshape[0], imshape[1]))
    # this is used to pick out the desired PSF region *after* KLIP
    mf_pickout_cube = np.zeros((len(mf_locations),imshape[0],imshape[1]))
    
    # inject the instrument PSFs - this cannot be done on a flattened cube
    # only inject PSFs at the specified positions
    # inject_psf will automatically work across the n_bases axis
    for i, p in enumerate(mf_locations):
        center = np.unravel_index(p, imshape) # injection location
        mf_template_cube[i] = inject_psf(np.zeros(imshape), psf, center)
        mf_pickout_cube[i]  = inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)

    # flatten along the pixel axis
    mf_flat_template = utils.flatten_image_axes(mf_template_cube)
    mf_flat_pickout =  utils.flatten_image_axes(mf_pickout_cube)

    # if no KL basis is supplied, return the *UNMODIFIED* psf
    if kl_basis is None:
        return mf_flat_template * mf_flat_pickout
    # otherwise, go on to the KLIP subtraction of the MF
    
    # store the klip-subtracted PSF models here - index across locations and KL modes
    mf_flat_template_klipped = np.zeros((len(n_bases), len(mf_locations), 
                                         imshape[0], imshape[1]))
    mf_flat_template_klipped = utils.flatten_image_axes(mf_flat_template_klipped)

    nloc = list(range(len(mf_locations)))
    for i in nloc:
        psf_template = mf_flat_template[i,region_pix]
        # don't be intimidated - fancy python crap to match array dims
        tmp = np.tile(klip_subtract_with_basis(psf_template,
                                               kl_basis,
                                               n_bases),
                      np.ones_like(mf_flat_template.shape))
        mf_flat_template_klipped[:,i,region_pix] = tmp
    mf_flat_template_klipped *= mf_flat_pickout
    mf_norm_flat = fmmf_throughput_correction(mf_flat_template[:,region_pix],
                                              kl_basis, n_bases)
    MF = mf_flat_template_klipped/np.expand_dims(mf_norm_flat,-1)
    return np.rollaxis(MF,0,2) # put the locations axis first 

def generate_matched_filter_noKL(psf, imshape, region_pix=None, mf_locations=None):
    """
    Generate a matched filter with just the theoretical PSF, not the KL-modified one
    Arguments:
        psf: a PSF stamp
        imshape: the shape of the image that the matched filter will be tested against
        region_pix: the raveled image pixel indices covered by the KL basis. 
            if None, assumed to cover the whole image
        mf_locations: the *raveled* pixel coordinates of the locations to apply the matched filter 
    Returns:
        MF: flux-normalized cube of flattened matched filters. 
            The first index tells you what pixel in the image it corresponds to, through
            np.unravel_index(i, imshape) if mf_locations is given, it will be 0 except 
            for the slices corresponding to mf_pix 
            (slice = np.ravel_multi_index(mf_pix.T, imshape, mode='clip')
    """

    # use the right number of KL basis vectors
    # if mf_locations is not set, assume you want a MF at every pixel in the image
    if mf_locations is None:
        mf_locations = list(range(imshape[0]*imshape[1]))
    # if region_pix is not set, assume the whole image was used for KLIP
    if region_pix is None:
        region_pix = list(range(imshape[0]*imshape[1]))
    
    mf_template_cube = np.zeros((len(mf_locations),imshape[0],imshape[1]))
    mf_pickout_cube = np.zeros_like(mf_template_cube) # this is used to pick out the PSF *after* KLIP

    # inject the instrument PSFs - this cannot be done on a flattened cube
    # only inject PSFs at the specified positions
    for i, p in enumerate(mf_locations):
        center = np.unravel_index(p, imshape) # injection location
        mf_template_cube[i] = inject_psf(np.zeros(imshape), psf, center)
        mf_pickout_cube[i]  = inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)
    # flatten all the MF images
    mf_flat_template = np.array([i.ravel() for i in mf_template_cube])
    mf_flat_pickout =  np.array([i.ravel() for i in mf_pickout_cube])
    # find the klip-modified PSFs
    mf_flat_template_klipped = np.zeros_like(mf_flat_template)
    # when you apply KLIP, be careful only to use the selected region of the image

    # leave only the region of interest in the images
    mf_flat_template *= mf_flat_pickout
    return mf_flat_template

############################
# APPLYING MATCHED FILTERS #
############################

def apply_matched_filter_to_image(image, matched_filter=None, locations=None,
                                  throughput_corr=False):
    """
    Apply a matched filter to an image. It is assumed that the image and the matched filter have already been sampled to the same resolution.
    Arguments:
        image: 2-D image
        matched_filter: Cube of flattened matched filters, one MF per pixel (flattened along the second axis)
        locations: flattened pixel indices of the pixels to apply the matched filter
    Returns:
        mf_map: 2-D image where the matched filter has been applied
        throughput_corr: [False] to apply throughput correction, pass an array of normalization factors that matches the shape of the matched filter
    """
    # get the image into the right shape
    orig_shape = np.copy(image.shape)
    im_shape = image.shape[-2:]
    if image.ndim == 2:
        image = image.ravel()
        image = np.expand_dims(image,0)
    elif image.ndim >= 3:
        image = utils.flatten_image_axes(image)
    
    # numpy cannot handle nan's so set all nan's to 0! the effect is the same
    nanpix_img = np.where(np.isnan(image))
    nanpix_mf = np.where(np.isnan(matched_filter))
    image[nanpix_img] = 0
    matched_filter[nanpix_mf] = 0

    # instantiate the map
    mf_map = np.zeros_like(image)
    if locations is None:
        locations = list(range(im_shape[0]*im_shape[1]))

    # apply matrix multiplication
    try:
        mf_map[...,locations] = np.array([np.diag(np.dot(mf, np.rollaxis(image,-1,-2)))
                                          for mf in matched_filter]).T
    except ValueError:
        mf_map[...,locations] = np.array([np.dot(mf, np.rollaxis(image,-1,-2))
                                          for mf in matched_filter]).T
    
    # if throughput corrections are provided, apply them to the matched filter results
    if not isinstance(throughput_corr, bool):
        mf_map /= throughput_corr
    # put the nans back
    matched_filter[nanpix_mf] = np.nan
    mf_map[nanpix_img] = np.nan
    return mf_map.reshape(orig_shape)


def apply_matched_filter_to_images(image, matched_filter=None, locations=None,
                                   throughput_corr=False, scale=1):
    """
    Apply a matched filter to an image. It is assumed that the image and the matched filter have already been sampled to the same resolution.
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
        image = np.expand_dims(image,0)
    elif image.ndim >= 3:
        image = utils.flatten_image_axes(image)
    
    # numpy cannot handle nan's so set all nan's to 0! the effect is the same
    nanpix_img = np.where(np.isnan(image))
    nanpix_mf = np.where(np.isnan(matched_filter))
    image[nanpix_img] = 0
    matched_filter[nanpix_mf] = 0

    # instantiate the map
    #mf_map = np.zeros_like(image)
    if locations is None:
        locations = list(range(im_shape[0]*im_shape[1]))

    mf_map = np.dot(matched_filter, np.rollaxis(image, -1, -2)).T
    # undo the funky linear algebra reshaping
    mf_map = np.rollaxis(mf_map.T, 0, mf_map.ndim)
    # if throughput corrections are provided, apply them to the matched filter results
    if not isinstance(throughput_corr, bool):
        mf_map /= throughput_corr
    # multiply the remaining scale factors
    mf_map *= scale
    # put the nans back
    matched_filter[nanpix_mf] = np.nan
    mf_map[nanpix_img] = np.nan

    return mf_map.reshape(orig_shape)

##############
# THROUGHPUT #
##############

def fmmf_throughput_correction(psfs, kl_basis, n_bases=None):
    """
    Calculate the normalization aka throughput correction factor for the matched filter, to get flux out
    Arguments:
        psfs: the flattened model psfs (Nloc, Region_pix)
        kl_basis: the KL basis for projection (KLmax, Region_pix)
        n_bases [None]: list of KL_max values
    returns:
        [(n_basis x) 1] throughput correction, per KLmax per location
    """
    if n_bases is None:
        n_bases = [len(kl_basis)+1]
    orig_shape = list(psfs.shape)
    psfs = utils.flatten_leading_axes(psfs)

    #psf_norm  = np.nansum(psfs**2, axis=-1)
    psf_norm = np.linalg.norm(psfs, axis=-1)**2
    oversub = np.array([np.nansum(np.dot(psfs, kl_basis[:n].T)**2, axis=-1) for n in n_bases])
    
    missing_flux = psf_norm - oversub
    return missing_flux.reshape([len(n_bases)] + orig_shape[:-1])
