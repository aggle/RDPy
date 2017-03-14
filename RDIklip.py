"""
All functions that use KLIP
"""
import sys
import os

import numpy as np
import numpy.fft as fft

import scipy.linalg as la
import scipy.ndimage as ndimage

from functools import reduce

import utils



def generate_kl_basis(references, kl_max=None,
                      return_evecs=False, return_evals=False):
    """
    Generate the KL basis from a set of reference images. Always returns the full KL basis
    This is pretty straight-up borrowed from pyKLIP (Wang et al. 2015)
    Args:
        references: Nimg x Npix flattened array of reference images
        kl_max [None]: number of KL modes to return. Default: all
        return_evecs [False]: if True, return the eigenvectors of the covar matrix
        return_evals [False]: if true, return the eigenvalues of the eigenvectorsnumbasis = np.clip(numbasis - 1, 0, tot_basis-1)
    Returns:
        kl_basis: Full KL basis (up to kl_max modes if given; default len(references))
        evecs: covariance matrix eigenvectors, if desired
        evals: eigenvalues of the eigenvectors, if desired
    """
    nrefs, npix = references.shape
    ref_psfs_mean_sub = references - np.expand_dims(np.nanmean(references, axis=-1), -1)
    # set nan's to 0 so that they don't contribute to the covar matrix, and don't mess up numpy
    nan_refs = np.where(np.isnan(references))
    ref_psfs_mean_sub[nan_refs] = 0

    covar = np.cov(ref_psfs_mean_sub)

    if kl_max is None: kl_max = covar.shape[0]
    # Limit to the valid number of KL modes
    tot_basis = covar.shape[0] # max number of KL modes
    numbasis = np.clip(kl_max-1, 0, tot_basis-1)
    max_basis = np.max(numbasis)+1

    # calculate eigenvales and eigenvectors
    evals, evecs = la.eigh(covar, eigvals=(tot_basis - max_basis, tot_basis-1))

    # reverse the order
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F')
    
    # check for negative eigenvalues
    check_nans = np.any(evals<=0)
    if check_nans is True:
        neg_evals = (np.where(evals <= 0))[0]
        kl_basis[:,neg_evals] = 0
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs).T/np.sqrt(evals[:,None])
    kl_basis = kl_basis * (1./np.sqrt(npix-1))
    
    # assemble objects to return
    return_objs = [kl_basis]
    if return_evecs is True:
        return_objs.append(evecs)
    if return_evals is True:
        return_objs.append(evals)
    if len(return_objs) == 1:
        return_objs = return_objs[0]
    return return_objs
    
def klip_subtract_with_basis_slower(img_flat, kl_basis, n_bases=None, double_project=False):
    """
    If you already have the KL basis, do the klip subtraction
    Arguments:
      img_flat: (Nparam x) Npix flattened image - pixel axis must be last
      kl_basis: (Nbases x ) Nklip x Npix array of KL basis vectors (possibly more than one basis)
      n_bases [None]: list of integers for Kmax, return one image per Kmax in n_bases.
          If None, use full basis
      double_project: apply KL twice (useful for some FMMF cases)
    Return:
      kl_sub: array with same shape as input image, after KL PSF subtraction. The KL basis axis is second-to-last (before image pixels)
    """
    """
    New idea: take arbitrary input shape where the last axes is the pixels, turn it into 
    whatever x Npix, do KLIP, and then return 
    """
    # reshape input
    # math assumes img_flat has 2 dimensions, with the last dimension being the image pixels
    leading_shape = img_flat.shape[:-1]
    if img_flat.ndim == 1:
        # add an empty axis to the front
        img_flat = np.expand_dims(img_flat, 0)
        flat_shape = img_flat.shape
    elif img_flat.ndim > 2:
        flat_shape = (reduce(lambda x,y: x*y, img_flat.shape[:-1]), img_flat.shape[-1])
    else:
        flat_shape = img_flat.shape
    orig_shape = np.copy(img_flat.shape)
    img_flat = img_flat.reshape(flat_shape)    

    kl_basis = np.asarray(kl_basis)
    img_flat_mean_sub = img_flat - np.nanmean(img_flat, axis=-1, keepdims=True)

    # make sure n_bases is iterable
    if n_bases is None:
        n_bases = [len(kl_basis)+1]
    if hasattr(n_bases, '__getitem__') is False:
        n_bases = [n_bases]
        
    # project the image onto the PSF basis
    psf_projection = np.array([np.dot(np.dot(img_flat_mean_sub, kl_basis[:n_bases[i]].T),
                                      kl_basis[:n_bases[i]])
                               for i in range(len(n_bases))])
    # subtract the projection from the image
    kl_sub = img_flat_mean_sub - psf_projection

    # project twice for faster forward modeling, if desired
    if double_project is True:
        # do mean subtraction again even though it should already be mean 0
        kl_sub -= np.nanmean(kl_sub, axis=-1, keepdims=True)
        # project again onto the subtracted image, and re-subtract
        kl_sub -= np.array([np.dot(np.dot(kl_sub[i], kl_basis[:n_bases[i]].T),
                                   kl_basis[:n_bases[i]])
                            for i in range(len(n_bases))])

    # put it back in the original shape
    # the KL axis ends up in front. We want to put it just before the image axis
    kl_sub = np.rollaxis(kl_sub, 0, -1)
    new_shape = list(leading_shape) + list(kl_sub.shape[-2:])
    return np.squeeze(kl_sub.reshape(new_shape))
    #kl_sub = np.reshape(kl_sub, new_shape)
    #return np.squeeze(kl_sub)


def klip_subtract_with_basis(img_flat, kl_basis, n_bases=None):
    """
    If you already have the KL basis, do the klip subtraction
    Arguments:
      img_flat: (Nparam x) Npix flattened image - pixel axis must be last
      kl_basis: (Nbases x ) Nklip x Npix array of KL basis vectors (possibly more than one basis)
      n_bases [None]: list of integers for Kmax, return one image per Kmax in n_bases.
          If None, use full basis
      double_project: apply KL twice (useful for some FMMF cases)
    Return:
      kl_sub: array with same shape as input image, after KL PSF subtraction.
        The KL basis axis is second-to-last (before image pixels)
    """

    # make sure n_bases is iterable
    if n_bases is None:
        n_bases = np.array([len(kl_basis)])
    elif hasattr(n_bases, '__getitem__') is False:
        n_bases = np.array([n_bases])
    else:  # explicitly cast to array
        n_bases = np.array(n_bases)
    # make sure the upper bound on n_bases is legal
    # also, this applies the -1 shift needed for proper indexing
    tot_basis = len(kl_basis)-1
    n_bases = np.clip(n_bases - 1, 0, tot_basis-1)

    # reshape input
    # math assumes img_flat has 2 dimensions, with the last dimension being
    # the image pixels
    leading_shape = img_flat.shape[:-1]
    if img_flat.ndim == 1:
        # add an empty axis to the front
        img_flat = np.expand_dims(img_flat, 0)
        flat_shape = img_flat.shape
    elif img_flat.ndim > 2:
        flat_shape = (reduce(lambda x, y: x*y, img_flat.shape[:-1]),
                      img_flat.shape[-1])
    else:
        flat_shape = img_flat.shape
    # orig_shape = np.copy(img_flat.shape)
    img_flat = img_flat.reshape(flat_shape)

    kl_basis = np.asarray(kl_basis)
    img_flat_mean_sub = img_flat - np.nanmean(img_flat, axis=-1, keepdims=True)

    # CAREFUL! tiling depends on dimensionality of img_flat_mean_sub
    imgs_tiled = np.tile(img_flat_mean_sub[..., None, :],
                         [1 for i in img_flat_mean_sub.shape[:-1]] + [len(kl_basis), 1])

    coeffs = np.dot(imgs_tiled, kl_basis.T)
    lower_tri = np.tril(np.ones((kl_basis.shape[0], kl_basis.shape[0])))
    coeffs = coeffs * lower_tri

    klip_psf = np.dot(coeffs[..., n_bases, :], kl_basis)

    # subtract the projection from the image
    kl_sub = imgs_tiled[..., n_bases, :] - klip_psf
    # put it back in the original shape
    # the KL axis ends up in front. We want it just before the image axis
    new_shape = list(leading_shape) + list(kl_sub.shape[-2:])
    return np.squeeze(kl_sub.reshape(new_shape))
