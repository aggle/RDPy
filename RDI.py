import abc

import sys
import os
import numpy as np
import numpy.fft as fft

import pandas as pd

from astropy.io import fits
from astropy import units

import scipy.linalg as la
import scipy.ndimage as ndimage
from scipy.stats import t

class ReferenceCube(object):
    """
    Things this class can/should do:
        - know the number of reference images and the image shape
        - calcualte the covariance matrix
        - return a subset of the reference cube
        - return a subset of the covariance matrix
        - calculate the distance of an image from the reference cube entries
    """
    def __init__(self, cube, region_mask=None):
        """
        Initialize the reference cube class starting with a cube of stacked reference images
        Initialization entails:
            - get the number of reference images
            - get the shape of the images
            - calculate the covariance matrix
        """
        self.cube = cube
        self.Nref = cube.shape[0]
        self.imshape = cube.shape[1:]
        self.region = region_mask

    ###############################
    # Instantiate required fields #
    ###############################
    # basic cube properties
    @property
    def cube(self):
        return self._cube
    @cube.setter
    def cube(self, newcube):
        self._cube = newcube
        # new cube - reset all the class attributes
        self.Nref = np.shape(newcube)[0]
        self.imshape = np.shape(newcube)[1:]
        self.flat_cube = newcube.reshape(self.Nref, reduce(lambda x,y: x*y, self.imshape)) # may already be flat?
        self.reference_indices = list(np.ogrid[:self.cube.shape[0]])
    @property
    def Nref(self):
        return self._Nref
    @Nref.setter
    def Nref(self, newNref):
        self._Nref = newNref
    @property
    def imshape(self):
        return self._imshape
    @imshape.setter
    def imshape(self, newshape):
        self._imshape = newshape
    @property
    def reference_indices(self):
        return self._reference_indices[:]
    @reference_indices.setter
    def reference_indices(self, newval):
        self._reference_indices = list(newval)
    
    # region mask
    @property
    def region(self):
        return self._region
    @region.setter
    def region(self, newregion):
        # once you have a mask, you should immediately generate the flat mask indixes
        if newregion is None:
            self._region = np.ones(self.imshape)
        else:
            self._region = newregion
        self.flat_region_ind = np.where(np.ravel(self._region)==1)[0]
    @property
    def flat_region_ind(self):
        return self._flat_region_ind
    @flat_region_ind.setter
    def flat_region_ind(self, newfri):
        # as soon as you get region indices, pull out the region
        self._flat_region_ind = newfri
        self.flat_cube_region = self.flat_cube[:,self.flat_region_ind].copy()
        # for some reason, whether you work on a copy or a view affects the answer
        self.npix_region = np.size(newfri)

    @property
    def npix_region(self):
        return self._npix_region
    @npix_region.setter
    def npix_region(self, new_npix_region):
        self._npix_region = new_npix_region
        
    # cube flattening
    @property
    def flat_cube(self):
        return self._flat_cube
    def flat_cube(self, newflatcube):
        self._flat_cube = newflatcube
        # see if you can pull out a region
        try:
            self.flat_cube_region = self.flat_cube[:,self.flat_region_ind].copy()
            # for some reason, whether you work on a copy or a view affects the answer
        except AttributeError:
            # mask doesn't exist
            self.region = None

    @property
    def flat_cube_region(self):
        return self._flat_cube_region
    @flat_cube_region.setter
    def flat_cube_region(self, newfcr):
        # this is the only place you implicitly calc the covariance matrix
        self._flat_cube_region = newfcr
        self.covar_matrix = self.calc_covariance_matrix(self._flat_cube_region)

    # derived attributes
    @property
    def covar_matrix(self):
        return self._covar_matrix
    @covar_matrix.setter
    def covar_matrix(self, newcm):
        self._covar_matrix = newcm
    @property
    def corr_matrix(self):
        return self._corr_matrix
    @corr_matrix.setter
    def corr_matrix(self, newcm):
        return self._corr_matrix
        
        
    def set_mask(self, mask):
        """
        Sets a new mask
        Inputs:
            mask: a 2-D binary mask, 1=included, 0=excluded (region = im*mask)
        """
        self.region = mask


    # Reference cube operations    
    def calc_covariance_matrix(self, references):
        """
        get the covariance matrix of the cube in the selected region
        Note: suspect normalization is wrong
        Input:
            references: Nref x Npix array of flattened reference images
        Returns:
            Nref x Nref covariance matrix
        """
        refs_mean_sub = references - np.nanmean(references, axis=-1)[:,None]
        covmat = np.cov(refs_mean_sub)
        covmat *= (references.shape[-1] - 1) # undo np.cov normalization
        return covmat

    def calc_correlation_matrix(self, references):
        """
        Calculate the Pearson correlation matrix for the references
        Input:
            References: Nref x Npix array of flattened reference images
        Returns:
            Nref x Nref correlation matrix
        """
        norm_refs = references/np.nansum(references, axis=-1)[:,None]
        ref_corr = np.corrcoef(norm_refs)
        return ref_corr

    def get_cube_subset(self, indices):
        """
        return a copy of the cube subset as given by indices
        """
        return self.cube[indices]

    def get_reduced_covariance_matrix(self, targ_ix):
        """
        return a copy of the covariance matrix where the covariance with one of the
        images of the reference cube has been removed
        """
        covmat_shape = self.covar_matrix.shape
        mask = np.ones(covmat_shape)
        mask[targ_ix,:] = 0
        mask[:,targ_ix] = 0
        return self.covar_matrix[np.where(mask==1)].reshape(np.array(covmat_shape)-1)

    def get_covariance_matrix_subset(self, indices, covar_matrix=None):
        """
        return a copy of the covariance matrix with only the references indicated by the indexes
        Input:
            indices: index corresponding to the reference cube of images to include
            covar_matrix: (optional) covariance matrix, if None use the object's covariance matrix
        Returns:
            new covariance matrix with only a subset of the selected images, shape is (len(indices),len(indices))
        """
        if covar_matrix is None:
            covar_matrix = self.covar_matrix
        return covar_matrix[np.meshgrid(indices, indices)].copy()

############################
## PSF Injection ###########
############################
class PSF_cutout(np.ndarray):
    """
    This object acts like an array except it ALSO stores the fractional flux contained in each slice.
    See PSF_cutout.__new__()    
    """
    def __new__(cls, input_array, frac_flux=None):
        obj = np.asarray(input_array).view(cls)
        obj.frac_flux = frac_flux
        return obj
    
    def __array_finalize__(self, obj):
        """ Default values go here """
        if obj is None: return
        self.frac_flux = getattr(obj, 'frac_flux', None)

    def __array_wrap__(self, out_arr, context=None):
        """For ufuncs"""
        return np.ndarray.__array_wrap__(self, out_arr, context)    
    
    @property
    def frac_flux(self):
        return self._frac_flux
    @frac_flux.setter
    def frac_flux(self, newval):
        self._frac_flux = newval
        
############################
## Other useful functions ##
############################
def sort_squared_distance(targ, references):
    """
    Calculate the euclidean distance between a target image and a set of reference images
    Definition of distance: sqrt( sum( (im1 - im2 )**2 ) )
    where the subtraction and squaring are pixelwise
    Gives back the indices of this set of references, sorted by distance and taking care of the case
    where the target itself is in the set of references
    Input:
    targ: Npix array
    references: Nimg x Npix array
    Returns:
    sorted indices of the reference images, closest first
    i.e. references[sorted_indices] gives you the images in order of distance
    """
    targ_norm = targ/np.nansum(targ)
    ref_norm = references/np.nansum(references, axis=-1)[:,None]
    ref_norm = ref_norm[None, :] # make sure dimensionality works out in all cases
    # np.squeeze gets rid of the empty dimension we added above
    dist = np.squeeze(np.sqrt( np.nansum( (targ_norm - ref_norm)**2, axis=-1) ))
    # if dist == 0 for an image, it's clearly the same image and should be excluded
    # use the fact that distance is positive definite so the same image will be closest
    sorted_dist = np.sort(dist)
    start_index = 0
    try:
        if np.where(sorted_dist==0): # target is in references
            start_index = np.max(np.squeeze(np.where(sorted_dist==0)))+1
    except: # target is not in references, do nothing
        pass
    sorted_image_indices = np.argsort(dist)[start_index:]
    return sorted_image_indices


def make_annular_mask(rad_range, phi_range, center, shape):
    """
    Make a mask that covers an annular section
    rad_range: tuple of (inner radius, outer radius)
    phi_range: tuple of (phi0, phi1) (0,2*pi)
    center: row, col center of the image (where rad is measured from)
    shape: image shape tuple
    """
    grid = np.mgrid[:np.float(shape[0]),:np.float(shape[1])]
    grid = np.rollaxis(np.rollaxis(grid, 0, grid.ndim) - center, -1, 0)
    rad2D = np.linalg.norm(grid, axis=0)
    phi2D = np.arctan2(grid[0],grid[1]) + np.pi # 0 to 2*pi
    mask = np.zeros(shape)
    if phi_range[0] <= phi_range[1]:
        mask[np.where(((rad2D >= rad_range[0]) & (rad2D < rad_range[1])) & 
                      ((phi2D >= phi_range[0]) & (phi2D <= phi_range[1])))] = 1
    elif phi_range[0] > phi_range[1]:
        mask[np.where(((rad2D >= rad_range[0]) & (rad2D < rad_range[1])) & 
                      ((phi2D >= phi_range[0]) | (phi2D <= phi_range[1])))] = 1
    else:
        pass # should probably throw an exception or something
    return mask

def make_circular_mask(center, rad, shape):
    """
    Make a circular mask around a point, cutting it off for points outside the image
    Inputs: 
        center: (y,x) center of the mask
        rad: radius of the circular mask
        shape: shape of the image
    Returns:
        mask: 2D mask where 1 is in the mask and 0 is out of the mask (multiple it by the image)
    """
    grid = np.mgrid[:np.float(shape[0]),:np.float(shape[1])]
    centered_grid = np.rollaxis(np.rollaxis(grid, 0, grid.ndim) - center, -1, 0)
    rad2D = np.linalg.norm(centered_grid, axis=0)
    mask = np.zeros(shape)
    mask[np.where(rad2D <= rad)] = 1
    return mask

def inject_psf(img, psf, center):
    """
    Inject a PSF into an image by adding it to the existing pixels
    This handles psfs close to the edge of the image by first padding the image, injecting the psf,
    and then cutting down to the original image
    """
    center = np.array(center)
    psf_rad = np.int(np.floor(psf.shape[0]/2.)) # assume psf is symmetric
    # pad the image by psf_rad+1
    padded_img = np.pad(img, psf_rad+1, mode='constant', constant_values = 0)
    center += psf_rad
#    return padded_img
    injected_psf = inject_psfs(padded_img, psf, center)
    return injected_psf[psf_rad+1:-(psf_rad+1),psf_rad+1:-(psf_rad+1)]

    
def inject_psfs(img, psf, center, scale_flux=None, return_psf=False, subtract_mean=False, return_flat=False):
    """
    Inject a PSF into an image at a location given by center. Optional: scale PSF
    Input:
        img: 2-D img or 3-D cube. Last two dimensions define an img (i.e. [(Nimg,)Nx,Ny])
        psf: 2-D img or 3-D cube, smaller than or equal to img in size. If cube, 
             must have same 1st dimension as img 
        center: center of the injection in the image
        scale_flux: multiply the PSF by this number. If this is an array,
             img and psf will be tiled to match its length
             if scale_flux is None, don't scale psf
        return_psf: (False) return an image with the PSF and zeros elsewhere
        subtract_mean: (False) mean-subtract before returning
        return_flat: (False) flatten the array along the pixels axis before returning
    Returns:
       injected_img: 2-D image or 3D cube with the injected PSF(s)
       injection_psf: (if return_psf=True) 2-D normalized PSF full image
    """
    if scale_flux is None:
        scale_flux = 1
    scale_flux = np.array(scale_flux)
    
    # get the right dimensions
    img_tiled = np.tile(img, (np.size(scale_flux),1,1))
    psf_tiled = np.tile(psf, (np.size(scale_flux),1,1))

    # get the injection pixels
    psf_rad =  np.array([np.int(np.floor(i/2.)) for i in psf.shape[-2:]])
    injection_corner = center - psf_rad
    # how much of the PSF will fit?
    injection_lb = np.max([(0,0), center - psf_rad], axis=0)
    injection_ub = np.min([center + psf_rad, img.shape], axis=0)
    injection_dim = injection_ub - injection_lb
    injection_pix = np.ogrid[injection_lb[0]:injection_ub[0],
                             injection_lb[1]:injection_ub[1]]
    #injection_pix = np.ogrid[injection_corner[0]:injection_corner[0]+psf.shape[-2],
    #                         injection_corner[1]:injection_corner[1]+psf.shape[-1]]

    # normalized full-image PSF in case you want it later
    injection_psf = np.zeros(img.shape)
    injection_psf[injection_pix[0], injection_pix[1]] += psf/np.nansum(psf)

    # add the scaled psfs
    injection_img = np.zeros(img_tiled.shape)
    injection_img[:,injection_pix[0], injection_pix[1]] += (psf_tiled.T*scale_flux).T
    # get rid of extra dimensions before returning
    #full_injection = np.squeeze(injection_img + img_tiled)
    full_injection = injection_img + img_tiled
    if subtract_mean is True:
        full_injection = (full_injection.T - np.nanmean(np.nanmean(full_injection, axis=-1),axis=-1)).T
        injection_psf = injection_psf - np.nanmean(injection_psf)
    if return_flat is True:
        shape = full_injection.shape
        if full_injection.ndim == 2:
            full_injection = np.ravel(full_injection)
        else:
            full_injection = np.reshape(full_injection, (shape[0],reduce(lambda x,y: x*y, shape[1:])))
    if return_psf is True:
        return full_injection, injection_psf
    return np.squeeze(full_injection)

def inject_region(flat_img, flat_psf, scaling=1, subtract_mean=False):
    """
    Inject a flattened psf into a flattened region with some scaling.
    Input:
        flat_img: 1-d array of the region of interest
        flat_psf: 1-d array of the psf in the region at the correct location
        scaling: (1) multiply the PSF by this number. If this is an array, 
                 img and psf will be tiled to match its length.
        subtract_mean: (False) mean-subtract images before returning
    """
    scaling = np.array(scaling)
    # get the right dimensions
    flat_img_tiled = np.tile(flat_img, (np.size(scaling),1))
    flat_psf_tiled = np.tile(flat_psf, (np.size(scaling),1))

    # assume the PSF is already properly aligned
    scaled_psf_tiled = (flat_psf_tiled.T*scaling).T
    
    injected_flat_img = np.squeeze(flat_img_tiled + scaled_psf_tiled)
    if subtract_mean == True:
        injected_flat_img = (injected_flat_img.T - np.nanmean(injected_flat_img, axis=-1)).T
    return injected_flat_img

def generate_mean_subtracted_fake_injections(flat_img, flat_psf, scaling=1):
    """
    flat_img: 2-D image to inject into
    flat_psf: 2-D psf 
    """
    scaling = np.array(scaling)
    # get the right dimensions
    flat_img_tiled = np.tile(flat_img, (np.size(scaling),1))
    flat_psf_tiled = np.tile(flat_psf, (np.size(scaling),1))

    # assume the PSF is already properly aligned
    scaled_psf_tiled = (flat_psf_tiled.T*scaling).T
    
    injected_flat_img = np.squeeze(flat_img_tiled + scaled_psf_tiled)
    injected_flat_img = (injected_flat_img.T - np.nanmean(injected_flat_img, axis=-1)).T
    return injected_flat_img


def make_image_from_region(region, indices, shape):
    """
    put the flattened region back into an image
    Input:
        region: [Nx,[Ny...]] x Npix array (any shape as long as the last dim is the pixels)
        indices: Npix array of flattened pixel coordinates 
                 corresponding to the region
        shape: image shape
    """
    oldshape = np.copy(region.shape)
    img = np.ravel(np.zeros(shape))
    # handle the case of region being a 2D array by extending the img axes
    if region.ndim > 1:
        # assume last dimension is the pixel
        region = np.reshape(region, (reduce(lambda x,y: x*y, oldshape[:-1]), oldshape[-1]))
        img = np.tile(img, (region.shape[0], 1))
    else:
        img = img[None,:]
    # fill in the image
    img[:,indices] = region
    # reshape and get rid of extra axes, if any
    img = np.squeeze(img.reshape(list(oldshape[:-1])+list(shape)))
                            
    return img


####################
# Forward modeling #
####################

def FM_flux(kl_sub_img, psf, kl_basis):
    """
    Use forward modeling to determine the flux from a point source whose location is given by psf
    Derivation can be found in Pueyo 2016, Appendix C
    len(kl_basis) must match the number of KL modes that went into generating kl_sub_img
    Input:
        kl_sub_img: Npix array of a klip-subtracted image with a point source inside
        psf: Npix array of the normalized PSF at a particular location
        kl_basis: KL basis vectors up to as many as you want to include (must be same as number
            used to generate kl_sub_img)
    returns:
        fm_flux: Nimg x len(kl_basis) array of fluxes for each klip-subtracted image provided
    """
    # first take care of just one image
    # make sure the psf is normalized
    psf = psf/np.nansum(psf)
    numerator = np.dot(kl_sub_img, psf)
    denominator = np.dot(psf,psf) - np.nansum(np.dot(kl_basis, psf)**2)
    fm_flux = numerator/denominator
    return fm_flux

def FM_noise(bgnd, psf, kl_basis):
    """
    Use forward modeling to determine the noise in the image
    Input:
        bgnd: Npix array pre-subtraction, with no point source
        psf: Npix array of the model PSF
        kl_basis: N_kl x Npix array of the KL basis vectors
    Returns:
        noise
    """
    psf = psf/np.nansum(psf)
    numerator = np.dot(bgnd, psf) - np.nansum(np.dot(kl_basis, bgnd)*np.dot(kl_basis,psf))
    denominator = np.linalg.norm(psf)**2 - np.nansum(np.dot(kl_basis, psf)**2)
    return numerator/denominator

##########
# My own klip copy that returns both the KL basis and the fake injections
##########
def klip_math(sci, ref_psfs, numbasis, covar_psfs=None, PSFarea_tobeklipped=None, PSFsarea_forklipping=None, return_basis=False, return_basis_and_eig=False):
    """
    Helper function for KLIP that does the linear algebra
    
    Args:
        sci: array of length p containing the science data
        ref_psfs: N x p array of the N reference PSFs that 
                  characterizes the PSF of the p pixels
        numbasis: number of KLIP basis vectors to use (can be an int or an array of ints of length b)
        covar_psfs: covariance matrix of reference psfs passed in so you don't have to calculate it here
        PSFarea_tobeklipped: Corresponds to sci but with the fake planets only. It is the section to be klipped. Can be a cube.
        PSFsarea_forklipping: Corresponds to ref_psfs but with the fake planets only. It is the set of sections used for
                              the klipping. ie from which the modes are calculated.
        return_basis: If true, return KL basis vectors (used when onesegment==True)
        return_basis_and_eig: If true, return KL basis vectors as well as the eigenvalues and eigenvectors of the
                                covariance matrix. Used for KLIP Forward Modelling of Laurent Pueyo.

    Returns:
        sub_img_rows_selected: array of shape (p,b) that is the PSF subtracted data for each of the b KLIP basis
                               cutoffs. If numbasis was an int, then sub_img_row_selected is just an array of length p
        KL_basis: array of shape (max(numbasis),p). Only if return_basis or return_basis_and_eig is True.
        evals: Eigenvalues of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
        evecs: Eigenvectors of the covariance matrix. The covariance matrix is assumed NOT to be normalized by (p-1).
                Only if return_basis_and_eig is True.
    """
    return_objs = {} # container to collect  objects to return
    
    # for the science image, subtract the mean and mask bad pixels
    sci_mean_sub = sci - np.nanmean(sci)
    # sci_nanpix = np.where(np.isnan(sci_mean_sub))
    # sci_mean_sub[sci_nanpix] = 0

    # do the same for the reference PSFs
    # playing some tricks to vectorize the subtraction
    ref_psfs_mean_sub = ref_psfs - np.nanmean(ref_psfs, axis=1)[:, None]
    ref_psfs_mean_sub[np.where(np.isnan(ref_psfs_mean_sub))] = 0

    # Replace the nans of the PSFs (~fake planet) area by zeros.
    # We don't want to subtract the mean here. Well at least JB thinks so...
    if PSFsarea_forklipping is not None:
        PSFsarea_forklipping[np.where(np.isnan(PSFsarea_forklipping))] = 0

    # calculate the covariance matrix for the reference PSFs
    # note that numpy.cov normalizes by p-1 to get the NxN covariance matrix
    # we have to correct for that a few lines down when consturcting the KL
    # vectors since that's not part of the equation in the KLIP paper
    if covar_psfs is None:
        covar_psfs = np.cov(ref_psfs_mean_sub)

    # maximum number of KL modes
    tot_basis = covar_psfs.shape[0]

    # only pick numbasis requested that are valid. We can't compute more KL basis than there are reference PSFs
    # do numbasis - 1 for ease of indexing since index 0 is using 1 KL basis vector
    numbasis = np.clip(numbasis - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
    max_basis = np.max(numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate

    # calculate eigenvalues and eigenvectors of covariance matrix, but only the ones we need (up to max basis)
    evals, evecs = la.eigh(covar_psfs, eigvals=(tot_basis-max_basis, tot_basis-1))

    # scipy.linalg.eigh spits out the eigenvalues/vectors smallest first so we need to reverse
    # we're going to recopy them to hopefully improve caching when doing matrix multiplication
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication

    # check if there are negative eignevalues as they will cause NaNs later that we have to remove
    # the eigenvalues are ordered smallest to largest
    check_nans = np.any(evals <= 0)
    if check_nans: # keep an index of the negative eignevalues for future reference if there are any
        neg_evals = np.squeeze(np.where(evals <= 0))

    # calculate the KL basis vectors
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs)
    # JB question: Why is there this [None, :]? (It adds an empty first dimension)
    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(sci) - 1)))[None, :]  #multiply a value for each row

    # sort to KL basis in descending order (largest first)
    # kl_basis = kl_basis[:,eig_args_all]

    # duplicate science image by the max_basis to do simultaneous calculation for different k_KLIP
    sci_mean_sub_rows = np.tile(sci_mean_sub, (max_basis, 1))
    sci_rows_selected = np.tile(sci_mean_sub, (np.size(numbasis), 1)) # this is the output image which has less rows

    # Do the same for the PFSs (fake planet)
    if PSFarea_tobeklipped is not None:
        # JA edits
        # old
        #PSFarea_tobeklipped_rows = np.tile(PSFarea_tobeklipped, (max_basis, 1))
        #PSFarea_tobeklipped_rows_selected = np.tile(PSFarea_tobeklipped, (np.size(numbasis), 1)) # this is the output image which has less rows
        # this version allows a cube of fake planets to be passed
        # by rolling the image index axis to the front (0), proper array broadcasting is maintained for the linear algebra
        # unused axes will be removed before returning, so the original shape is maintained
        # these changes should be transparent to the user
        PSFarea_tobeklipped_rows = np.tile(PSFarea_tobeklipped, (max_basis, 1, 1))
        PSFarea_tobeklipped_rows = np.rollaxis(PSFarea_tobeklipped_rows, 1, 0)
        PSFarea_tobeklipped_rows_selected = np.tile(PSFarea_tobeklipped, (np.size(numbasis), 1, 1)) 
        PSFarea_tobeklipped_rows_selected = np.rollaxis(PSFarea_tobeklipped_rows_selected, 1, 0)


    # bad pixel mask
    # do it first for the image we're just doing computations on but don't care about the output
    sci_nanpix = np.where(np.isnan(sci_mean_sub_rows))
    sci_mean_sub_rows[sci_nanpix] = 0
    # now do it for the output image
    sci_nanpix = np.where(np.isnan(sci_rows_selected))
    sci_rows_selected[sci_nanpix] = 0

    # Do the same for the PFSs (fake planet)
    if PSFarea_tobeklipped is not None:
        PSFarea_tobeklipped_rows[np.where(np.isnan(PSFarea_tobeklipped_rows))] = 0
        solePSFs_nanpix = np.where(np.isnan(PSFarea_tobeklipped_rows_selected))
        PSFarea_tobeklipped_rows_selected[solePSFs_nanpix] = 0

    # do the KLIP equation, but now all the different k_KLIP simultaneously
    # calculate the inner product of science image with each of the different kl_basis vectors
    # TODO: can we optimize this so it doesn't have to multiply all the rows because in the next lines we only select some of them
    inner_products = np.dot(sci_mean_sub_rows, np.require(kl_basis, requirements=['F']))
    # select the KLIP modes we want for each level of KLIP by multiplying by lower diagonal matrix
    lower_tri = np.tril(np.ones([max_basis, max_basis]))
    inner_products = inner_products * lower_tri
    # if there are NaNs due to negative eigenvalues, make sure they don't mess up the matrix multiplicatoin
    # by setting the appropriate values to zero
    if check_nans:
        needs_to_be_zeroed = np.where(lower_tri == 0)
        inner_products[needs_to_be_zeroed] = 0
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        kl_basis[:, neg_evals] = 0
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)
        # for KLIP PSFs that use so many KL modes that they become nans, we have to put nan's back in those
        badbasis = np.where(numbasis >= np.min(neg_evals)) #use basis with negative eignevalues
        klip_psf[badbasis[0], :] = np.nan
    else:
        # make a KLIP PSF for each amount of klip basis, but only for the amounts of klip basis we actually output
        klip_psf = np.dot(inner_products[numbasis,:], kl_basis.T)

    # make subtracted image for each number of klip basis
    sub_img_rows_selected = sci_rows_selected - klip_psf

    # restore NaNs
    sub_img_rows_selected[sci_nanpix] = np.nan
    # add it to the returned objects dictionary
    # need to flip them so the output is shaped (p,b) for sci img
    return_objs['klipped_sci'] = sub_img_rows_selected.transpose()

    # Apply klip similarly but this time on the sole PSFs (The fake planet only)
    # Note that we use the same KL basis as before. Just the inner product changes.
    if PSFarea_tobeklipped is not None:
        inner_products_solePSFs = np.dot(PSFarea_tobeklipped_rows, np.require(kl_basis, requirements=['F']))
        inner_products_solePSFs = inner_products_solePSFs * np.tril(np.ones([max_basis, max_basis]))
        klip_solePSFs = np.dot(inner_products_solePSFs[:,numbasis,:], kl_basis.T)
        PSFarea_tobeklipped_rows_selected = PSFarea_tobeklipped_rows_selected - klip_solePSFs
        PSFarea_tobeklipped_rows_selected[solePSFs_nanpix] = np.nan

        # need to flip them so the output is shaped (p,b) for sci img
        # and (nfake,p,b) for fakes (or just (p,b) if only one fake PSF was passed)
        return_objs['klipped_fakePSFs'] = np.squeeze(np.rollaxis(PSFarea_tobeklipped_rows_selected.transpose(),-1,0))
        #return sub_img_rows_selected.transpose(), np.squeeze(np.rollaxis(PSFarea_tobeklipped_rows_selected.transpose(),-1,0))

    if return_basis is True:
        return_objs['kl_basis'] = kl_basis.transpose()
        #return sub_img_rows_selected.transpose(), kl_basis.transpose()
    elif return_basis_and_eig is True:
        return_objs['kl_basis'] = kl_basis.transpose()
        return_objs['eig'] = (evals*(np.size(sci-1)),evecs)
        #return sub_img_rows_selected.transpose(), kl_basis.transpose(),evals*(np.size(sci)-1), evecs
    else:
        pass
        #return sub_img_rows_selected.transpose()
    return return_objs

