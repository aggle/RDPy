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

    # target image (optional)
    @property
    def target(self):
        """
        Target image (cube?) for RDI reduction.
        """
        return self._target
    @target.setter
    def target(self, newval, sort_refs=False):
        """
        If sort_refs is true, the reference cube gets sorted in increasing order
        of distance from the target
        """
        self._target = newval
        if sort_refs is True:
            new_cube_order = sort_squared_distance(newval, self.cube)
            self.cube = self.cube[new_cube_order]
                 

    # Forward modeling and Matched Filter
    @property
    def kl_basis(self):
        return self._kl_basis
    @kl_basis.setter
    def kl_basis(self, newval):
        self.kl_basis = newval

    
    @property
    def matched_filter(self):
        # Npix x (Nx x Ny) datacube of matched filters for each position
        return self._matched_filter
    @matched_filter.setter
    def matched_filter(self, newval):
        self.matched_filter = newval

    
        
    # Misc
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
    targ: Npix array (can also be 2-D image)
    references: Nimg x Npix array (can also be 3-D cube)
    Returns:
    sorted indices of the reference images, closest first
    i.e. references[sorted_indices] gives you the images in order of distance
    """
    targ = targ.ravel() # 1-d
    references = references.reshape(references.shape[0],
                                    reduce(lambda x,y: x*y, references.shape[1:]))
                                    
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
    img_pix, psf_pix = get_stamp_coordinates(center, psf.shape[0],psf.shape[1], img.shape)

    injected_img = img.copy()
    injected_img[img_pix[0], img_pix[1]] += psf[psf_pix[0], psf_pix[1]]

    return injected_img
    
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
#    injection_corner = center - psf_rad
    # how much of the PSF will fit?
#    injection_lb = np.max([(0,0), center - psf_rad], axis=0)
#    injection_ub = np.min([center + psf_rad, img.shape], axis=0)
#    injection_lb = np.array([np.int(i) for i in injection_lb])
#    injection_ub = np.array([np.int(i) for i in injection_ub])
#    injection_dim = injection_ub - injection_lb
#    injection_pix = np.ogrid[injection_lb[0]:injection_ub[0],
#                             injection_lb[1]:injection_ub[1]]
    #injection_pix = np.ogrid[injection_corner[0]:injection_corner[0]+psf.shape[-2],
    #                         injection_corner[1]:injection_corner[1]+psf.shape[-1]]
    injection_pix, psf_pix = get_stamp_coordinates(center, psf.shape[0], psf.shape[1], img.shape)
    
    # normalized full-image PSF in case you want it later
    injection_psf = np.zeros(img.shape)
    cut_psf = psf[psf_pix[0],psf_pix[1]]
    injection_psf[injection_pix[0], injection_pix[1]] += cut_psf/np.nansum(psf)

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

def FM_MF(kl_basis, psf, shape):
    """
    Generate a matched filter 
    """
    pass




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

#######################################
# Image to Sky coordinate conversions #
#######################################
def image_2_seppa(coord, orientation=0, center=(40,40), pix_scale = 75):
    """
    convert from image coordinates to sep and pa
    coord: (row, col) coord in image
    orientation: clockwise angle between y in image and north
    center: image center
    pix_scale: 75 mas/pix for nicmos
    """
    coord = np.array(coord)
    center = np.array(center)
    centered_coord = coord-center
    sep = np.linalg.norm(centered_coord)*pix_scale
    pa = np.arctan(centered_coord[0]/centered_coord[1]) * 180/np.pi
    pa -= orientation
    return (sep, pa)

def seppa_2_image(sep, pa, orientation=0, center=(40,40), pix_scale = 75,
                  return_raveled=False, shape=None):
    """
    convert from separation and position angle to image coordinates
    Arguments:
      sep: separation in mas
      pa: on-sky position angle in degrees
      orientation: clockwise angle between y in image and north
      center: ([40,40]) row, col image center
      pix_scale: 75 mas/pix for nicmos
      return_raveled: [True] instead of row,col coord, return the coordinate for a linearized array. 
        If True, you must also provide the image shape
    shape: (nrow, ncol) tuple of the image shape
    Returns:
        Nearest integer of the coordinate, either 1-D or 2-D depending on value of return_raveled
    """
    # convert all angles from degrees to radians
    center = np.array(center)
    pa = pa*np.pi/180
    orientation = orientation*np.pi/180
    tot_ang = pa+orientation
    row = sep*np.cos(tot_ang)/pix_scale + center[0]
    col = sep*np.sin(tot_ang)/pix_scale + center[1]
    row, col = (np.int(np.round(row)), np.int(np.round(col)))
    if return_raveled is True:
        ind = np.ravel_multi_index((row, col), shape, mode='clip')
        return ind
    return row, col


def klip_subtract_with_basis(img_flat, kl_basis):
    """
    If you already have the KL basis, do the klip subtraction
    Arguments:
      img_flat: Npix 1-D flattened image 
      kl_basis: Nklip x Npix array of KL basis vectors
    Return:
      subtracted_img: array with same shape as input image, after KL PSF subtraction
    """
    img_flat_mean_sub = img_flat - np.nanmean(img_flat)
    return img_flat_mean_sub - np.nansum([np.dot(img_flat_mean_sub, k)*k for k in kl_basis], axis=0)
    

def get_stamp_coordinates(center, drow, dcol, imshape):
    """
    get pixel coordinates for a stamp with width dcol, height drow, and center `center` embedded
    in an image of dimensions imshape
    Arguments:
        center: (row, col) center of the stamp
        drow: height of stamp
        dcol: width of stamp
        imshape: total size of image the stamp is a part of
    Returns:
        img_coords: the stamp indices for the full image array (img[img_coords_for_stamp])
        stamp_coords: the stamp indices for selecting the part of the stamp that goes in the image (stamp[stamp_coords])
    """
    colrad = np.int(np.floor(dcol))/2
    rowrad = np.int(np.floor(drow))/2
    rads = np.array([rowrad, colrad], dtype=np.int)
    center = np.array([center[0],center[1]],dtype=np.int)
    img = np.zeros(imshape)
    stamp = np.ones((drow,dcol))
    full_stamp_coord = np.indices(stamp.shape) + center[:,None,None]  - rads[:,None,None]
    # check for out-of-bounds values
    # boundaries
    row_lb,col_lb = (0, 0)
    row_hb,col_hb = imshape
    
    rowcheck_lo, colcheck_lo = (center - rads)
    rowcheck_hi, colcheck_hi = ((imshape-center) - rads)    

    row_start, col_start = 0,0
    row_end, col_end = stamp.shape
    
    if rowcheck_lo < 0:
        row_start = -1*rowcheck_lo
    if colcheck_lo < 0:
        col_start = -1*colcheck_lo
    if rowcheck_hi < 0:
        row_end = rowcheck_hi
    if colcheck_hi < 0:
        col_end = colcheck_hi

    # pull out the selections
    img_coords = full_stamp_coord[:,row_start:row_end,col_start:col_end]    
    stamp_coords = np.indices(stamp.shape)[:,row_start:row_end,col_start:col_end]
    return (img_coords, stamp_coords)

def fmmf_throughput_correction(psfs, kl_basis):
    """
    Calculate the normalization aka throughput correction factor for the matched filter, to get flux out
    Arguments:
        psfs: the flattened psf models
        kl_basis: the KL basis for projection
    """
    mf_norm_1 = np.linalg.norm(psfs, axis=-1)**2
    mf_norm_2 = np.nansum([np.dot(psfs, k)**2 for k in kl_basis], axis=0)
    mf_norm = mf_norm_1 - mf_norm_2
    return mf_norm

def generate_matched_filter(psf, kl_basis, imshape, kl_pix_coordinates=None):
    """
    generate a matched filter with a model of the PSF. For now we assume that the KL basis covers the entire image
    Arguments:
        psf: square postage stamp of the PSF
        kl_basis: karhunen-loeve basis for PC projection
        imshape: Nrows x Ncols
        kl_pix_coordinates: the pixel coordinates covered by the KL basis. if None, assumed to cover the whole image
    Returns:
        MF: cube of flattened matched filters. The first index tells you what pixel in the
            image it corresponds to, through np.unravel_index(i, imshape)
    """
    mf_template_cube = np.zeros((imshape[0]*imshape[1],imshape[0],imshape[1]))
    mf_pickout_cube = np.zeros_like(mf_template_cube) # this is used to pick out the PSF *after* KLIP on the PSFs
    # inject the instrument PSFs - this cannot be done on a flattened cube
    for i in range(len(mf_template_cube)):
        center = np.unravel_index(i, imshape)
        mf_template_cube[i] = inject_psf(np.zeros(imshape), psf, center)
        mf_pickout_cube[i]  = inject_psf(mf_pickout_cube[i], np.ones_like(psf), center)
    # flatten so that the first index is the target location index and the second index is the image pixel index
    mf_flat_template = np.array([i.ravel() for i in mf_template_cube])
    mf_flat_pickout =  np.array([i.ravel() for i in mf_pickout_cube])
    # find the klip-modified PSFs
    mf_flat_template_klipped = np.zeros_like(mf_flat_template)
    for i in range(len(mf_flat_template_klipped)):
        template = mf_flat_template[i]
        mf_flat_template_klipped[i] = klip_subtract_with_basis(template, kl_basis)
    # throughput normalization
    mf_norm_flat = fmmf_throughput_correction(mf_flat_template, kl_basis)
    MF = mf_flat_template_klipped/mf_norm_flat
    return MF


def apply_matched_filter_to_image(image, matched_filter):
    """
    Apply a matched filter to an image. It is assumed that the image and the matched filter have already been sampled to the same resolution.
    Arguments:
        image: 2-D image
        matched_filter: Cube of flattened matched filters, one MF per pixel (flattened along the second axis)
    Returns:
        mf_map: 2-D image where the matched filter has been applied
    """
    flat_image = image.ravel()
    mf_map = np.zeros_like(flat_image)
    for i in range(np.size(mf_map.shape)):
        mf_map[i] = np.dot(matched_filter[i], flat_image)
    return mf_map.reshape(image.shape)
                           
