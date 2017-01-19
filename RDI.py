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

from functools import reduce

import utils
import MatchedFilter as MF

class ReferenceCube(object):
    """
    Things this class can/should do:
        - know the number of reference images and the image shape
        - calcualte the covariance matrix
        - return a subset of the reference cube
        - return a subset of the covariance matrix
        - calculate the distance of an image from the reference cube entries
    """
    def __init__(self, cube, region_mask=None, target=None, instrument=None):
        """
        Initialize the reference cube class starting with a cube of stacked reference images
        Initialization entails:
            - get the number of reference images
            - get the shape of the images
            - calculate the covariance matrix
        Args:
          cube: Nref x (Nrow x Ncol) cube of reference images (can be flat cube)
          region_mask [None]: a binary mask of which pixels in the image to use
          target: image to be PSF-subtracted; used to sort ref images
          instrument: Instrument class object that tells ReferenceCube which instr params to use for
              things like forward-modeling the PSF
        """
        # instrument where the data comes from
        self.instrument = instrument
        # reference images
        self.cube = cube
        self.Nref = cube.shape[0]
        self.imshape = cube.shape[1:]
        # target image
        self.target = target
        self.target_mean = None
        # reduction region
        self.region = region_mask
        # KL stuff
        self.n_basis=None
        self.kl_basis = None
        self.evecs = None
        self.evals = None
        # matched filter
        self.matched_filter = None
        self.matched_filter_locations = None

    ###############################
    # Instantiate required fields #
    ###############################

    ##################
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
        self.flat_cube = newcube.reshape(self.Nref, reduce(lambda x,y: x*y, self.imshape)) 
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
    ##################

    ##################

    # target image
    @property
    def target(self):
        """
        Target image for RDI reduction, can be either 2-D or 1-D
        """
        return self._target
    @target.setter
    def target(self, newval):
        """
        If you give the RC a target, it will automatically sort the reference cube 
        in increasing order of distance from the target
        """
        try:
            self._target = newval.ravel()
        except AttributeError:
            self._target = None
            return
        # when you change the target:
        # 1. update the region
        # 2. update the cube order
        try:
            self.target_region = self._target[self.flat_region_ind]
            new_cube_order = sort_squared_distance(self._target[self.flat_region_ind],
                                                   utils.flatten_image_axes(self.cube.reshape)[:,self.flat_region_ind])
        except AttributeError:
            self.target_region = self.target[:]
            new_cube_order = sort_squared_distance(self._target, self.cube)
        self.target_mean = np.nanmean(self.target_region)
        self.cube = self.cube[new_cube_order]
        self.reference_indices = new_cube_order
        
    @property
    def target_region(self):
        """
        The target image with the region mask applied
        """
        return self._target_region
    @target_region.setter
    def target_region(self, newval):
        self._target_region = newval
    ##################

    ##################
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
    def flat_region_ind(self, newval):
        # as soon as you get region indices, pull out the region
        self._flat_region_ind = newval
        try:
            self.flat_cube_region = self.flat_cube[:,self.flat_region_ind].copy()
        except AttributeError:
            self.flat_cube_region = self.flat_cube[:]
        if self.target is not None:
            try:
                self.target_region = self.target[self.flat_region_ind].copy()
            except AttributeError:
                self.target_region = self.target[:]
            # for some reason, whether you work on a copy or a view affects the answer
        self.npix_region = np.size(newval)

    @property
    def npix_region(self):
        return self._npix_region
    @npix_region.setter
    def npix_region(self, new_npix_region):
        self._npix_region = new_npix_region
    ##################

    ##################
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
    ##################
        
    ##################
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

    ##################
    # KL basis components 
    @property
    def evals(self):
        """
        Eigenvalues associated with the covariance matrix eigenvectors
        """
        return self._evals
    @evals.setter
    def evals(self, newval):
        self._evals = newval

    @property
    def evecs(self):
        """
        Eigenvectors of the covariance matrix
        """
        return self._evecs
    @evecs.setter
    def evecs(self, newval):
        self._evecs = newval
        
    @property
    def kl_basis(self):
        """
        The KL basis that is used for PSF subtraction
        """
        return self._kl_basis
    @kl_basis.setter
    def kl_basis(self, newval):
        self._kl_basis = newval
        
    @property
    def full_kl_basis(self):
        """
        KL basis corresponding to the full reference cube
        """
        return self._full_kl_basis
    @full_kl_basis.setter
    def full_kl_basis(self, newval):
        self._full_kl_basis = newval

    # Forward modeling and Matched Filter
    @property
    def matched_filter(self):
        # Npix x (Nx x Ny) datacube of matched filters for each position
        return self._matched_filter
    @matched_filter.setter
    def matched_filter(self, newval):
        self._matched_filter = newval
        # if we have a matched filter, AND it's for every pixel, initialize the indices
        if self.matched_filter is not None:
            shape = self.matched_filter.shape
            if shape[0] == reduce(lambda x,y: x*y, shape[1:]): # works if mf is already flattened
                self.matched_filter_locations = np.arange(self.matched_filter.shape[0])
    @property
    def matched_filter_locations(self):
        """
        Array that tells you what pixels the matched filter slices correspond to
        If not given, assume all locations in image 
        """
        return self._matched_filter_locations
    @matched_filter_locations.setter
    def matched_filter_locations(self, newval=None):
        if newval is None and self.matched_filter is not None:
            newval = np.arange(len(self.matched_filter))
        elif newval is None and self.matched_filter is None:
            newval = np.arange(self.npix_region)
        self._matched_filter_locations = newval
        
        
    # Mask
    def set_mask(self, mask):
        """
        Sets a new mask
        Inputs:
            mask: a 2-D binary mask, 1=included, 0=excluded (region = im*mask)
        Sets the region, which triggers selecting the appropriate reference cube and target regions
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

    def get_reduced_covariance_matrix(self, ix):
        """
        return a copy of the covariance matrix where the covariance with one of the
        images of the reference cube has been removed
        Args:
          ix: index of the image to remove from the covariance matrix
        """
        covmat_shape = self.covar_matrix.shape
        mask = np.ones(covmat_shape)
        mask[ix,:] = 0
        mask[:,ix] = 0
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

    #####################################
    ### Wrappers for module functions ###
    #####################################
    def klip_subtract_with_basis(self, **kwargs):
        """
        This is a wrapper for RDI.klip_subtract_with_basis that defaults to the
        reference cube properties as arguments
        Useage:
        {0}
        """.format(klip_subtract_with_basis.__doc__)
        argdict={}
        argdict['img_flat'] =  kwargs.get('img_flat', getattr(self,'target_region'))
        argdict['kl_basis'] =  kwargs.get('kl_basis', getattr(self,'kl_basis'))
        try:
            argdict['n_bases'] =  kwargs.get('n_bases', getattr(self,'n_basis'))
        except AttributeError:
            argdict['n_bases'] = len(getattr(self,'kl_basis'))
        argdict['double_project']= kwargs.get('double_project', False)
        vals = klip_subtract_with_basis(argdict['img_flat'],
                                        argdict['kl_basis'],
                                        argdict['n_bases'],
                                        argdict['double_project'])
        return vals
    
    def generate_kl_basis(self, return_vals=False, **kwargs):
        """
        This is a wrapper for RDI.generate_kl_basis that defaults to the
        reference cube properties as arguments
        Useage:
        {0}
        """.format(generate_kl_basis.__doc__)
        argdict={}
        argdict['references'] = kwargs.get('references', getattr(self,'flat_cube_region'))
        argdict['kl_max'] = kwargs.get('kl_max', np.max(self.n_basis))#len(argdict['references']))
        argdict['return_evecs'] = kwargs.get('return_evecs', False)
        argdict['return_evals'] = kwargs.get('return_evals', False)
        vals = generate_kl_basis(references = argdict['references'],
                                 kl_max = argdict['kl_max'],
                                 return_evecs = argdict['return_evecs'],
                                 return_evals = argdict['return_evals'])
        if return_vals is True:
            return vals
        else:
            self.kl_basis = vals
                    
        
    def remove_ref_from_kl_basis(self, ref_index, **kwargs):
        """
        This is a wrapper for RDI.remove_ref_from_kl_basis that defaults to the
        reference cube properties as arguments
        Useage:
        {0}
        """.format(remove_ref_from_kl_basis.__doc__)
        argdict={}
        argdict['references'] = kwargs.get('references', getattr(self,'flat_cube_region'))
        argdict['kl_basis'] = kwargs.get('kl_basis', getattr(self,'full_kl_basis'))
        argdict['evecs'] = kwargs.get('evecs', getattr(self,'evecs'))
        argdict['evals'] = kwargs.get('evals', getattr(self,'evals'))

        new_basis = remove_ref_from_kl_basis(ref_index, **argdict)

        return new_basis

    #@classmethod
    def generate_matched_filter(self, return_mf=False, no_kl=True, **kwargs):
        """
        This is a wrapper for RDI.generate_matched_filter that defaults to the 
        reference cube properties as arguments
        Default behavior: returns nothing: sets self.matched_filter
        If return_mf is True, then returns matched filter
        For more help, see RDI.generate_matched_filter()
        """
        argdict = {}
        argdict['psf'] = kwargs.get('psf',self.instrument.psf)
        # KL basis
        if no_kl is True:
            argdict['kl_basis'] = None
        else:
            try:
                argdict['kl_basis'] = kwargs.get('kl_basis', self.kl_basis)
            except AttributeError as e:
                print(e)
                print("No KL basis - generating MF with unmodified PSFs")
                argdict['kl_basis'] = None
        argdict['n_bases'] = kwargs.get('n_bases', self.n_basis)
        argdict['imshape'] = kwargs.get('imshape', self.imshape)
        argdict['region_pix'] = kwargs.get('region_pix', self.flat_region_ind)
        argdict['mf_locations'] = kwargs.get('mf_locations', self.matched_filter_locations)
        # set matched_filter_locations
        self.matched_filter_locations = argdict['mf_locations']
        mf = MF.generate_matched_filter(**argdict) # this calls the module method
        if return_mf is False:
            self.matched_filter = mf
        else:
            return mf

    @classmethod
    def apply_matched_filter_to_image(self, image, **kwargs):
        """
        This is a wrapper for RDI.generate_matched_filter that defaults to the 
        reference cube properties as arguments. 
        For additional arguments, see MF.apply_matched_filter_to_image()
        Args:
            image: 2-D image
        Returns:
            mf_map: 2-D image where the matched filter has been applied 
                    at the designated positions
        For more help, see RDI.generate_matched_filter()
        """
        #argdict={}
        #argdict['matched_filter'] = kwargs.get('matched_filter', getattr(self,'matched_filter'))
        #argdict['locations'] = kwargs.get('locations', getattr(self,'matched_filter_locations'))
        kwargs['matched_filter'] = kwargs.get('matched_filter', getattr(self,'matched_filter'))
        #print(kwargs['matched_filter'])
        #print(type(kwargs['matched_filter']))
        kwargs['locations'] = kwargs.get('locations', getattr(self,'matched_filter_locations'))
        mf_map = MF.apply_matched_filter_to_image(image, **kwargs)
                                               #matched_filter,
                                               #matched_filter_locations)
        return mf_map

    @classmethod
    def apply_matched_filter_to_images(self, image, **kwargs):
        """
        This is a wrapper for RDI.generate_matched_filter that defaults to the 
        reference cube properties as arguments. For additional arguments, see MF.apply_matched_filter_to_images()
        Args:
            image: 2-D image or 3-D image cube
        Returns:
            mf_map: 2-D image where the matched filter has been applied 
                    at the designated positions
        For more help, see RDI.generate_matched_filter()
        """
        kwargs['matched_filter'] = kwargs.get('matched_filter', getattr(self,'matched_filter'))
        kwargs['locations'] = kwargs.get('locations', getattr(self,'matched_filter_locations'))
        return  MF.apply_matched_filter_to_images(image, **kwargs)




    
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


def make_image_from_region(region, indices, shape):
    """
    put the flattened region back into an image
    Input:
        region: [Nimg,[Nx,[Ny...]]] x Npix array (any shape as long as the last dim is the pixels)
        indices: Npix array of flattened pixel coordinates 
                 corresponding to the region
        shape: image shape
    Returns:
        img: an image (or array of) with dims `shape` and with nan's in 
            whatever indices are not explicitly set
    """
    oldshape = np.copy(region.shape)
    img = np.ravel(np.zeros(shape))*np.nan
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


def klip_subtract_with_basis(img_flat, kl_basis, n_bases=None, double_project=False):
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
    # the KL axis ends up in front. We want to put it just before the image axis
    kl_sub = np.rollaxis(kl_sub, 0, -1)
    return np.squeeze(kl_sub)
    #kl_sub = np.reshape(kl_sub, new_shape)
    #return np.squeeze(kl_sub)


    




################
### KL BASIS ###
################
def generate_kl_basis(references, kl_max, return_evecs=False, return_evals=False):
    """
    Generate the KL basis from a set of reference images. Always returns the full KL basis
    This is pretty straight-up borrowed from pyKLIP (Wang et al. 2015)
    Args:
        references: Nimg x Npix flattened array of reference images
        kl_max [None]: number of KL modes to return. Default: all
        return_evecs [False]: if True, return the eigenvectors of the covar matrix
        return_evals [False]: if true, return the eigenvalues of the eigenvectors
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
    

def old_generate_kl_basis(references=None, kl_max=None, return_evecs=False, return_evals=False):
    """
    Generate the KL basis from a set of reference images. Always returns the full KL basis
    This is pretty straight-up borrowed from pyKLIP (Wang et al. 2015)
    Args:
        references: Nimg x Npix flattened array of reference images
        kl_max [None]: number of KL modes to return. Default: all
        return_evecs [False]: if True, return the eigenvectors of the covar matrix
        return_evals [False]: if true, return the eigenvalues of the eigenvectors
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

    # check kl_max cases: cannot be greater than the number of references
    # if kl_max isn't provided
    if kl_max is None:
        kl_max = nrefs-1
    # if it's a number, make sure it's less than nrefs
    elif kl_max >= nrefs:
        kl_max = nrefs-1
    else: # whatever else, just use all the kl modes
        pass #kl_max = nrefs-1
    
    # generate the covariance matrix
    covar = np.cov(ref_psfs_mean_sub) * (npix-1) # undo numpy's normalization
    evals, evecs = la.eigh(covar, eigvals=(0,kl_max-1))
    
    # reverse the evals and evecs to have the biggest first
    evals = np.copy(evals[::-1])
    evecs = np.copy(evecs[:,::-1], order='F')
    
    kl_basis = np.dot(ref_psfs_mean_sub.T, evecs).T / np.sqrt(evals[:,None])
    check_nans = np.any(evals <= 0)
    if check_nans:
        neg_evals = (np.where(evals <= 0))[0]
        kl_basis[:,neg_evals] = 0

    # take care of nans
    if check_nans:
        kl_basis[:, neg_evals] = 0
    
    return_objs = [kl_basis]
    if return_evecs:
        return_objs.append(evecs)
    if return_evals:
        return_objs.append(evals)
    if len(return_objs) == 1:
        return_objs = return_objs[0]
    return return_objs


def remove_ref_from_kl_basis(ref_index, references, kl_basis=None, evecs=None, evals=None):
    """
    Subtract the contribution of a reference PSF from the KL basis
    Args:
        references: set of reference images Nref x Npix (i.e. flattened)
        ref_index: index of the ref image to remove from the basis
        kl_basis [None]: the full kl basis Nref x Npix
        evecs [None]: eigenvectors of the covariance matrix
        evals [None]: eigenvalues of the covariance matrix
        If kl_basis, evecs, and evals are None, regenerate them
    Returns:
        new_basis: KL basis with the contributions from one image removed
    """
    if np.any([i is None for i in [kl_basis, evecs, evals]]):
        kl_basis, evecs, evals = generate_kl_basis(references,
                                                   return_evecs=True,
                                                   return_evals=True)
    refs_mean_sub = references - np.mean(references, axis=-1)[:,None]
    n_modes = len(evals)
    weights = evecs[ref_index,:]/np.sqrt(evals)
    contributions = weights[:,None] * refs_mean_sub
    new_basis = kl_basis - contributions
    return new_basis

############################
# REFERENCE CUBE ITERATION #
############################
def iterate_references(references, n_bases, matched_filter=None, mf_locations=None):
    """
    Takes a reference cube and returns an array of the whole klipped thing, and the normalizations
    """
    for ind in list(range(len(references))):
        rc_ref_indices = list(range(len(references)))
        rc_targ_index = rc_ref_indices.pop(ind)
        tmp_targ = references[rc_targ_index].copy()
        references = references[rc_ref_indices]
        rc = RDI.ReferenceCube(references, target=tmp_targ, region_mask=planet_mask, instrument=NICMOS)
        ref_klipped_flat, kl_basis, evals, evecs = klip.klip_math(rc.target_region, rc.flat_cube_region, 
                                                                  numbasis=np.array(n_bases), 
                                                                  return_basis_and_eig=True)
        rc.kl_basis = kl_basis
        rc.n_basis = n_bases
        
        ref_klflat = RDI.klip_subtract_with_basis(rc.target_region, rc.kl_basis, 
                                                  rc.n_basis, double_project=True)
        ref_klflat_img = RDI.make_image_from_region(ref_klflat, rc.flat_region_ind, rc.imshape)
        
        # Set up and apply matched filter
        rc.matched_filter = master_mf
        rc.matched_filter_locations = rc.flat_region_ind #[np.ravel_multi_index(planet_pix, rc.imshape)]
        ref_mf_fluxes.append(rc.apply_matched_filter_to_image(ref_klflat_img))
        mf_norm = MF.fmmf_throughput_correction(rc.matched_filter[:,rc.flat_region_ind], 
                                                 rc.kl_basis, 
                                                 rc.n_basis)
        ref_mf_fluxes[-1] = utils.flatten_image_axes(ref_mf_fluxes[-1])[:,rc.flat_region_ind]/mf_norm

        


def apply_klip_and_mf(img_flat, img_shape, kl_basis, kl_pix, n_bases=None, double_project=False,
                      matched_filter=None, mf_locations=None, throughput_corr=False, flux_scale=1):
    """
    Wrapper that handles both doing KLIP on an image and getting the flux with a matched filter.
    Args:
      img_flat: (Nparam x) Npix flattened image - pixel axis must be last
      img_shape: 2-D Nrow, Ncol image shape
      kl_basis: (Nbases x ) Nklip x Npix array of KL basis vectors (possibly more than one basis)
      n_bases [None]: list of integers for Kmax, return one image per Kmax in n_bases.
          If None, use full basis
      double_project: apply KL twice (useful for some FMMF cases)
      matched_filter: Cube of flattened matched filters, one MF per pixel (flattened along the second axis)
      mf_locations: flattened pixel indices of the pixels to apply the matched filter
      throughput_corr: [False] to apply throughput correction, pass an array of normalization factors 
            that matches the shape of the matched filter
      flux_scale: any additional scaling you want to apply to the matched filter

    Returns:
      mf_map: image with the matched filter results
    """
    kl_sub = klip_subtract_with_basis(img_flat, kl_basis, n_bases, double_project)
    kl_sub_img = make_image_from_region(kl_sub, kl_pix, img_shape)
    mf = MF.apply_matched_filter_to_images(kl_sub_img, matched_filter, locations=mf_locations,
                                           throughput_corr=throughput_corr, scale=flux_scale)
    return mf
