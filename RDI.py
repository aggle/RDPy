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
import RDIklip as RK

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
        """.format(RK.klip_subtract_with_basis.__doc__)
        argdict={}
        argdict['img_flat'] =  kwargs.get('img_flat', getattr(self,'target_region'))
        argdict['kl_basis'] =  kwargs.get('kl_basis', getattr(self,'kl_basis'))
        try:
            argdict['n_bases'] =  kwargs.get('n_bases', getattr(self,'n_basis'))
        except AttributeError:
            argdict['n_bases'] = len(getattr(self,'kl_basis'))
        vals = RK.klip_subtract_with_basis(argdict['img_flat'],
                                           argdict['kl_basis'],
                                           argdict['n_bases'])
        return vals
    
    def generate_kl_basis(self, return_vals=False, **kwargs):
        """
        This is a wrapper for RDI.generate_kl_basis that defaults to the
        reference cube properties as arguments
        Useage:
        {0}
        """.format(RK.generate_kl_basis.__doc__)
        argdict = {}
        argdict['references'] = kwargs.get('references', getattr(self, 'flat_cube_region'))
        argdict['kl_max'] = kwargs.get('kl_max', np.max(self.n_basis))  #len(argdict['references']))
        argdict['return_evecs'] = kwargs.get('return_evecs', False)
        argdict['return_evals'] = kwargs.get('return_evals', False)
        vals = RK.generate_kl_basis(references=argdict['references'],
                                    kl_max=argdict['kl_max'],
                                    return_evecs=argdict['return_evecs'],
                                    return_evals=argdict['return_evals'])
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
        """.format(RK.remove_ref_from_kl_basis.__doc__)
        argdict = {}
        argdict['references'] = kwargs.get('references', getattr(self, 'flat_cube_region'))
        argdict['kl_basis'] = kwargs.get('kl_basis', getattr(self, 'full_kl_basis'))
        argdict['evecs'] = kwargs.get('evecs', getattr(self, 'evecs'))
        argdict['evals'] = kwargs.get('evals', getattr(self, 'evals'))

        new_basis = remove_ref_from_kl_basis(ref_index, **argdict)

        return new_basis

    #@classmethod
    def generate_matched_filter(self, return_mf=False, no_kl=True, **kwargs):
        """
        This is a wrapper for RDI.generate_matched_filter that defaults to the
        reference cube properties as arguments
        Default behavior: returns nothing: sets self.matched_filter
        If return_mf is True, then returns matched filter
        For more help, see MatchedFilter.generate_matched_filter()
        """
        argdict = {}
        argdict['psf'] = kwargs.get('psf', self.instrument.psf)
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

    # old
    # targ_norm = targ/np.nansum(targ)
    # ref_norm = references/np.nansum(references, axis=-1)[:, None]
    targ_norm = targ/np.linalg.norm(targ)
    ref_norm = references/np.linalg.norm(references, axis=-1, keepdims=True)

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


###################################
#   FUNCTIONS DEFINED ELSEWHERE   #
#              LEGACY             #
###################################

# not actually used
make_image_from_region = utils.make_image_from_region
# fixed
klip_subtract_with_basis = RK.klip_subtract_with_basis
# fixed
generate_kl_basis = RK.generate_kl_basis

