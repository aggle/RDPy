
"""
Assorted helper functions for RDI stuff with ALICE data
"""

import numpy as np
from astropy import units
from functools import reduce


# general rotation matrix
#def rot_mat(angle):
#    """
#    Generate a rotation matrix for a given angle in degrees. 
#    It is designed to operate on images in python and rotate them
#    such that north is up and east is to the left (i.e. CCW).
#    Args:
#        angle: rotation angle in degrees
#    Returns: 
#        2x2 rotation matrix to rotate a 2-D vector by the given angle
#    """
rot_mat = lambda angle: np.array([[np.cos(angle*np.pi/180), np.sin(angle*np.pi/180)],
                                  [-np.sin(angle*np.pi/180), np.cos(angle*np.pi/180)]])


###############
# COORDINATES #
###############
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
    # handle odd and even: 1 if odd, 0 if even
    oddflag = np.array((dcol%2, drow%2))
    colrad = np.int(np.floor(dcol))/2 
    rowrad = np.int(np.floor(drow))/2 

    rads = np.array([rowrad, colrad], dtype=np.int)
    center = np.array([center[0],center[1]],dtype=np.int) #+ oddflag
    img = np.zeros(imshape)
    stamp = np.ones((drow,dcol))
    full_stamp_coord = np.indices(stamp.shape) + center[:,None,None]  - rads[:,None,None]
    # check for out-of-bounds values
    # boundaries
    row_lb,col_lb = (0, 0)
    row_hb,col_hb = imshape

    rowcheck_lo, colcheck_lo = (center - rads)
    rowcheck_hi, colcheck_hi = ((imshape-center) - rads) - oddflag[::-1]

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


def rotate_centered_coordinates(coord, angle, center=(40,40)):
    """
    Calculate a clockwise rotation of (row,col) coordinates about an angle centered on `center`
    returns the new (row, col)
    """
    coord = np.array(coord[::-1])
    center = np.array(center[::-1])
    angle = angle * np.pi/180 # convert to radians
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    newx = np.dot(rot_mat, coord-center) + center
    return newx[::-1]

def image_2_seppa(coord, ORIENTAT=0, center=(40,40), pix_scale = 75):
    """
    convert from image coordinates to sep and pa
    coord: (row, col) coord in image
    orientat: angle between y in image and north in nicmos, measured to the east
    center: image center
    pix_scale: 75 mas/pix for nicmos
    """
    coord = np.array(coord)
    center = np.array(center)
    centered_coord = coord-center
    sep = np.linalg.norm(centered_coord, axis=-1)*pix_scale
    # rotation and ORIENTAT are defined CCW so need a -1 here
    pa = -1*np.arctan2(centered_coord.T[1],centered_coord.T[0])*180/np.pi
    pa += ORIENTAT
    return (sep, pa)

def seppa_2_image(sep, pa, ORIENTAT=0, center=(40,40), pix_scale = 75):
    """
    convert from separation and position angle to image coordiates
    pix_scale: 75 mas/pix for nicmos
    """
    # convert all angles from degrees to radians
    center = np.array(center)
    pa = pa*np.pi/180
    ORIENTAT = ORIENTAT*np.pi/180
    # rotation and ORIENTAT are defined CCW so need a -1 here
    tot_ang = -1*(pa-ORIENTAT)
    row = sep*np.cos(tot_ang)/pix_scale + center[0]
    col = sep*np.sin(tot_ang)/pix_scale + center[1]
    return (row, col)


#########
# MASKS #
#########
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

def make_circular_mask(center, rad, shape, invert=False):
    """
    Make a circular mask around a point, cutting it off for points outside the image
    Inputs: 
        center: (y,x) center of the mask
        rad: radius of the circular mask
        shape: shape of the image
        invert: [False] if True, 1 *outside* radius and 0 inside radius
    Returns:
        mask: 2D mask where 1 is in the mask and 0 is out of the mask (multiple it by the image)
    """
    grid = np.mgrid[:np.float(shape[0]),:np.float(shape[1])]
    centered_grid = np.rollaxis(np.rollaxis(grid, 0, grid.ndim) - center, -1, 0)
    rad2D = np.linalg.norm(centered_grid, axis=0)
    mask = np.zeros(shape)
    mask[np.where(rad2D <= rad)] = 1
    return mask


#######################
### Array Reshaping ###
#######################
def flatten_image_axes(array):
    """
    returns the array with the final two axes - assumed to be the image pixels - flattened
    """
    shape = array.shape
    imshape = shape[-2:]
    newshape = [i for i in list(shape[:-2])]

    newshape += [reduce(lambda x,y: x*y, shape[-2:])]
    return array.reshape(newshape)

def flatten_leading_axes(array, axis=-1):
    """
    For an array of flattened images of with N axes where the first N-1 axes
    index parameters (or whatever else), return an array with the first N-1 axes flattened
    so the final result is 2-D, with the last axis being the pixel axis
    Args:
        array: an array of at least 2 dimensions
        axis [-1]: flattens shape up to this axis (e.g. -1 to flatten up to 
          the last axis, -2 to preserve last two axes, etc.)
    """
    # test axis value is valid
    if np.abs(axis) >= array.ndim:
        print("Preserving all axes shapes")
        return array
    oldshape = array.shape
    newshape = [reduce(lambda x,y: x*y, oldshape[:axis])] + list(oldshape[axis:])
    return np.reshape(array, newshape)

def make_image_from_region(region, indices=None, shape=None):
    """
    put the flattened region back into an image. if no indices or shape are specified, assumes that
    the region of N pixels is a square with Nx = Ny = sqrt(N)
    Input:
        region: [Nimg,[Nx,[Ny...]]] x Npix array (any shape as long as the last dim is the pixels)
        indices: [None] Npix array of flattened pixel coordinates 
                 corresponding to the region
        shape: [None] image shape
    Returns:
        img: an image (or array of) with dims `shape` and with nan's in 
            whatever indices are not explicitly set
    """
    oldshape = np.copy(region.shape)
    if shape is None:
        Npix = oldshape[-1]
        Nside = np.int(np.sqrt(Npix))
        indices = np.array(range(Npix))
        shape = (Nside, Nside)

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


#################
# PSF INJECTION #
#################
def _inject_psf(img, psf, center, scale_flux=None, subtract_mean=False, return_flat=False):
    """
    ### use inject_psf() as a wrapper, do not call this function directly ###
    Inject a PSF into an image at a location given by center. Optional: scale PSF.
    Input:
        img: 2-D img or 3-D cube. Last two dimensions define an img (i.e. [(Nimg,)Nx,Ny])
        psf: 2-D img or 3-D cube, smaller than or equal to img in size. If cube, 
             must have same 1st dimension as img 
        center: center of the injection in the image
        scale_flux: multiply the PSF by this number. If this is an array,
             img and psf will be tiled to match its length
             if scale_flux is None, don't scale psf
        subtract_mean: (False) mean-subtract before returning
        return_flat: (False) flatten the array along the pixels axis before returning
    Returns:
       injected_img: 2-D image or 3D cube with the injected PSF(s)
       injection_psf: (if return_psf=True) 2-D normalized PSF full image
    """
    if scale_flux is None:
        scale_flux = np.array([1])
    elif np.ndim(scale_flux) == 0:
        scale_flux = np.array([scale_flux])
    scale_flux = np.array(scale_flux)

    # get the right dimensions
    img_tiled = np.tile(img, (np.size(scale_flux),1,1))
    psf_tiled = np.tile(psf, (np.size(scale_flux),1,1))

    # get the injection pixels
    injection_pix, psf_pix = get_stamp_coordinates(center, psf.shape[0], psf.shape[1], img.shape)
    
    # normalized full-image PSF in case you want it later
    #injection_psf = np.zeros(img.shape)
    #cut_psf = psf[psf_pix[0],psf_pix[1]]
    #injection_psf[injection_pix[0], injection_pix[1]] += cut_psf/np.nansum(psf)

    # add the scaled psfs
    injection_img = np.zeros(img_tiled.shape)
    #injection_img[:,injection_pix[0], injection_pix[1]] += (psf_tiled.T*scale_flux).T
    injection_img[:,injection_pix[0], injection_pix[1]] += psf_tiled[:,psf_pix[0], psf_pix[1]]*scale_flux[:,None,None]
    full_injection = injection_img + img_tiled
    if subtract_mean is True:
        full_injection = full_injection - np.nanmean(np.nanmean(full_injection, axis=-1),axis=-1)[:,None,None]
    if return_flat is True:
        shape = full_injection.shape
        if full_injection.ndim == 2:
            full_injection = np.ravel(full_injection)
        else:
            full_injection = np.reshape(full_injection, (shape[0],reduce(lambda x,y: x*y, shape[1:])))
    #if return_psf is True:
    #    return full_injection, injection_psf
    return np.squeeze(full_injection)

def inject_psf(img, psf, center, scale_flux=None, subtract_mean=False, return_flat=False):
    """
    Inject a PSF into an image at a location given by center. Optional: scale PSF
    The PSF is injected by *adding* it to the provided image, not by replacing the pixels
    Input:
        img: 2-D img or 3-D cube. Last two dimensions define an img (i.e. [(Nimg,)Nx,Ny])
        psf: 2-D img or 3-D cube, smaller than or equal to img in size. If cube, 
             must have same 1st dimension as img 
        center: center of the injection in the image (can be more than one location)
        scale_flux: multiply the PSF by this number. If this is an array,
             img and psf will be tiled to match its length
             if scale_flux is None, don't scale psf
        subtract_mean: (False) mean-subtract before returning
        return_flat: (False) flatten the array along the pixels axis before returning
    Returns:
       injected_img: 2-D image or 3D cube with the injected PSF(s)
       injection_psf: (if return_psf=True) 2-D normalized PSF full image
    """

    injected_psf=None
    if np.ndim(center) == 1:
        injected_psf  = _inject_psf(img, psf, center, scale_flux, subtract_mean, return_flat)
    elif np.ndim(center) > 1:
        injected_psf = np.sum(np.array([_inject_psf(img, psf, c, scale_flux,
                                                    subtract_mean, return_flat)
                                        for c in center]), axis=0)
    return injected_psf
    

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

def mean_subtracted_fake_injections(flat_img, flat_psf, scaling=1):
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


####################
# NaN manipulation #
####################
def denan(array):
    """convert nans to 0's"""
    new_array = array[:]
    new_array[np.where(np.isnan(new_array))] = 0
    return new_array

def renan(array, indices=None):
    """replace 0's with NaNs"""
    new_array = array[:]
    if indices is None:
        new_array[np.where(array==0)] = np.nan
    else:
        new_array[indices] = np.nan
    return new_array
