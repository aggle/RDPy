"""
Assorted helper functions for RDI stuff with ALICE data
"""

import numpy as np
from astropy import units


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


def rotate_centered_coordinates(coord, angle, center=(40,40)):
    """
    Calculate a clockwise rotation of (row,col) coordinates about an angle centered on `center`
    returns the new (row, col)
    """
    coord = np.array(coord[::-1])
    center = np.array(center[::-1])
    angle = angle * np.pi/180 # convert to radians
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle), ]])
    newx = np.dot(rot_mat, coord-center) + center
    return newx[::-1]

def image_2_seppa(coord, ORIENTAT, center=(40,40), pix_scale = 75):
    """
    convert from image coordinates to sep and pa
    coord: (row, col) coord in image
    orientat: clockwise angle betwee y in image and north in nicmos
    center: image center
    pix_scale: 75 mas/pix for nicmos
    """
    coord = np.array(coord)
    center = np.array(center)
    centered_coord = coord-center
    sep = np.linalg.norm(centered_coord)*pix_scale
    pa = np.arctan(centered_coord[1]/centered_coord[0])*180/np.pi
    pa -= ORIENTAT
    return (sep, pa)

def seppa_2_image(sep, pa, ORIENTAT, center=(40,40), pix_scale = 75):
    """
    convert from separation and position angle to image coordiates
    pix_scale: 75 mas/pix for nicmos
    """
    # convert all angles from degrees to radians
    center = np.array(center)
    pa = pa*np.pi/180
    ORIENTAT = ORIENTAT*np.pi/180
    tot_ang = pa+ORIENTAT
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
        axis [-1]: preserves shape up to this axis (e.g. -1 for last axis, -2 for last two axes)
    """
    # test axis value is valid
    if np.abs(axis) >= array.ndim:
        print("Preserving all axes shapes")
        return array
    oldshape = array.shape
    newshape = [reduce(lambda x,y: x*y, oldshape[:axis])] + list(oldshape[axis:])
    return np.reshape(array, newshape)
