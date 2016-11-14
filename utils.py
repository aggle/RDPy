"""
Assorted helper functions for RDI stuff with ALICE data
"""

import numpy as np
from astropy import units

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


