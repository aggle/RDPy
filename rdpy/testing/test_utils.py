##import sys
#sys.path.append('../')
from .. import utils

import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pathlib import Path
import pandas as pd

# use the same random seed every time
random_seed = 1

# change this line to change the HPF widths you want to test
hpf_index = [0, 5, 10, 15, 20, 25, 30, 35, 40]
hpf_index = hpf_index[::3]

# NICMOS properties
NICMOS_imshape = utils.np.array([80, 80])

@pytest.fixture()
def set_random_seed():
    """
    Used to make the random seed predictable.
    Call whenever you have a test that needs random numbers
    """
    utils.np.random.seed(random_seed)

@pytest.fixture
def real_data():
    """
    Loads the beta Pic, HR 8799, and reference data for testing
    Dataframe columns: 
    """
    #initial_data_path = Path(__file__).resolve().parent / '..' / '..' / 'BetaPic' / 'staticdata'
    #data_path = initial_data_path / 'init_data_hpf.pkl'
    #print('oops', data_path)
    data_df = pd.read_pickle('/Users/jaguilar/Projects/RDI/testing/../../BetaPic/staticdata/init_data_hpf.pkl')
    #print(type(data_df))
    return(data_df)

def test_get_correct_stamp_indices():
    # Setup
    stamp_shape = (3, 3)
    cube = utils.np.arange(400).reshape(4, 10, 10)
    imshape = cube.shape[-2:]
    test_index = utils.np.array([0, 0])
    loc_ravel = utils.np.ravel_multi_index(test_index, imshape)
    # Exercise
    stamps = utils.get_stamp_from_cube(cube, stamp_shape, loc_ravel)
    stamp_sum = utils.np.nansum(stamps, axis=(-1, -2)).astype(utils.np.int)
    # Validate
    answer = 22 + 400*utils.np.arange(4)
    utils.np.testing.assert_array_equal(answer, stamp_sum)
    # cleanup - none

def test_get_cube_of_stamps():
    # Setup
    stamp_shape = (11, 11)
    # Exercise
    # pixels of interest: a circle in the middle
    rad = 10
    region_mask = utils.make_circular_mask((40, 40), rad, NICMOS_imshape)
    pixel_indices = utils.np.ravel_multi_index(utils.np.where(region_mask), NICMOS_imshape)
    stamp_cube = utils.get_cube_of_stamps_from_image(region_mask, stamp_shape, pixel_indices)
    # Validate
    assert(stamp_cube.shape == (len(pixel_indices), stamp_shape[0], stamp_shape[1]))

    # we should be able to calculate the sum and check that
    # exclude the edge stamps because those are hard
    edge_stamps = [i for i in range(len(stamp_cube)) if utils.np.any(stamp_cube[i]==0)]
    cent_stamps = [i for i in range(len(stamp_cube)) if utils.np.all(stamp_cube[i]==1)]
    # check that the edge stamps and cent stamps cover all the pixels
    assert(len(edge_stamps) + len(cent_stamps) == len(pixel_indices))
    # check that the sum of the center stamps is what it should be if the stamps
    # are truly in the center
    assert(utils.np.sum(stamp_cube[cent_stamps]) == len(cent_stamps)*stamp_shape[0]*stamp_shape[1])
    #assert(np.nansum(stamp_cube) == 100)
    # Cleanup - none

def test_orient_alice_image():
    # setup
    image = utils.np.ones((100,100))
    angle = 90
    # Exercise
    rot_img = utils.orient_alice_image(image, orientat=angle)
    # Validate
    assert(utils.np.shape(rot_img) == utils.np.shape(image))
    # Cleanup - none

def test_inject_psf():
    """
    test that when you inject a psf into an image it has the expected properties
    """
    # setup
    psf = utils.np.array([[1.]])
    img = utils.np.zeros((11,11))
    loc = utils.np.array([5,5])
    flux = 10.
    # exercise
    injected_img = utils.inject_psf(img, psf, loc, scale_flux=flux,
                                    subtract_mean=False,
                                    return_flat=False,
                                    hpf=None)
    # validate
    assert(injected_img[loc[0], loc[1]] == flux)
    assert(utils.np.sum(injected_img) == flux)
    # cleanup - none

