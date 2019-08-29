from .. import RDI

import pytest
from numpy import testing as nptest
from pathlib import Path


@pytest.fixture()
def set_random_seed():
    """
    Used to make the random seed predictable.
    Call whenever you have a test that needs random numbers
    """
    MF.np.random.seed(random_seed)


@pytest.fixture
def generate_test_data():
    """
    Fixture to generate test data
    """
    targ = RDI.np.array([1, 2, 3, 4])
    references = RDI.np.array([[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [1, 1, 1, 1],
                               [4, 3, 2, 1]])
    return targ, references


def test_mean_squared_error(generate_test_data):
    """
    Test MSE computation
    """
    # Setup
    targ, references = generate_test_data
    # Exercise
    mse = RDI.calc_refcube_mse(targ, references)
    truth = RDI.np.array([0., 16., 3.5, 5.])
    # Validate
    nptest.assert_array_almost_equal(mse, truth, 13)
    # Cleanup - none


def test_pearson_correlation(generate_test_data):
    # Setup
    targ, references = generate_test_data
    # Exercise
    pcc = RDI.calc_refcube_pcc(targ, references)
    truth = RDI.np.array([4/3, 4/3, RDI.np.nan, -4/3])
    # Validate
    nptest.assert_array_almost_equal(pcc, truth, 13)
    # Cleanup - none
