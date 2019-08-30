from .. import RDI

import pytest
from numpy import testing as nptest
from pathlib import Path
import itertools

# change this line to change the HPF widths you want to test
hpf_index = [0, 5, 10, 15, 20, 25, 30, 35, 40]
hpf_index = hpf_index[::4]


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

@pytest.fixture
def real_data():
    """
    Loads the beta Pic, HR 8799, and reference data for testing
    Dataframe columns:
    """
    #initial_data_path = Path(__file__).resolve().parent / '..' / '..' / 'BetaPic' / 'staticdata'
    #data_path = initial_data_path / 'init_data_hpf.pkl'
    #print('oops', data_path)
    data_df = RDI.pd.read_pickle('/Users/jaguilar/Projects/BetaPic/staticdata/init_data_hpf.pkl')
    return(data_df)

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
    truth = RDI.np.array([1., 1., RDI.np.nan, -1.]) # RDI.np.array([4/3, 4/3, RDI.np.nan, -4/3])
    # Validate
    nptest.assert_array_almost_equal(pcc, truth, 13)
    
    # Cleanup - none

@pytest.mark.parametrize('hpf, test_func',
                         list(itertools.product(hpf_index,
                                                [RDI.ssim_luminance,
                                                 RDI.ssim_contrast,
                                                 RDI.ssim_structural])
                         )
)
def test_ssim_helper_functions(real_data, hpf, test_func):
    """
    All values should be between -1 and 1
    """
    # cut the number of references used by 4 to save time for
    result = test_func(real_data.loc[hpf, 'betaPic'],
                       real_data.loc[hpf, 'refcube'][::4])
    assert(RDI.np.all((-1<= result) & (result <= 1)))

@pytest.mark.skip("redundant?")
def test_ssim_luminance(real_data, hpf):
    """
    Test the Luminance part of the calculation.
    All values should be between 0 and 1
    """
    # cut the number of references used by 4 to save time for
    lum = RDI.ssim_luminance(real_data.loc[hpf, 'hr8799'][0],
                             real_data.loc[hpf, 'refcube'][::4])
    assert(RDI.np.all((0 <= lum) & (lum <= 1)))



def test_structural_similarity(generate_test_data):
    """
    Value should be only between 0 and 1
    """
    pass
