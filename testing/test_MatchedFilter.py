##import sys
#sys.path.append('../')
from .. import MatchedFilter as MF

import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pathlib import Path


@pytest.fixture
def load_initial_data():
    """
    Loads the beta Pic, HR 8799, and reference data for testing
    Dataframe columns: 
    """
    #initial_data_path = Path(__file__).resolve().parent / '..' / '..' / 'BetaPic' / 'staticdata'
    #data_path = initial_data_path / 'init_data_hpf.pkl'
    #print('oops', data_path)
    data_df = MF.pd.read_pickle('/Users/jaguilar/Projects/RDI/testing/../../BetaPic/staticdata/init_data_hpf.pkl')
    #print(type(data_df))
    return(data_df)


def test_load_initial_data():
    # Setup
    df = load_initial_data()
    # Exercise - none
    # Validate
    assert(isinstance(df, MF.pd.core.frame.DataFrame))
    # cleanup - none


def test_calc_matched_filter_throughput():
    """
    No idea how to test this
    """
    pass


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_create_matched_filter(hpf):
    """
    Test that you can create a matched filter
    Should have mean 0
    """
    # Setup
    print(hpf)
    psf = load_initial_data()['psf'][hpf]
    # Exercise
    mf = MF.create_matched_filter(psf)
    # Validate
    mean_flux = mf.mean()
    assert_almost_equal(mean_flux, 0, 5)
    # Cleanup - none needed


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_null_psf(hpf):
    """
    Just apply a matched filter to an image of 0's
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_psf = MF.np.zeros_like(mf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result, 0, 10) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_flat_psf(hpf):
    """
    Just apply a matched filter to an image of 1's
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_psf = MF.np.ones_like(mf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result, 0, 6) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_psf_template(hpf):
    """
    Just apply a matched filter the PSF model which has a total flux of 1
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 1.
    flux = MF.np.nansum(data_df['psf'][0] * flux_scale)
    flux_psf = MF.RK.high_pass_filter(data_df['psf'][0] * flux_scale, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result/flux, 1, 2) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_scaled_psf(hpf):
    """
    This time, scale the PSF model to some arbitrary flux
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 479.4
    flux = MF.np.nansum(data_df['psf'][0] * flux_scale)
    flux_psf = MF.RK.high_pass_filter(data_df['psf'][0] * flux_scale, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 10e7
    assert_almost_equal(mf_result/flux, 1, 7) 

    # Cleanup
    del data_df


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_random_stamp(hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)



    stamp = MF.np.random.normal(data_df['kl'][0].mean(),
                                data_df['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get 0 flux....
    assert_almost_equal(mf_result, flux_scale, 4)

    # Cleanup
    del data_df



#### FFT Testing ####
@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_to_random_stamp_plus_psf(hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)


    stamp = MF.np.random.normal(data_df['kl'][0].mean(),
                                data_df['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    flux_scale = 5 * data_df['kl'][0].std()
    flux = MF.np.nansum(data_df['psf'][0] * flux_scale)
    stamp = data_df['psf'][0] * flux_scale + stamp

    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get get within 10% of the flux for an SNR of >= 5
    print('mf: {0:0.2f}, true: {1:0.2f}'.format(mf_result, flux_scale))
    diff = MF.np.abs(1 - mf_result/flux)
    assert(diff <= 0.25)
    #assert_almost_equal(mf_result/flux_scale, 1, 2)

    # Cleanup
    del data_df


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_fft(hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 479.4
    flux = MF.np.nansum(data_df['psf'][0] * flux_scale)
    flux_psf = MF.RK.high_pass_filter(data_df['psf'][0] * flux_scale, hpf)

    # Exercise
    # in this case, you just want the central pixel
    mf_result = MF.apply_matched_filter_fft(mf, flux_psf).max()

    # Validate
    # check that you get the flux back to better than 3%
    diff = MF.np.abs(1-mf_result/flux)
    assert(diff < 0.03)

    # Cleanup
    del data_df

@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_fft_to_random_stamp(hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)


    stamp = MF.np.random.normal(data_df['kl'][0].mean(),
                                data_df['kl'][0].std()*100,
                                mf.size).reshape(mf.shape)
    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter_fft(mf, flux_psf)
    # normalize by the sigma
    mf_snr = mf_result/mf_result.std()

    # Validate
    # check that the SNR is low
    assert(MF.np.all(mf_snr < 4))

    # Cleanup
    del data_df


@pytest.mark.parametrize('hpf', load_initial_data().index)
def test_apply_matched_filter_fft_to_random_stamp_with_psf(hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    data_df = load_initial_data()
    mf_template_psf = data_df['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)


    stamp = MF.np.random.normal(data_df['kl'][0].mean(),
                                data_df['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    flux_scale = 10 * data_df['kl'][0].std()
    stamp = mf_template_psf * flux_scale + stamp
    flux = Mf.np.nansum(data_df['psf'][0] * flux_scale)
    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf).max()

    # Validate
    # check that you get the flux back to better than 3%
    diff = MF.np.abs(1-mf_result/flux_scale)
    assert(diff < 0.03)
    #assert_almost_equal(mf_result/flux_scale, 1, 2)

    # Cleanup
    del data_df
