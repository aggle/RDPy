##import sys
#sys.path.append('../')
from .. import MatchedFilter as MF

import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pathlib import Path

# use the same random seed every time
random_seed = 1

# change this line to change the HPF widths you want to test
hpf_index = [0, 5, 10, 15, 20, 25, 30, 35, 40]
hpf_index = hpf_index[::3]

# NICMOS properties
NICMOS_shape = MF.np.array([80, 80])

# helper functions
def prepare_injection(psf, flux_scale=1, hpf=0):
    """
    prepare a stamp to be injected with the correct flux and high-pass filtering
    Args:
      psf: psf stam, no processing
      flux_scale [1]: the total flux you want in the stamp
      hpf [0]: high-pass filter width
    Returns:
      processed_psf: the scaled and filtered PSF
      flux: the total flux in the original psf
    """
    scaled_stamp = psf/MF.np.nansum(psf) * flux_scale
    tot_flux = MF.np.nansum(scaled_stamp)
    processed_psf = MF.utils.high_pass_filter_stamp(scaled_stamp, hpf, NICMOS_shape)
    return processed_psf, tot_flux


@pytest.fixture()
def set_random_seed():
    """
    Used to make the random seed predictable.
    Call whenever you have a test that needs random numbers
    """
    MF.np.random.seed(random_seed)

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

    # Exercise 1 - test the data type
    # Validate
    assert(isinstance(df, MF.pd.core.frame.DataFrame))

    # Exercise 2 - test that the psfs are filtered correctly
    hpf = df.index[-1]
    psf_loaded = df.loc[hpf, 'psf']
    psf_template = MF.utils.high_pass_filter_stamp(df.loc[0, 'psf'],
                                                   hpf,
                                                   NICMOS_shape)
    # Validate
    assert_array_almost_equal(psf_loaded, psf_template, 5, verbose=True)

    # cleanup
    del df


def test_calc_matched_filter_throughput():
    """
    No idea how to test this
    """
    pass


@pytest.mark.parametrize('hpf', hpf_index)
def test_create_matched_filter(load_initial_data, hpf):
    """
    Test that you can create a matched filter
    Should have mean 0
    """
    # Setup
    psf = load_initial_data['psf'][hpf]
    # Exercise
    mf = MF.create_matched_filter(psf)
    # Validate
    mean_flux = mf.mean()
    assert_almost_equal(mean_flux, 0, 5)
    # Cleanup - none needed


@pytest.mark.parametrize('hpf', hpf_index)
def test_apply_matched_filter_to_null_psf(load_initial_data, hpf):
    """
    Just apply a matched filter to an image of 0's
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_psf = MF.np.zeros_like(mf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result, 0, 10) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
def test_apply_matched_filter_to_flat_psf(load_initial_data, hpf):
    """
    Just apply a matched filter to an image of 1's
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_psf = MF.np.ones_like(mf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result, 0, 5) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
def test_apply_matched_filter_to_psf_template(load_initial_data, hpf):
    """
    Just apply a matched filter the PSF model which has a total flux of 1
    """
    # Setup
    #data_df = load_initial_data()
    mf_template_psf = load_initial_data.loc[hpf, 'psf']
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 1.
    flux_psf, flux = prepare_injection(load_initial_data['psf'][0],
                                       flux_scale,
                                       hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get the flux back to 1 part in 100
    assert_almost_equal(mf_result, flux, 2) 
    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
def test_apply_matched_filter_to_scaled_psf(load_initial_data, hpf):
    """
    This time, scale the PSF model to some arbitrary flux
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 479.4
    img, expected_flux = prepare_injection(load_initial_data.loc[0, 'psf'],
                                           flux_scale,
                                           hpf)
    # Exercise
    mf_result = MF.apply_matched_filter(mf, img)

    # Validate
    # check that you get the flux back to 1 part in 1e3
    assert_almost_equal(mf_result/expected_flux, 1, 7)

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
#@pytest.mark.skip(reason="Using FFT matched filter, not dot product")
def test_apply_matched_filter_to_random_stamp(hpf, load_initial_data):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)


    flux_center = 0  # load_initial_data['kl'][0].mean()
    noise_scale = 1  # load_initial_data['kl'][0].std()

    MF.np.random.seed(random_seed)
    img = MF.np.random.normal(flux_center,  # load_initial_data['kl'][0].mean(),
                              noise_scale,  # load_initial_data['kl'][0].std(),
                              mf.size).reshape(mf.shape)
    img_hpf = MF.utils.high_pass_filter_stamp(img, hpf, 
                                              NICMOS_shape)  # load_initial_data.loc[0, 'betaPic'].shape)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, img_hpf)

    # Validate
    # check that the matched filter is better than half the noise level
    assert(mf_result < 0.5 * noise_scale)

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
@pytest.mark.skip(reason="Using FFT matched filter, not dot product")
def test_apply_matched_filter_to_random_stamp_plus_psf(load_initial_data, hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    MF.np.random.seed(random_seed)
    stamp = MF.np.random.normal(0, #load_initial_data['kl'][0].mean(),
                                load_initial_data['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    flux_scale = 5 * load_initial_data['kl'][0].std()
    flux = MF.np.nansum(load_initial_data['psf'][0] * flux_scale)
    stamp = load_initial_data['psf'][0] * flux_scale + stamp

    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf)

    # Validate
    # check that you get get within 10% of the flux for an SNR of >= 5
    print('mf: {0:0.2f}, true: {1:0.2f}'.format(mf_result, flux_scale))
    diff = MF.np.abs(1 - mf_result/flux)
    assert(diff <= 0.25)
    #assert_almost_equal(mf_result/flux_scale, 1, 2)

    # Cleanup - none


#####################
#### FFT Testing ####
#####################

@pytest.mark.parametrize('hpf', hpf_index)
#@pytest.mark.skip(reason="Currently fails")
def test_apply_matched_filter_fft_to_psf_loc(load_initial_data,  hpf):
    """
    Apply the FFT MF to a PSF model - check that the location is the central pixel
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 1
    flux = MF.np.nansum(load_initial_data['psf'][0] * flux_scale)
    img = MF.utils.high_pass_filter_stamp(load_initial_data.loc[0, 'psf'] * flux_scale,
                                          hpf,
                                          load_initial_data.loc[0, 'betaPic'].shape)
    #img = MF.RK.high_pass_filter(load_initial_data['psf'][0] * flux_scale, hpf)

    # Exercise
    # in this case, you just want the central pixel
    mf_result = MF.np.argmax(MF.apply_matched_filter_fft(mf, img))
    mf_result = MF.np.unravel_index(mf_result, img.shape)
    # Validate
    # check that you get the flux back to better than 3%
    center = MF.np.floor(MF.np.array(img.shape)/2).astype(MF.np.int)
    assert(MF.np.all(mf_result == center))

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
def test_apply_matched_filter_fft_to_psf_flux(load_initial_data, hpf):
    """
    Apply the FFT MF to a PSF model - check that the flux is correct
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    flux_scale = 479.4
    flux = MF.np.nansum(load_initial_data['psf'][0] * flux_scale)
    img = MF.utils.high_pass_filter_stamp(load_initial_data.loc[0, 'psf'] * flux_scale,
                                          hpf,
                                          load_initial_data.loc[0, 'betaPic'].shape)
    #img = MF.RK.high_pass_filter(load_initial_data['psf'][0] * flux_scale, hpf)

    # Exercise
    # in this case, you just want the central pixel
    center = MF.np.floor(MF.np.array(img.shape)/2).astype(MF.np.int)
    mf_result = MF.apply_matched_filter_fft(mf, img)[center[0], center[1]]

    # Validate
    # check that you get the flux back to better than 3%
    diff = MF.np.abs(mf_result/flux - 1)
    assert(diff < 3e-2)

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
@pytest.mark.skip(reason="Currently fails")
def test_apply_matched_filter_fft_to_random_stamp(load_initial_data, hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    MF.np.random.seed(random_seed)
    stamp = MF.np.random.normal(0, #load_initial_data['kl'][0].mean(),
                                load_initial_data['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    img = MF.utils.high_pass_filter_stamp(stamp,
                                          hpf,
                                          load_initial_data.loc[0, 'betaPic'].shape)

    # Exercise
    mf_result = MF.apply_matched_filter_fft(mf, img)
    # normalize by the sigma
    mf_snr = mf_result.std()/img.std()

    # Validate
    # check that the MF kills the noise by at least 50%
    assert(mf_snr < 0.5)

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
@pytest.mark.skip(reason="Currently fails")
def test_apply_matched_filter_fft_to_random_stamp_with_psf(load_initial_data, hpf):
    """
    Apply MF to a klip-subtracted stamp with no PSF injected
    """
    # Setup
    mf_template_psf = load_initial_data['psf'][hpf]
    mf = MF.create_matched_filter(mf_template_psf)

    MF.np.random.seed(random_seed)
    stamp = MF.np.random.normal(0, #load_initial_data['kl'][0].mean(),
                                load_initial_data['kl'][0].std(),
                                mf.size).reshape(mf.shape)
    flux_scale = 10 * load_initial_data['kl'][0].std()
    stamp = load_initial_data.loc[0, 'psf'] * flux_scale + stamp
    flux = MF.np.nansum(load_initial_data['psf'][0] * flux_scale)
    flux_psf = MF.RK.high_pass_filter(stamp, hpf)

    # Exercise
    mf_result = MF.apply_matched_filter(mf, flux_psf).max()

    # Validate
    # check that you get the flux back to better than 3%
    diff = MF.np.abs(1-mf_result/flux_scale)
    assert(diff < 0.03)
    #assert_almost_equal(mf_result/flux_scale, 1, 2)

    # Cleanup - none


@pytest.mark.parametrize('hpf', hpf_index)
@pytest.mark.skip("Not sure what the best way to test this is, actually")
def test_high_pass_filter_stamp(load_initial_data, hpf):
    """
    Make sure that the high pass filter gets filtered correctly
    To do this, check that the padded version gives a similar answer to
    the width-rescaled version?
    """
    # Setup
    initial_psf = load_initial_data['psf'][0]
    # reference case - rescale the filter width
    rescale_hpf = initial_psf.shape[-1]/load_initial_data['hr8799'][0].shape[-1]
    ref_filt = MF.RK.high_pass_filter(initial_psf, hpf * rescale_hpf)

    # Exercise
    # test case
    test_filt = MF.utils.high_pass_filter_stamp(initial_psf, hpf, load_initial_data['hr8799'][0].shape)

    # Validate
    # chi^2, the error is estimated as sqrt(original psf)**2 ::shrug::
    diff = MF.np.sum(((ref_filt-test_filt)**2)/initial_psf) / initial_psf.size
    assert(diff < 1e-2)

    # cleanup - none
