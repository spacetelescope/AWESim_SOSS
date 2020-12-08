"""
NGHXRG - Teledyne HxRG Noise Generator

Modification History:

18 September 2020 Joe Filippazzo

  Made code more Pythonic, faster with Numpy array calls rather than list comprehensions,
and moved much of the heavy lifting caluclations from mknoise() out to dedicated methods.
Also added noise_sources property to track contributions from each noise source.

13 August 2020   Kevin Volk

  Modify this version to return the simulated ramp and to not change to 
unsigned integers.

11 August 2020   Kevin Volk

  Change the random number generation calls to use the new numpy interface.

23 July 2020    Kevin Volk

  A couple of bug fixes to the dark current image option were made.  The code 
was changed so it will work properly with a sub-array simulation.  Previously 
the values in the lower left corner were used not the values at the target 
position.

22 June 2020    Kevin Volk

  Port to python version 3.  Added writing out bias_pattern.fits.

16 September 2019   Kevin Volk

  Add an option to use an image for the dark current rather than a constant 
value.

2 June 2017   Kevin Volk

  I have added in pedestal offsets; the self.pedestal_drift variable in the code, 
described as the "magnitude of the pedestal drift in electrons" does not appear 
to be used in the original code.  I added a Gaussian offset per channel per 
frame based on this parameter.  Finally, I added an optional noise seed so one 
can reproduce a calculation.

26 May 2017   Kevin Volk

  I have added a uniform dark current to the code (electrons/frame).  I also added 
in a gain factor.

16 May 2017   Kevin Volk

  Some changes were made for use with NIRISS...the pedestal file name has 
been changed and a couple of the defaults have been changed as well.

8-15 April 2015, B.J. Rauscher, NASA/GSFC
- Implement Pierre Ferruit's (ESA/ESTEC)recommendation to use
  numpy.fft.rfft and numpy.fft.irfft for faster execution. This saves
  about 30% in execution time.
- Clean up setting default values per several suggestions from
  Chaz Shapiro (NASA/JPL).
- Change from setting the shell variable H2RG_PCA0 to point to the PCA-zero
  file to having a shell variable point to the NG home directory. This is per
  a suggestion from Chaz Shapiro (NASA/JPL) that would allow seamlessly adding  
  more PCA-zero templates.
- Implement a request form Chaz Shapiro (NASA/JPL) for status
  reporting. This was done by adding the "verbose" arguement.
- Implement a request from Pierre Ferruit (ESA/ESTEC) to generate
  3-dimensional data cubes.
- Implement a request from Pierre Ferruit to treat ACN as different 1/f
  noise in even/odd columns. Previously ACN was treated purely as a feature
  in Fourier space.
- Version 2(Beta)

16 April 2015, B.J. Rauscher
- Fixed a bug in the pinkening filter definitions. Abs() was used where
  sqrt() was intended. The bug caused power spectra to have the wrong shape at
  low frequency.
- Version 2.1(Beta)

17 April 2015, B.J. Rauscher
- Implement a request from Chaz Shapiro for HXRGNoise() to exit gracefully if
  the pca0_file is not found.
- Version 2.2 (Beta)

8 July 2015, B.J. Rauscher
- Address PASP referee comments
    * Fast scan direction is now reversible. To reverse the slow scan
      direction use the numpy flipud() function.
    * Modifications to support subarrays. Specifically,
        > Setting reference_pixel_border_width=0 (integer zero);
            + (1) eliminates the reference pixel border and
            + (2) turns off adding in a bias pattern
                  when simulating data cubes. (Turned on in v2.5)
- Version 2.4

12 Oct 2015, J.M. Leisenring, UA/Steward
- Make compatible with Python 2.x
    * from __future__ import division
- Included options for subarray modes (FULL, WINDOW, and STRIPE)
    * Keywords x0 and y0 define a subarray position (lower left corner)
    * Selects correct pca0 region for subarray underlay
    * Adds reference pixels if they exist within subarray window
- Tie negative values to 0 and anything >=2^16 to 2^16-1
- Version 2.5

20 Oct 2015, J.M. Leisenring, UA/Steward
- Padded nstep to the next power of 2 in order to improve FFT runtime
    * nstep2 = int(2**np.ceil(np.log2(nstep)))
    * Speeds up FFT calculations by ~5x
- Don't generate noise elements if their magnitudes are equal to 0.
- mknoise() now returns copy of final HDU result for easy retrieval
- Version 2.6

"""
from copy import copy
import datetime
import math
import warnings

from hotsoss import utils
import numpy as np
from scipy.ndimage.interpolation import zoom

from .jwst_utils import SUB_SLICE, add_refpix

warnings.filterwarnings('ignore')


def add_signal(signals, cube, pyimage, frametime, gain, zodi, zodi_scale, photon_yield=False):
    """
    Add the science signal to the generated noise

    Parameters
    ----------
    signals: sequence
        The science frames
    cube: sequence
        The generated dark ramp
    pyimage: sequence
        The photon yield per order
    frametime: float
        The number of seconds per frame
    gain: float
        The detector gain
    zodi: sequence
        The zodiacal background image
    zodi_scale: float
        The scale factor for the zodi background
    """
    # Get the data dimensions
    dims1 = cube.shape
    dims2 = signals.shape
    if dims1 != dims2:
        raise ValueError(dims1, "not equal to", dims2)

    # Make a new ramp
    newcube = np.zeros_like(cube, dtype=np.float32)

    # The background is assumed to be in electrons/second/pixel, not ADU/s/pixel.
    background = zodi * zodi_scale * frametime

    # Iterate over each group
    for n in range(dims1[0]):
        framesignal = signals[n, :, :] * gain * frametime

        # Add photon yield
        if photon_yield:
            newvalues = np.random.poisson(framesignal)
            target = pyimage - 1.
            for k in range(dims1[1]):
                for l in range(dims1[2]):
                    if target[k, l] > 0.:
                        n = int(newvalues[k, l])
                        values = np.random.poisson(target[k, l], size=n)
                        newvalues[k, l] = newvalues[k, l] + np.sum(values)
            newvalues = newvalues + np.random.poisson(background)

        # Or don't
        else:
            vals = np.abs(framesignal * pyimage + background)
            newvalues = np.random.poisson(vals)

        # First ramp image
        if n == 0:
            newcube[n, :, :] = newvalues
        else:
            newcube[n, :, :] = newcube[n - 1, :, :] + newvalues

    newcube = cube + newcube / gain

    return newcube


def unlinearize(image, nonlinearity, sat, maxiter=10, accuracy=0.000001):
    """
    Add non linearity to the ramp images
    Modified from Bryan Hilbert's code: https://github.com/spacetelescope/mirage/blob/master/mirage/ramp_generator/unlinearize.py

    Parameters
    ----------
    image: array-like
        The image to unlinearize
    nonlinearity: array-like
        The 3D coefficients from the nonlinearity reference file
    sat: array-like
        Saturation map for the nonlinear ramps
    maxiter: int
        The maximum iterations
    accuracy: float
        The allowed accuracy

    Returns
    -------
    np.ndarray
        The unlinearized ramp data
    """
    # Check if the sizes of the satmap or coeffs are different than the data
    if sat.shape != image.shape[-2:]:
        raise ValueError(("WARNING: image y,x shape is {}, but input saturation map shape is {}.".format(image.shape[-2:], sat.shape)))

    # Get the image shape
    shape = image.shape

    # Transpose linearity coefficient array (x and y switched in ref file)
    # Reverse coefficients, x, and y dimensions (all 3 are backwards in ref file)
    # Trim coeffs to subarray (ref file is FULL frame)
    coeffs = np.transpose(nonlinearity, (0, 2, 1))
    coeffs = coeffs[::-1, ::-1, ::-1]
    sl = SUB_SLICE['SUBSTRIP256' if shape[1] == 256 else 'SUBSTRIP96' if shape[1] == 96 else 'FULL']
    coeffs = coeffs[:, sl, :]

    # REWORK SO THAT THE LINARIZED SATURATION MAP IS AN INPUT
    # RATHER THAN BEING CREATED HERE. THIS IS BECAUSE THE SUPERBIAS
    # AND REFPIX SIGNALS MUST BE SUBTRACTED BEFORE LINEARIZING
    # THE ORIGINAL RAW SATURATION MAP

    # Translate to saturation maps for the linear ramps here so
    # that we can pay attention only to non-saturated pixels in
    # the input linear image. Do this by applying the standard
    # linearity correction to the saturation map
    lin_satmap = nonLinFunc(sat, coeffs, sat)

    # Find pixels with "good" signals, to have the nonlin applied.
    # Negative pix or pix with signals above the requested max
    # value will not be changed.
    x = np.copy(image)
    i1 = np.where((image > 0.) & (image < lin_satmap))
    dev = np.zeros_like(image, dtype=float)
    dev[i1] = 1.
    i2 = np.where((image <= 0.) | (image >= lin_satmap))
    numhigh = np.where(image >= lin_satmap)
    i = 0

    # Initial run of the nonlin function - when calling the
    # non-lin function, give the original satmap for the
    # non-linear signal values
    val = nonLinFunc(image, coeffs, sat)
    val[i2] = 1.

    x[i1] = (image[i1] + image[i1] / val[i1]) / 2.
    while i < maxiter:
        i = i + 1
        val = nonLinFunc(x, coeffs, sat)
        val[i2] = 1.
        dev[i1] = np.abs(image[i1] / val[i1] - 1.)
        inds = np.where(dev[i1] > accuracy)
        if inds[0].size < 1:
            break
        val1 = nonLinDeriv(x, coeffs, sat)
        val1[i2] = 1.
        x[i1] = x[i1] + (image[i1] - val[i1]) / val1[i1]

    return x


class HXRGNoise:
    """
    HXRGNoise is a class for making realistic Teledyne HxRG system
    noise. The noise model includes correlated, uncorrelated,
    stationary, and non-stationary components. The default parameters
    make noise that resembles Channel 1 of JWST NIRSpec. NIRSpec uses
    H2RG detectors. They are read out using four video outputs at
    1.e+5 pix/s/output.
    """
    # These class variables are common to all HxRG detectors
    nghxrg_version = 2.6 # Sofware version

    def __init__(self, subarray, ngrps, dt=1.e-5, nroh=12, nfoh=1, reverse_scan_direction=False, verbose=False):
        """
        Simulate Teledyne HxRG+SIDECAR ASIC system noise.

        Parameters
        ----------
        subarray: str
            The NIRISS subarray, ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
        ngrps: int
            Number of groups per integration
        nfoh: int
            New frame overhead in rows. This allows for a short wait at the end of a frame before
            starting the next one.
        nroh: int
            New row overhead in pixels. This allows for a short wait at the end of a row before
            starting the next one.
        dt: float
            Pixel dwell time in seconds
        verbose: bool
            Enable this to provide status reporting
        reverse_scan_direction: bool
            Enable this to reverse the fast scanner eadout directions. This capability was added to support
            Teledyne's programmable fast scan readout directions. The default setting =False corresponds to
            what HxRG detectors default to upon power up.
        """
        # Save all arguments as attributes
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})

        # Subarray Mode
        subarrays = ['SUBSTRIP96', 'SUBSTRIP256', 'FULL']
        if subarray not in subarrays:
            raise ValueError("{} not a valid subarray. Please use {}".format(subarray, subarrays))

        # Get subarray detector settings
        self.sub_specs = utils.subarray_specs(subarray)
        self.row_slice = SUB_SLICE[self.subarray]
        self.nrows = self.sub_specs['y']
        self.ncols = self.sub_specs['x']
        self.ref_all = np.array([self.sub_specs[i] for i in ['y1', 'y2', 'x1', 'x2']])
        self.x0 = 0
        self.y0 = 0 if subarray == 'FULL' else 1792
        self.x1 = 4
        self.x2 = 2040
        self.y1 = 4 if subarray == 'FULL' else 1792
        self.y2 = self.y1 + self.nrows - self.sub_specs['y2']
        self.n_out = 4 if subarray == 'FULL' else 1

        # Compute the number of pixels in the fast-scan direction per output
        self.xsize = self.nrows // self.n_out

        # Compute the number of time steps per integration, per output
        self.nstep = (self.xsize + self.nroh) * (self.ncols + self.nfoh) * self.ngrps

        # Pad nsteps to a power of 2, which is much faster (JML)
        self.nstep2 = int(2**np.ceil(np.log2(self.nstep)))

        # For adding in ACN, it is handy to have masks of the even
        # and odd pixels on one output neglecting any gaps
        self.m_even = np.zeros((self.ngrps, self.ncols, self.xsize))
        self.m_odd = np.zeros_like(self.m_even)
        for x in np.arange(0, self.xsize, 2):
            self.m_even[:, :self.ncols, x] = 1
            self.m_odd[:, :self.ncols, x + 1] = 1
        self.m_even = np.reshape(self.m_even, np.size(self.m_even))
        self.m_odd = np.reshape(self.m_odd, np.size(self.m_odd))

        # Also for adding in ACN, we need a mask that point to just
        # the real pixels in ordered vectors of just the even or odd
        # pixels
        self.m_short = np.zeros((self.ngrps, self.ncols + self.nfoh, (self.xsize + self.nroh) // 2))
        self.m_short[:, :self.ncols, :self.xsize // 2] = 1
        self.m_short = np.reshape(self.m_short, np.size(self.m_short))

        # Define frequency arrays
        self.f1 = np.fft.rfftfreq(self.nstep2) # Frequencies for nstep elements
        self.f2 = np.fft.rfftfreq(2 * self.nstep2) # ... for 2*nstep elements

        # Define pinkening filters. F1 and p_filter1 are used to
        # generate ACN. F2 and p_filter2 are used to generate 1/f noise.
        self.alpha = -1 # Hard code for 1/f noise until proven otherwise
        self.p_filter1 = np.sqrt(self.f1**self.alpha)
        self.p_filter2 = np.sqrt(self.f2**self.alpha)
        self.p_filter1[0] = 0.
        self.p_filter2[0] = 0.

        # Reset the calculations
        self.reset()

    def reset(self):
        """
        Reset calculations
        """
        # Dictionary for tracking noise contributions
        sources = ['dark_current', 'bias_pattern', 'read_noise', 'corr_pink_noise', 'uncorr_pink_noise',
                   'alt_col_noise', 'pca0_noise', 'pedestal_drift']
        self.noise_sources = {source: [] for source in sources}
        self.nints = 0

    def calculate_dark_current(self, dark_current, gain, floor=0.0000001):
        """
        Generate dark current image

        Parameters
        ----------
        dark_current: str, array_like
            The dark current data or reference file to use
        gain: float
            The gain in electrons/ADU
        floor: float
            The minimum dark current value for a pixel
        """
        # Read in the dark current data
        if isinstance(dark_current, str):
            dark_current = fits.getdata(dark_current)

        # Take median dark current image and multiply by the gain
        dark_current = np.median(dark_current, axis=0) * gain

        # Set dark current floor
        dark_current[np.where(dark_current <= floor)] = floor

        # Save at attribute
        self.dark_current = dark_current

    def calculate_pca0(self, superbias):
        """
        Generate pca0 image from the superbias reference file

        One needs to take the superbias reference file, find the mean and standard deviation values,
        subtract the mean value, divide by the standard deviation value, and then transform to the
        raw detector orientation.

        Parameters
        ----------
        superbias: str, array_like
            The superbias data or reference file to use
        """
        # Read in the superbias data
        if isinstance(superbias, str):
            superbias = fits.getdata(superbias_file)

        # Get the mean and standard deviation
        superbias_mean = np.nanmean(superbias)
        superbias_std = np.nanstd(superbias)

        # Normalize the superbias
        superbias_norm = (superbias - superbias_mean) / superbias_std

        # Set transposed pca0 array and STD
        self.pca0 = superbias_norm.T
        self.pca0_amp = np.nanstd(self.pca0)

    def message(self, message_text):
        """
        Used for status reporting

        Parameters
        ----------
        message_text: str
            The message to print
        """
        if self.verbose:
            print('NG: ' + message_text + ' at DATETIME = ', datetime.datetime.now().time())

    def white_noise(self, mygen, nstep=None):
        """
        Generate white noise for an HxRG including all time steps (actual pixels and overheads).

        Parameters
        ----------
        mygen: numpy.random._generator.Generator
            The random seed generator
        nstep: int
            Length of vector returned
        """
        return(mygen.standard_normal(nstep))

    def pink_noise(self, mygen, mode):
        """
        Generate a vector of non-periodic pink noise.

        Parameters
        ----------
        mygen: numpy.random._generator.Generator
            The random seed generator
        mode: str
            Pink noise mode, ['pink', 'acn']
        """
        # Configure depending on mode setting
        if mode == 'pink':
            nstep = 2 * self.nstep
            nstep2 = 2 * self.nstep2 # JML
            f = self.f2
            p_filter = self.p_filter2

        else:
            nstep = self.nstep
            nstep2 = self.nstep2 # JML
            f = self.f1
            p_filter = self.p_filter1

        # Generate seed noise
        mynoise = self.white_noise(mygen, nstep2)

        # Save the mean and standard deviation of the first
        # half. These are restored later. We do not subtract the mean
        # here. This happens when we multiply the FFT by the pinkening
        # filter which has no power at f=0.
        the_mean = np.mean(mynoise[:nstep2 // 2])
        the_std = np.std(mynoise[:nstep2 // 2])

        # Apply the pinkening filter.
        thefft = np.fft.rfft(mynoise)
        thefft = np.multiply(thefft, p_filter)
        result = np.fft.irfft(thefft)
        result = result[:nstep // 2] # Keep 1st half of nstep

        # Restore the mean and standard deviation
        result *= the_std / np.std(result)
        result = result - np.mean(result) + the_mean

        return(result)

    def mknoise(self, rd_noise=5.0, pedestal_drift=4, c_pink=3, u_pink=1, acn=0.5, gain=1., superbias=None, reference_pixel_noise_ratio=0.8, ktc_noise=29., bias_offset=20994.06, bias_amp=5358.87, dark_current=None, dc_seed=0, noise_seed=0, skip=[]):
        """
        Generate a FITS cube containing only noise and the optional dark signal.

        Note1:
        Because of the noise correlations, there is no simple way to
        predict the noise of the simulated images. However, to a
        crude first approximation, these components add in
        quadrature.

        Note2:
        The units are mostly "electrons". This follows convention
        in the astronomical community. From a physics perspective, holes are
        actually the physical entity that is collected in Teledyne's p-on-n
        (p-type implants in n-type bulk) HgCdTe architecture.

        Parameters
        ----------
        pedestal_drift:float
            Magnitude of pedestal drift in electrons
        rd_noise: float
            Standard deviation of read noise in electrons
        c_pink: float
            Standard deviation of correlated pink noise in electrons
        u_pink: float
            Standard deviation of uncorrelated pink noise in electrons
        acn: float
            Standard deviation of alternating column noise in electrons
        reference_pixel_noise_ratio: float
            Ratio of the standard deviation of the reference pixels to the regular pixels.
            Reference pixels are usually a little lower noise.
        ktc_noise: float
            kTC noise in electrons. Set this equal to sqrt(k*T*C_pixel)/q_e, where k is Boltzmann's
            constant, T is detector temperature, and C_pixel is pixel capacitance. For an H2RG,
            the pixel capacitance is typically about 40 fF.
        bias_offset: float
            On average, integrations start here in electrons. Set this so that all pixels are in range.
        bias_amp: float
            A multiplicative factor that we multiply PCA-zero by to simulate a bias pattern.
            This is completely independent from adding in "picture frame" noise.
        dark_current: float, array-like
            The dark current signal in electrons/frame, a single value or an image of size 2048x2048
        dc_seed: int
            A seed value for the Poission noise generation
        gain: float
            Gain value in electrons/ADU
        noise_seed: float
            Seed value for the noise generation, to allow a simulation to be repeated
        skip: sequence
            The names of the noise sources to skip

        Returns
        -------
        array
            The resulting noise cube
        """
        self.message('Starting mknoise()')

        # Set noise parameters
        self.rd_noise = rd_noise
        self.pedestal_drift = pedestal_drift
        self.c_pink = c_pink
        self.u_pink = u_pink
        self.acn = acn
        self.dc_seed = dc_seed if dc_seed > 0. else int(4294967280. * np.random.uniform()) + 10
        self.gain = max(gain, 1.)
        self.noise_seed = noise_seed
        self.bias_offset = bias_offset
        self.bias_amp = bias_amp
        self.reference_pixel_noise_ratio = reference_pixel_noise_ratio
        self.ktc_noise = ktc_noise

        # Generate pca0
        self.calculate_pca0(superbias)

        # Generate dark current
        self.calculate_dark_current(dark_current, self.gain)

        if self.dc_seed == self.noise_seed:
            self.dc_seed = int(math.sqrt(self.dc_seed))

        # Seed generators
        rseed1 = np.random.SeedSequence(self.noise_seed)
        mygen = np.random.default_rng(rseed1)
        rseed2 = np.random.SeedSequence(self.dc_seed)
        darkgen = np.random.default_rng(rseed2)

        # Initialize the result cube
        result = np.zeros((self.ngrps, self.ncols, self.nrows), dtype=np.float32)

        # ==========================================================================
        # bias_pattern =============================================================
        # ==========================================================================
        if 'bias_pattern' not in skip:

            self.message('Adding bias_pattern')

            # Inject the bias pattern. Works for WINDOW and STRIPE (JML)
            bias_pattern = self.pca0 * self.bias_amp + self.bias_offset

            # Add in some kTC noise. Since this should always come out
            # in calibration, we do not attempt to model it in detail.
            bias_pattern += self.ktc_noise * mygen.standard_normal((self.ncols, self.nrows))

            # Add in the bias pattern
            for z in np.arange(self.ngrps):
                result[z, :, :] += bias_pattern

        else:

            self.message('Skipping bias_pattern')
            bias_pattern = np.zeros_like(result)

        # Save it
        self.noise_sources['bias_pattern'].append(self.noise_stats(result - bias_pattern))
        del bias_pattern

        # ==========================================================================
        # read_noise ===============================================================
        # ==========================================================================
        # Make white read noise. This is the same for all pixels.
        pre_rdnoise = copy(result)
        if self.rd_noise > 0 and 'read_noise' not in skip:

            self.message('Adding read_noise')
            w = self.ref_all
            r = self.reference_pixel_noise_ratio
            for z in np.arange(self.ngrps):
                read_noise = np.zeros((self.ncols, self.nrows))

                # Noisy reference pixels for each side of detector
                if w[0] > 0: # lower
                    read_noise[:w[0], :] = r * self.rd_noise *  mygen.standard_normal((w[0], self.nrows))
                if w[1] > 0: # upper
                    read_noise[-w[1]:, :] = r * self.rd_noise *  mygen.standard_normal((w[1], self.nrows))
                if w[2] > 0: # left
                    read_noise[:, :w[2]] = r * self.rd_noise *  mygen.standard_normal((self.ncols, w[2]))
                if w[3] > 0: # right
                    read_noise[:, -w[3]:] = r * self.rd_noise * mygen.standard_normal((self.ncols, w[3]))

                # Noisy regular pixels
                if np.sum(w) > 0: # Ref. pixels exist in frame
                    read_noise[w[0]:self.ncols-w[1],w[2]:self.nrows-w[3]] = self.rd_noise * mygen.standard_normal((self.ncols - w[0] - w[1], self.nrows - w[2] - w[3]))
                else: # No Ref. pixels, so add only regular pixels
                    read_noise = self.rd_noise * mygen.standard_normal((self.ncols, self.nrows))

                # Add the noise in to the result
                result[z, :, :] += read_noise

        else:

            self.message('Skipping read_noise')

        # Save it
        self.noise_sources['read_noise'].append(self.noise_stats(result - pre_rdnoise))
        del pre_rdnoise

        # ==========================================================================
        # corr_pink_noise ==========================================================
        # ==========================================================================
        # Add correlated pink noise.
        pre_cpink = copy(result)
        if self.c_pink > 0 and 'corr_pink_noise' not in skip:
            self.message('Adding corr_pink_noise noise')
            corr_pink = self.c_pink * self.pink_noise(mygen, 'pink')
            corr_pink = np.reshape(corr_pink, (self.ngrps, self.ncols + self.nfoh, self.xsize + self.nroh))[:, :self.ncols, :self.xsize]
            for op in np.arange(self.n_out):
                x0 = op * self.xsize
                x1 = x0 + self.xsize

                # By default fast-scan readout direction is [-->,<--,-->,<--]
                # If reverse_scan_direction is True, then [<--,-->,<--,-->]
                # Would be nice to include option for all --> or all <--
                modnum = 1 if self.reverse_scan_direction else 0
                if np.mod(op,2) == modnum:
                    result[:, :, x0:x1] += corr_pink
                else:
                    result[:, :, x0:x1] += corr_pink[:, :, ::-1]

        else:

            self.message('Skipping corr_pink_noise')

        # Save it
        self.noise_sources['corr_pink_noise'].append(self.noise_stats(result - pre_cpink))
        del pre_cpink

        # ==========================================================================
        # uncorr_pink_noise ========================================================
        # ==========================================================================
        # Add uncorrelated pink noise. Because this pink noise is stationary and
        # different for each output, we don't need to flip it.
        pre_upink = copy(result)
        if self.u_pink > 0 and 'uncorr_pink_noise' not in skip:
            self.message('Adding uncorr_pink_noise noise')
            for op in np.arange(self.n_out):
                x0 = op * self.xsize
                x1 = x0 + self.xsize
                uncorr_pink = self.u_pink * self.pink_noise(mygen, 'pink')
                uncorr_pink = np.reshape(uncorr_pink, (self.ngrps, self.ncols + self.nfoh, self.xsize + self.nroh))[:, :self.ncols, :self.xsize]
                result[:, :, x0:x1] += uncorr_pink

        else:

            self.message('Skipping uncorr_pink_noise')

        # Save it
        self.noise_sources['uncorr_pink_noise'].append(self.noise_stats(result - pre_upink))
        del pre_upink

        # ==========================================================================
        # alt_col_noise ============================================================
        # ==========================================================================
        # Add ACN
        pre_acn = copy(result)
        if self.acn > 0 and 'alt_col_noise' not in skip:
            self.message('Adding alt_col_noise noise')
            for op in np.arange(self.n_out):

                # Generate new pink noise for each even and odd vector.
                # We give these the abstract names 'a' and 'b' so that we
                # can use a previously worked out formula to turn them
                # back into an image section.
                a = self.acn * self.pink_noise(mygen, 'acn')
                b = self.acn * self.pink_noise(mygen, 'acn')

                # Pick out just the real pixels (i.e. ignore the gaps)
                a = a[np.where(self.m_short == 1)]
                b = b[np.where(self.m_short == 1)]

                # Reformat into an image section. This uses the formula
                # mentioned above.
                acn_cube = np.reshape(np.transpose(np.vstack((a,b))), (self.ngrps,self.ncols,self.xsize))

                # Add in the ACN. Because pink noise is stationary, we can
                # ignore the readout directions. There is no need to flip
                # acn_cube before adding it in.
                x0 = op * self.xsize
                x1 = x0 + self.xsize
                result[:, :, x0:x1] += acn_cube

        else:

            self.message('Skipping alt_col_noise')

        # Save it
        self.noise_sources['alt_col_noise'].append(self.noise_stats(result - pre_acn))
        del pre_acn

        # ==========================================================================
        # pca0_noise ===============================================================
        # ==========================================================================
        # Add PCA-zero. The PCA-zero template is modulated by 1/f.
        pre_pca0 = copy(result)
        if self.pca0_amp > 0 and 'pca0_noise' not in skip:
            self.message('Adding pca0_noise')
            gamma = self.pink_noise(mygen, mode='pink')
            zoom_factor = self.ncols * self.ngrps / np.size(gamma)
            gamma = zoom(gamma, zoom_factor, order=1, mode='mirror')
            gamma = np.reshape(gamma, (self.ngrps, self.ncols))
            for z in np.arange(self.ngrps):
                for y in np.arange(self.ncols):
                    result[z, y, :] += self.pca0_amp * self.pca0[y, :] * gamma[z, y]

        else:

            self.message('Skipping pca0_noise')

        # Save it
        self.noise_sources['pca0_noise'].append(self.noise_stats(result - pre_pca0))
        del pre_pca0

        # ==========================================================================
        # pedestal_drift ===========================================================
        # ==========================================================================
        # Add in channel offsets
        pre_drift = copy(result)
        if self.pedestal_drift > 0. and 'pedestal_drift' not in skip:
            self.message('Adding pedestal_drift')
            offsets = mygen.standard_normal((self.n_out, self.ngrps))
            for z in range(self.ngrps):
                x0 = 0
                for n in range(self.n_out):
                    result[z, :, x0:x0 + self.xsize] += self.pedestal_drift * offsets[n, z]
                    x0 += self.xsize

        else:

            self.message('Skipping pedestal_drift')

        # Save it
        self.noise_sources['pedestal_drift'].append(self.noise_stats(result - pre_drift))
        del pre_drift

        # ==========================================================================
        # dark_current =============================================================
        # ==========================================================================
        # Add in dark current
        pre_dark = copy(result)
        if self.dark_current is not None and 'dark_current' not in skip:
            self.message('Adding dark_current')

            # Generate dark current with Poisson distribution sampling
            dark = darkgen.poisson(self.dark_current, (self.ngrps, self.nrows, self.ncols))

            # Make dark current count cumulative
            dark = np.cumsum(dark, axis=0)

            # Zero out reference pixels
            dark = add_refpix(dark)

            for n1 in range(self.nrows):
                for n2 in range(self.ncols):
                    values = darkgen.poisson(lam=self.dark_current[n1, n2], size=self.ngrps)
                    cvalues = np.cumsum(values)
                    result[:, n2, n1] += cvalues

            # Add dark current to data
            result += np.transpose(dark, (0, 2, 1))

        else:

            self.message('Skipping dark_current')

        # Save it
        self.noise_sources['dark_current'].append(list(np.nanmean(result - pre_dark, axis=(1, 2))))
        del pre_dark

        # ==========================================================================
        # ==========================================================================
        # ==========================================================================

        # If the data cube has only 1 frame, reformat into a 2-dimensional image
        if self.ngrps == 1:
            self.message('Reformatting cube into image')
            result = result[0, :, :]

        # Add gain
        result /= self.gain

        # Transpose to (frame, nrows, ncols)
        result = np.transpose(result, (0, 2, 1))

        # Add to integration count
        self.nints += 1

        return result

    @staticmethod
    def noise_stats(data, func=np.nanmax, axis=(1, 2)):
        """
        Calculate some statistic for the given noise data

        Parameters
        ----------
        data: array-like
            The data to analyze
        func: function
            The function to use
        axis: tuple, int
            The axis over which to apply the function

        Returns
        -------
        float
            The calculated statistic
        """
        return list(func(data, axis=axis))


def make_photon_yield(photon_yield, orders):
    """
    Generates a map of the photon yield for each order.
    The shape of both arrays should be [order, nrows, ncols]

    Parameters
    ----------
    photon_yield: array-like
        An array for the calculated photon yield at each pixel
    orders: sequence
        An array of the median image of each order

    Returns
    -------
    np.ndarray
        The array containing the photon yield map for each order
    """
    # Get the shape and create empty arrays
    dims = orders.shape
    sum1 = np.zeros((dims[1], dims[2]), dtype=np.float32)
    sum2 = np.zeros((dims[1], dims[2]), dtype=np.float32)

    # Add the photon yield for each order
    for n in range(dims[0]):
        sum1 = sum1 + photon_yield[n, :, :] * orders[n, :, :]
        sum2 = sum2 + orders[n, :, :]

    # Take the ratio of the photon yield to the signal
    pyimage = sum1 / sum2
    pyimage[np.where(sum2 == 0.)] = 1.

    return pyimage


def nonLinFunc(image, coeffs, limits):
    """
    Apply linearity correction coefficients

    Parameters
    ----------
    image
    coeffs
    limits

    Returns
    -------

    """
    # to image.
    values = np.copy(image)
    bady = 0
    badx = 1
    if len(image.shape) == 3:
        bady = 1
        badx = 2
    elif len(image.shape) == 4:
        bady = 2
        badx = 3

    bad = np.where(values > limits)
    values[bad] = limits[bad[bady], bad[badx]]
    ncoeff = coeffs.shape[0]
    t = np.copy(coeffs[-1, :, :])
    for i in range(ncoeff - 2, -1, -1):
        t = coeffs[i, :, :] + values * t

    return t


def nonLinDeriv(image, coeffs, limits):
    """
    First derivative of non-lin correction

    Parameters
    ----------
    image
    coeffs
    limits

    Returns
    -------

    """
    # function
    values = np.copy(image)
    bady = 0
    badx = 1
    if len(image.shape) == 3:
        bady = 1
        badx = 2
    elif len(image.shape) == 4:
        bady = 2
        badx = 3

    bad = np.where(values > limits)
    values[bad] = limits[bad[bady], bad[badx]]
    ncoeff = coeffs.shape[0]
    t = (ncoeff - 1) * np.copy(coeffs[-1, :, :])
    for i in range(ncoeff - 3, -1, -1):
        t = (i + 1) * coeffs[i + 1, :, :] + values * t

    return t
