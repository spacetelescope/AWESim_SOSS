"""
A module to generate simulated 2D time-series SOSS data

Authors: Joe Filippazzo, Kevin Volk, Jonathan Fraine, Michael Wolfe
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import batman
import astropy.units as q
import astropy.constants as ac
import multiprocessing
import time
import AWESim_SOSS
import inspect
import warnings
import datetime
import webbpsf
from . import generate_darks as gd
from ExoCTK import svo
from ExoCTK import core
from ExoCTK.ldc import ldcfit as lf
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage.interpolation import zoom
from functools import partial
from sklearn.externals import joblib

warnings.simplefilter('ignore')

cm = plt.cm
FILTERS = svo.filters()
DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
FRAME_TIMES = {'SUBSTRIP96':2.213, 'SUBSTRIP256':5.491, 'FULL':10.737}
SUBARRAY_Y = {'SUBSTRIP96':96, 'SUBSTRIP256':256, 'FULL':2048}

def dist(p0, Poly):
    """
    Calculate the minimum Euclidean distance from a point to a given polynomial
    
    Parameters
    ----------
    p0: sequence
        The (x,y) coordinate of the point
    Poly: sequence
        The (X,Y) coordinates of the trace center in each column
    
    Returns
    -------
    float
        The minimum distance from pixel (x,y) to the polynomial points (X,Y)
    """
    # Calculate the distance from each point on the line to p0
    distances = np.sqrt((p0[0]-Poly[0])**2 + (p0[1]-Poly[1])**2)
    d_min = np.min(distances)
    
    # Check if above or below the polynomial
    d_min *= float(np.sign(p0[1] - Poly[1][np.argmin(distances)]))
    
    return d_min

def distance_map(order, coeffs='', subarr='SUBSTRIP256', generate=False, plot=False):
    """
    Generate a map where each pixel is the distance from the trace polynomial
    
    Parameters
    ----------
    order: int
        The order
    coeffs: sequence (optional)
        Custom polynomial coefficients of the trace
    subarr: str
        The subarray to use, ['SUBSTRIP96','SUBSTRIP256','FULL']
    plot: bool
        Plot the distance map
    
    Returns
    -------
    np.ndarray
        An array the same shape as masked_data
    
    Example
    -------
    The default polynomials for CV3 data are
    order1 = [1.71164931e-11, -9.29379122e-08, 1.91429367e-04, -1.43527531e-01, 7.13727478e+01]
    order2 = [2.35705513e-13, -2.62302311e-08, 1.65517682e-04, -3.19677081e-01, 2.81349581e+02]
    """
    # Set the polynomial coefficients to use
    if not coeffs:
            
        if order==1:
            coeffs = [1.71164931e-11, -9.29379122e-08, 1.91429367e-04, -1.43527531e-01, 7.13727478e+01]
        elif order==2:
            coeffs = [2.35705513e-13, -2.62302311e-08, 1.65517682e-04, -3.19677081e-01, 2.81349581e+02]
        else:
            print('Order {} not supported.'.format(order))
            
    # If coefficients are provided, generate a new map
    if isinstance(coeffs, (list, tuple, np.ndarray)) and generate:
        
        print('Generating distance map for order {} with coefficients {}...'.format(order,coeffs))
        
        # Get the dimensions
        dims = (2048, SUBARRAY_Y[subarr])
        
        # Generate an array of pixel coordinates
        flat = np.zeros(list(dims)+[2])
        for i in range(4,2044):
            for j in range(dims[1]):
                flat[i,j] = (i,j)
        flat = flat.reshape(np.prod(dims),2)
        
        # Make the (X,Y) coordinates of the polynomial on an oversampled grid
        X = np.arange(4, 2044, 0.1)
        Y = np.polyval(coeffs, X)
        
        # Set pixel independent inputs of distance function
        func = partial(dist, Poly=(X,Y))
        
        # Run multiprocessing
        pool = multiprocessing.Pool(8)
        
        # Generate the distance at each pixel
        d_map = pool.map(func, flat)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Reshape into frame
        d_map = np.asarray(d_map).reshape(dims).T
        
        # Write to file
        joblib.dump(d_map, DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    # Or just use the stored map
    else:
        d_map = joblib.load(DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    if plot:
        plt.figure(figsize=(13,2))
        plt.title('Order {}'.format(order))
        plt.imshow(d_map, interpolation='none', origin='lower')
        try:
            plt.plot(X, Y)
            plt.xlim(0,2048)
            plt.ylim(0,dims[1])
        except:
            pass
        plt.colorbar()
    
    return d_map

def generate_psf(filt, wavelength, oversample=4, to1D=False, plot=False, save=''):
    """
    Generate the SOSS psf with 'CLEAR' or 'F277W' filter
    
    Parameters
    ----------
    filt: str
        The filter to use, 'CLEAR' or 'F277W'
    wavelength: float, sequence
        The wavelength in microns
    oversample: int
        The factor by which the pixel grid will be mode finely sampled
    plot: bool
        Plot the 1D and 2D psf for visual inspection
    
    Returns
    -------
    np.ndarray
        The 1D psf
    """
    print("Generating the psf with {} filter and GR700XD pupil mask...".format(filt))
    
    # Get the NIRISS class from webbpsf and set the filter
    ns = webbpsf.NIRISS()
    ns.filter = filt
    ns.pupil_mask = 'GR700XD'
    
    # For case where one wavelength is specified
    if isinstance(wavelength, (int,float)):
        wavelength = [wavelength]
    
    # Make an array for all psfs
    psfs = []
    for w in wavelength:
        print('... at {} um ...'.format(w))
        psf2D = ns.calcPSF(monochromatic=w*1E-6, oversample=oversample)[0].data
        psfs.append(psf2D)
    psfs = np.asarray(psfs)
        
    # Collapse to 1D
    if to1D:
        psfs = np.sum(psfs, axis=-2)
        
    # # Plot it
    # if plot:
    #     plt.figure(figsize=(6,9))
    #     plt.suptitle('PSF for NIRISS GR700XD and {} filter'.format(filt))
    #     gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    #     ax1 = plt.subplot(gs[0])
    #     ax1.imshow(psf2D)
    #
    #     ax2 = plt.subplot(gs[1])
    #     ax2.plot(psf1D)
    #     ax2.set_xlim(0,psf2D.shape[0])
    #
    #     plt.tight_layout()
    
    return psfs.squeeze()

def get_frame_times(subarray, ngrps, nints, t0, nresets=1):
    """
    Calculate a time axis for the exposure in the given SOSS subarray
    
    Parameters
    ----------
    subarray: str
        The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
    ngrps: int
        The number of groups per integration
    nints: int
        The number of integrations for the exposure
    t0: float
        The start time of the exposure
    nresets: int
        The number of reset frames per integration
    
    Returns
    -------
    sequence
        The time of each frame
    """
    # Check the subarray
    if subarray not in ['SUBSTRIP256','SUBSTRIP96','FULL']:
        subarray = 'SUBSTRIP256'
        print("I do not understand subarray '{}'. Using 'SUBSTRIP256' instead.".format(subarray))
    
    # Get the appropriate frame time
    ft = FRAME_TIMES[subarray]
    
    # Generate the time axis, removing reset frames
    time_axis = []
    t = t0
    for _ in range(nints):
        times = t+np.arange(nresets+ngrps)*ft
        t = times[-1]+ft
        time_axis.append(times[nresets:])
    
    time_axis = np.concatenate(time_axis)
    
    return time_axis

def psf_position(distance, filt='CLEAR', generate=False, plot=False):
    """
    Scale the flux based on the pixel's distance from the center of the cross dispersed psf
    
    Parameters
    ----------
    distance: float
        The distance from the center of the pdf
    filt: str
        The filter used, 'CLEAR' or 'F277W'
    generate: bool
        Generate the psf from webbosf
    plot: bool
        Plot the psf and the interpolated position
    
    Returns
    -------
    float
        The interpolated value from the psf
    """
    # Generate the PSF from webbpsf
    if generate:
        
        # Generate the 1D psf
        psf1D = generate_psf(filt)
        
    # Or just use these
    else:
        
        if filt=='F277W':
            psf1D = np.array([ 0.00019988, 0.0002117 , 0.00021934, 0.00023641, 0.00026375, 0.0003073 , 0.00033975, 0.0003496 , 0.00038376, 0.00044702, 0.00047626, 0.00046785, 0.0004981 , 0.000573 , 0.00067364, 0.00076275, 0.00088491, 0.00100102, 0.00111439, 0.00132894, 0.00162288, 0.00188021, 0.00240184, 0.00370172, 0.00555634, 0.0072683 , 0.01021882, 0.01486976, 0.02364649, 0.03089798, 0.03602238, 0.03632454, 0.02973947, 0.0199811 , 0.01537851, 0.01493862, 0.01572836, 0.02018245, 0.01958941, 0.01427322, 0.01350232, 0.01688971, 0.02418887, 0.03537919, 0.03975415, 0.03609305, 0.02878872, 0.02192827, 0.01334616, 0.00835515, 0.00543218, 0.00380858, 0.00249234, 0.00204165, 0.00180445, 0.00132019, 0.00095559, 0.00082456, 0.0008002 , 0.00074513, 0.00068732, 0.00060487, 0.00050018, 0.00046889, 0.00045214, 0.00042667, 0.00038588, 0.00037849, 0.00036246, 0.00032832, 0.00030491, 0.00028396, 0.00026219, 0.00023077, 0.00021248, 0.0001944 ])
            
        else:
            psf1D = np.array([ 0.00013901, 0.00016542, 0.00015689, 0.00015623, 0.00017186, 0.00016965, 0.00018987, 0.00019185, 0.00023556, 0.00023899, 0.00029408, 0.00029346, 0.00035087, 0.00038025, 0.00042468, 0.00051199, 0.00066475, 0.00065935, 0.00089182, 0.00121465, 0.00148197, 0.00200335, 0.00266229, 0.00321626, 0.00485754, 0.00829491, 0.01690362, 0.02678705, 0.03461085, 0.03111909, 0.03046135, 0.02457325, 0.01972383, 0.02033947, 0.01575621, 0.01368122, 0.01234645, 0.01643781, 0.0200006 , 0.01972283, 0.0190382 , 0.02177633, 0.02355946, 0.02783534, 0.02869785, 0.03584818, 0.0343261 , 0.02788289, 0.0176711 , 0.00835883, 0.00510581, 0.0038801 , 0.00286961, 0.00192759, 0.00153765, 0.00103414, 0.00083634, 0.00078 , 0.00067393, 0.00050412, 0.00049817, 0.00041538, 0.00037431, 0.00035373, 0.00027228, 0.00027909, 0.00019483, 0.00020813, 0.00019951, 0.00017749, 0.00016526, 0.00016733, 0.00015403, 0.00013206, 0.00013329, 0.00011637])
            
    # Scale the transmission to 1
    psf = psf1D/np.trapz(psf1D)
    
    # Shift the psf so that the points go from -38 to 38
    l = len(psf)
    x = np.linspace(-1*l/2., l/2., l)
    
    # Interpolate lpsf to distance
    val = np.interp(distance, x, psf)
    
    if plot:
        plt.plot(x, psf)
        plt.scatter(distance, val, c='r', zorder=5)
        
    return val

def lambda_lightcurve(wavelength, response, psf_loc, pfd2adu, ld_coeffs, ld_profile, star, planet, time, params, filt, trace_radius=25, snr=100, floor=2, plot=False, verbose=False):
    """
    Generate a lightcurve for a given wavelength
    
    Parameters
    ----------
    wavelength: float
        The wavelength value in microns
    response: float
        The spectral response of the detector at the given wavelength
    psf_loc: float
        The location on the psf given the Euclidean distance from the trace center
    ld_coeffs: array-like
        A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
    ld_profile: str
        The limb darkening profile to use
    pfd2adu: sequence
        The factor that converts photon flux density to ADU/s
    star: sequence
        The wavelength and flux of the star
    planet: sequence
        The wavelength and Rp/R* of the planet at t=0 
    t: sequence
        The time axis for the TSO
    params: batman.transitmodel.TransitParams
        The transit parameters of the planet
    filt: str
        The filter to apply, ['CLEAR','F277W']
    throughput: float
        The CLEAR or F277W filter throughput at the given wavelength
    trace_radius: int
        The radius of the trace
    snr: float
        The signal-to-noise for the observations
    floor: int
        The noise floor in counts
    extend: int
        The number of points to extend the lpsf wings by
    plot: bool
        Plot the lightcurve
    verbose: bool
        Print some details
    
    Returns
    -------
    sequence
        A 1D array of the lightcurve with the same length as *t* 
    """
    nframes = len(time)
    
    if not isinstance(ld_coeffs, list) or not isinstance(ld_coeffs, np.ndarray):
        ld_coeffs  = [ld_coeffs]
        ld_profile = 'linear'
    
    # If it's a background pixel, it's just noise
    if psf_loc>trace_radius \
    or psf_loc<-trace_radius \
    or wavelength<np.nanmin(star[0].value) \
    or (filt=='F277W' and wavelength<2.36989) \
    or (filt=='F277W' and wavelength>3.22972):
        
        # flux = np.abs(np.random.normal(loc=floor, scale=1, size=nframes))
        flux = np.repeat(floor, nframes)
        
    else:
        
        # I = (Stellar Flux)*(LDC)*(Transit Depth)*(Filter Throughput)*(PSF position)
        # Don't use astropy units! It quadruples the computing time!
        
        # Get the energy flux density [erg/s/cm2/A] at the given wavelength [um] at t=t0
        flux0 = np.interp(wavelength, star[0], star[1], left=0, right=0)
        if verbose:
            print('Energy Flux Density [erg/s/cm2/A]:',flux0)
        
        # Convert from energy flux density [erg/s/cm2/A] to photon flux density
        # [photons/s/cm2/A] by multiplying by (lambda/h*c)
        flux0 *= wavelength*503411665111.4543 # [1/erg*um]
        if verbose:
            print('Photon Flux Density [1/s/cm2/A]:',flux0)
        
        # Convert from photon flux density to ADU/s by multiplying by the wavelength 
        # interval [um/pixel] and primary mirror area [cm2], and dividing by the gain [e-/ADU]
        flux0 *= pfd2adu
        if verbose:
            print('Count Rate [ADU/s * 1/e-]:',flux0)
        
        # Expand to shape of time axis
        flux = np.repeat(flux0, nframes)
        
        # If there is a transiting planet...
        if not isinstance(planet,str):
            
            # Set the wavelength dependent orbital parameters
            params.limb_dark = ld_profile
            params.u = ld_coeffs
            
            # Set the radius at the given wavelength from the transmission spectrum (Rp/R*)**2
            tdepth = np.interp(wavelength, planet[0], planet[1])
            params.rp = np.sqrt(tdepth)
            
            # Generate the light curve for this pixel
            model = batman.TransitModel(params, time) 
            lightcurve = model.light_curve(params)
            
            # Scale the flux with the lightcurve
            flux *= lightcurve
            
        # Apply the filter response
        flux *= response
        
        # Scale pixel based on distance from the center of the cross-dispersed psf
        flux *= psf_loc
        
        # Replace very low signal pixels with noise floor
        flux[flux<floor] += np.repeat(floor, len(flux[flux<floor]))
        
        # Plot
        if plot:
            plt.plot(t, flux)
            plt.xlabel("Time from central transit")
            plt.ylabel("Flux Density [photons/s/cm2/A]")
        
    return flux

def ldc_lookup(ld_profile, grid_point, delta_w=0.005, nrows=256, save=''):
    """
    Generate a lookup table of limb darkening coefficients for full SOSS wavelength range
    
    Parameters
    ----------
    ld_profile: str
        A limb darkening profile name supported by `ExoCTK.ldc.ldcfit.ld_profile()`
    grid_point: dict, sequence
        The stellar parameters [Teff, logg, FeH] or stellar model dictionary from `ExoCTK.core.ModelGrid.get()`
    delta_w: float
        The width of the wavelength bins in microns
    save: str
        The path to save to file to
    
    Example
    -------
    from AWESim_SOSS.sim2D import awesim
    lookup = awesim.ldc_lookup('quadratic', [3300, 4.5, 0])
    """
    print("Go get a coffee! This takes about 5 minutes to run.")
    
    # Initialize the lookup table
    lookup = {}
    
    # Get the full wavelength range
    wave_maps = wave_solutions(nrows)
    
    # Get the grid point
    if isinstance(grid_point, (list,tuple,np.ndarray)):
        
        grid_point = core.ModelGrid(os.environ['MODELGRID_DIR'], resolution=700, wave_rng=(0.6,2.6)).get(*grid_point)
        
    # Abort if no stellar dict
    if not isinstance(grid_point, dict):
        print('Please provide the grid_point argument as [Teff, logg, FeH] or ExoCTK.core.ModelGrid.get(Teff, logg, FeH).')
        return
        
    # Define function for multiprocessing
    def gr700xd_ldc(wavelength, delta_w, ld_profile, grid_point, model_grid):
        """
        Calculate the LCDs for the given wavelength range in the GR700XD grism
        """
        try:
            # Get the bandpass in that wavelength range
            mn = (wavelength-delta_w/2.)*q.um
            mx = (wavelength+delta_w/2.)*q.um
            throughput = np.genfromtxt(DIR_PATH+'/files/NIRISS.GR700XD.1.txt', unpack=True)
            bandpass = svo.Filter('GR700XD', throughput, n_bins=1, wl_min=mn, wl_max=mx, verbose=False)
            
            # Calculate the LDCs
            ldcs = lf.ldc(None, None, None, model_grid, [ld_profile], bandpass=bandpass, grid_point=grid_point.copy(), mu_min=0.08, verbose=False)
            coeffs = list(zip(*ldcs[ld_profile]['coeffs']))[1::2]
            coeffs = [coeffs[0][0],coeffs[1][0]]
            
            return ('{:.9f}'.format(wavelength), coeffs)
            
        except:
            
            print(wavelength)
            
            return ('_', None)
            
    # Pool the LDC calculations across the whole wavelength range for each order
    for order in [1,2]:
        
        # Get the wavelength limits for this order
        min_wave = np.nanmin(wave_maps[order-1][wave_maps[order-1]>0])
        max_wave = np.nanmax(wave_maps[order-1][wave_maps[order-1]>0])
        
        # Generate list of binned wavelengths
        wavelengths = np.arange(min_wave, max_wave, delta_w)
        
        # Turn off printing
        print('Calculating order {} LDCs at {} wavelengths...'.format(order,len(wavelengths)))
        sys.stdout = open(os.devnull, 'w')
        
        # Pool the LDC calculations across the whole wavelength range
        processes = 8
        start = time.time()
        pool = multiprocessing.pool.ThreadPool(processes)
        
        func = partial(gr700xd_ldc, 
                       delta_w    = delta_w,
                       ld_profile = ld_profile,
                       grid_point = grid_point,
                       model_grid = model_grid)
                       
        # Turn list of coeffs into a dictionary
        order_dict = dict(pool.map(func, wavelengths))
        
        pool.close()
        pool.join()
        
        # Add the dict to the master
        try:
            order_dict.pop('_')
        except:
            pass
        lookup['order{}'.format(order)] = order_dict
        
        # Turn printing back on
        sys.stdout = sys.__stdout__
        print('Order {} LDCs finished: '.format(order), time.time()-start)
        
    if save:
        t, g, m = grid_point['Teff'], grid_point['logg'], grid_point['FeH']
        joblib.dump(lookup, save+'/{}_ldc_lookup_{}_{}_{}.save'.format(ld_profile,t,g,m))
        
    else:
    
        return lookup

def ld_coefficient_map(lookup_file, subarray='SUBSTRIP256', save=''):
    """
    Generate  map of limb darkening coefficients at every NIRISS pixel for all SOSS orders
    
    Parameters
    ----------
    lookup_file: str
        The path to the lookup table of LDCs
    
    Example
    -------
    ld_coeffs_lookup = ld_coefficient_lookup(1, 'quadratic', star, model_grid)
    """
    # Get the lookup table
    ld_profile = os.path.basename(lookup_file).split('_')[0]
    lookup = joblib.load(lookup_file)
    
    # Get the wavelength map
    nrows = 256 if subarray=='SUBSTRIP256' else 96 if subarray=='SUBSTRIP96' else 2048
    wave_map = wave_solutions(nrows)
        
    # Make dummy array for LDC map results
    ldfunc = lf.ld_profile(ld_profile)
    ncoeffs = len(inspect.signature(ldfunc).parameters)-1
    ld_coeffs = np.zeros((3, nrows*2048, ncoeffs))
    
    # Calculate the coefficients at each pixel for each order
    for order,wavelengths in enumerate(wave_map[:1]):
        
        # Get a flat list of all wavelengths for this order
        wave_list = wavelengths.flatten()
        lkup = lookup['order{}'.format(order+1)]
        
        # Get the bin size
        delta_w = np.mean(np.diff(sorted(np.array(list(map(float,lkup))))))/2.
        
        # For each bin in the lookup table...
        for bin, coeffs in lkup.items():
            
            try:
                
                # Get all the pixels that fall within the bin
                w = float(bin)
                idx, = np.where(np.logical_and(wave_list>=w-delta_w,wave_list<=w+delta_w))
                
                # Place them in the coefficient map
                ld_coeffs[order][idx] = coeffs
                
            except:
                 
                print(bin)
                
    if save:
        path = lookup_file.replace('lookup','map')
        joblib.dump(ld_coeffs, path)
        
        print('LDC coefficient map saved at',path)
        
    else:
        
        return ld_coeffs

class TSO(object):
    """
    Generate NIRISS SOSS time series observations
    """

    def __init__(self, ngrps, nints, star,
                        planet      = '', 
                        params      = '', 
                        ld_coeffs   = '', 
                        ld_profile  = 'quadratic',
                        snr         = 700,
                        subarray    = 'SUBSTRIP256',
                        t0          = 0,
                        target      = ''):
        """
        Iterate through all pixels and generate a light curve if it is inside the trace
        
        Parameters
        ----------
        ngrps: int
            The number of groups per integration
        nints: int
            The number of integrations for the exposure
        star: sequence
            The wavelength and flux of the star
        planet: sequence (optional)
            The wavelength and Rp/R* of the planet at t=0 
        params: batman.transitmodel.TransitParams (optional)
            The transit parameters of the planet
        ld_coeffs: array-like (optional)
            A 3D array that assigns limb darkening coefficients to each pixel, i.e. wavelength
        ld_profile: str (optional)
            The limb darkening profile to use
        snr: float
            The signal-to-noise
        subarray: str
            The subarray name, i.e. 'SUBSTRIP256', 'SUBSTRIP96', or 'FULL'
        t0: float
            The start time of the exposure
        target: str (optional)
            The name of the target
                        
        Example
        -------
        from AWESim_SOSS.sim2D import awesim
        import astropy.units as q, os, AWESim_SOSS
        DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
        vega = np.genfromtxt(DIR_PATH+'/files/scaled_spectrum.txt', unpack=True) # A0V with Jmag=9
        vega = [vega[0]*q.um, (vega[1]*q.W/q.m**2/q.um).to(q.erg/q.s/q.cm**2/q.AA)]
        tso = awesim.TSO(3, 5, vega)
        tso.run_simulation()
        """
        # Set instance attributes for the exposure
        self.subarray     = subarray
        self.nrows        = 256 if '256' in subarray else 96 if '96' in subarray else 2048
        self.ncols        = 2048
        self.ngrps        = ngrps
        self.nints        = nints
        self.nresets      = 1
        self.frame_time   = FRAME_TIMES[subarray]
        self.time         = get_frame_times(subarray, ngrps, nints, t0, self.nresets)
        self.nframes      = len(self.time)
        self.target       = target or 'Simulated Target'
        self.obs_date     = ''
        self.filter       = 'CLEAR'
        self.header       = ''
        
        # ========================================================================
        # ========================================================================
        # Change this to accept StellarModel onbject for star and planet and
        # move planet params and transmission spectrum input to run_simulation()
        # ========================================================================
        # ========================================================================
        
        # Set instance attributes for the target
        self.star         = star
        self.planet       = planet
        self.params       = params
        self.ld_coeffs    = ld_coeffs or np.zeros((2, self.nrows*self.ncols, 2))
        self.ld_profile   = ld_profile or 'quadratic'
        self.snr          = snr
        self.wave         = wave_solutions(str(self.nrows))
        
        # Calculate a map for each order that converts photon flux density to ADU/s
        self.gain = 1.61 # [e-/ADU]
        self.primary_mirror = 253260 # [cm2]
        avg_wave = np.mean(self.wave, axis=1)
        self.pfd2adu = np.ones((3,self.ncols*self.nrows))
        for n,aw in enumerate(avg_wave):
            coeffs = np.polyfit(aw[:-1], np.diff(aw), 1)
            wave_int = (np.polyval(coeffs, self.wave[n])*q.um).to(q.AA)
            self.pfd2adu[n] = (wave_int*self.primary_mirror*q.cm**2/self.gain).to(q.cm**2*q.AA).value.flatten()
            
        # Add the orbital parameters as attributes
        for p in [i for i in dir(self.params) if not i.startswith('_')]:
            setattr(self, p, getattr(self.params, p))
            
        # Create the empty exposure
        dims = (self.nframes, self.nrows, self.ncols)
        self.tso = np.zeros(dims)
        self.tso_ideal = np.zeros(dims)
        self.tso_order1 = np.zeros(dims)
        self.tso_order2 = np.zeros(dims)
    
    def run_simulation(self, orders=[1,2], filt='CLEAR', noise=True):
        """
        Generate the simulated 2D data given the initialized TSO object
        
        Parameters
        ----------
        orders: sequence
            The orders to simulate
        filt: str
            The element from the filter wheel to use, i.e. 'CLEAR' or 'F277W'
        noise: bool
            Run add_noise method to generate ramps with noise
        """
        # Set single order to list
        if isinstance(orders,int):
            orders = [orders]
        if not all([o in [1,2] for o in orders]):
            raise TypeError('Order must be either an int, float, or list thereof; i.e. [1,2]')
        orders = list(set(orders))
        
        # Check if it's F277W to speed up calculation
        if 'F277W' in filt.upper():
            orders = [1]
            self.filter = 'F277W'
            
        # If there is a planet transmission spectrum but no LDCs, generate them
        if not isinstance(self.planet, str) and not any(self.ld_coeffs):
            
            # Generate the lookup table
            lookup = ldc_lookup(self.ld_profile, [3300, 4.5, 0])
            
            # Generate the coefficient map
            self.ld_coeffs = ld_coefficient_map(lookup, subarray=self.subarray)
            
        # Generate simulation for each order
        for order in orders:
            
            # Get the wavelength map
            wave = self.wave[order-1].flatten()
            
            # Get limb darkening map
            ld_coeffs = self.ld_coeffs.copy()[order-1]
            
            # Get the psf location given the distance from the trace center
            distance = distance_map(order=order).flatten()
            psf_loc = psf_position(distance, filt=self.filter)
            
            # Get relative spectral response map
            throughput = np.genfromtxt(DIR_PATH+'/files/gr700xd_{}_order{}.dat'.format(self.filter,order), unpack=True)
            response = np.interp(wave, throughput[0], throughput[-1], left=0, right=0)
            
            # Get the wavelength interval per pixel map
            pfd2adu = self.pfd2adu[order-1]
            
            # print(isinstance(ld_coeffs[0], float), ld_coeffs)
            
            if isinstance(ld_coeffs[0], float):
                ld_coeffs = np.transpose([[ld_coeffs[0], ld_coeffs[1]]] * wave.size)
            
            # Run multiprocessing
            print('Calculating order {} light curves...'.format(order))
            start = time.time()
            pool = multiprocessing.Pool(8)
            
            # Set wavelength independent inputs of lightcurve function
            func = partial(lambda_lightcurve,
                           ld_profile    = self.ld_profile,
                           star          = self.star,
                           planet        = self.planet,
                           time          = self.time,
                           params        = self.params,
                           filt          = self.filter,
                           snr           = self.snr)
                           
            # Generate the lightcurves at each pixel
            lightcurves = pool.starmap(func, zip(wave, response, psf_loc, pfd2adu, ld_coeffs))
            
            # Close the pool
            pool.close()
            pool.join()
            
            # Clean up and time of execution
            tso_order = np.asarray(lightcurves).swapaxes(0,1).reshape([self.nframes, self.nrows, self.ncols])
            
            print('Order {} light curves finished: '.format(order), time.time()-start)
            
            # Add to the master TSO
            self.tso += tso_order
            
            # Add it to the individual order
            setattr(self, 'tso_order{}'.format(order), tso_order)
            
        # Add noise to the observations using Kevin Volk's dark ramp simulator
        self.tso_ideal = self.tso.copy()
        
        # Add noise and ramps
        if noise:
            self.add_noise()
    
    def add_noise(self, zodi_scale=1., offset=500):
        """
        Generate ramp and background noise
        
        Parameters
        ----------
        zodi_scale: float
            The scale factor of the zodiacal background
        offset: int
            The dark current offset
        """
        print('Adding noise to TSO...')
        
        # Get the separated orders
        orders = np.asarray([self.tso_order1,self.tso_order2])
        
        # Load all the reference files
        photon_yield = fits.getdata(DIR_PATH+'/files/photon_yield_dms.fits')
        pca0_file = DIR_PATH+'/files/niriss_pca0.fits'
        zodi = fits.getdata(DIR_PATH+'/files/soss_zodiacal_background_scaled.fits')
        nonlinearity = fits.getdata(DIR_PATH+'/files/substrip256_forward_coefficients_dms.fits')
        pedestal = fits.getdata(DIR_PATH+'/files/substrip256pedestaldms.fits')
        darksignal = fits.getdata(DIR_PATH+'/files/substrip256signaldms.fits')*self.gain
        
        # Generate the photon yield factor values
        pyf = gd.make_photon_yield(photon_yield, np.mean(orders, axis=1))
        
        # Remove negatives from the dark ramp
        darksignal[np.where(darksignal < 0.)] = 0.
        
        # Make the exposure
        RAMP = gd.make_exposure(1, self.ngrps, darksignal, self.gain, pca0_file=pca0_file, offset=offset)
        
        # Iterate over integrations
        for n in range(self.nints):
            
            # Add in the SOSS signal
            ramp = gd.add_signal(self.tso_ideal[self.ngrps*n:self.ngrps*n+self.ngrps], RAMP.copy(), pyf, self.frame_time, self.gain, zodi, zodi_scale, photon_yield=False)
            
            # apply the non-linearity function
            ramp = gd.non_linearity(ramp, nonlinearity, offset=offset)
            
            # add the pedestal to each frame in the integration
            ramp = gd.add_pedestal(ramp, pedestal, offset=offset)
            
            # Update the TSO with one containing noise
            self.tso[self.ngrps*n:self.ngrps*n+self.ngrps] = ramp
        
    def plot_frame(self, frame='', scale='linear', order='', noise=True, cmap=cm.jet):
        """
        Plot a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        scale: str
            Plot in linear or log scale
        order: int (optional)
            The order to isolate
        noise: bool
            Plot with the noise model
        cmap: str
            The color map to use
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            if noise:
                tso = self.tso
            else:
                tso = self.tso_ideal
        
        vmax = int(np.nanmax(tso))
        
        plt.figure(figsize=(13,2))
        if scale=='log':
            plt.imshow(tso[frame or self.nframes//2].data, origin='lower', interpolation='none', norm=matplotlib.colors.LogNorm(), vmin=1, vmax=vmax, cmap=cmap)
        else:
            plt.imshow(tso[frame or self.nframes//2].data, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('Injected Spectrum')
    
    def plot_snr(self, frame='', cmap=cm.jet):
        """
        Plot the SNR of a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        cmap: matplotlib.cm.colormap
            The color map to use
        """
        # Get the SNR
        snr  = np.sqrt(self.tso[frame or self.nframes//2].data)
        vmax = int(np.nanmax(snr))
        
        # Plot it
        plt.figure(figsize=(13,2))
        plt.imshow(snr, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title('SNR over Spectrum')
        
    def plot_saturation(self, frame='', saturation=80.0, cmap=cm.jet):
        """
        Plot the saturation of a TSO frame
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        saturation: float
            Percentage of full well that defines saturation
        cmap: matplotlib.cm.colormap
            The color map to use
        """
        # The full well of the detector pixels
        fullWell = 65536.0
        
        # Get saturated pixels
        saturated = np.array(self.tso[frame or self.nframes//2].data) > (saturation/100.0) * fullWell
        
        # Plot it
        plt.figure(figsize=(13,2))
        plt.imshow(saturated, origin='lower', interpolation='none', cmap=cmap)
        plt.colorbar()
        plt.title('{} Saturated Pixels'.format(len(saturated[saturated>fullWell])))
    
    def plot_slice(self, col, trace='tso', frame=0, order='', **kwargs):
        """
        Plot a column of a frame to see the PSF in the cross dispersion direction
        
        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot a light curve for
        trace: str
            The attribute name to plot
        frame: int
            The frame number to plot
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
            
        f = tso[frame].T
        
        if isinstance(col, int):
            col = [col]
            
        for c in col:
            plt.plot(f[c], label='Column {}'.format(c), **kwargs)
            
        plt.xlim(0,256)
        
        plt.legend(loc=0, frameon=False)
        
    def plot_ramp(self):
        """
        Plot the total flux on each frame to display the ramp
        """
        plt.figure()
        plt.plot(np.sum(self.tso, axis=(1,2)), ls='none', marker='o')
        plt.xlabel('Group')
        plt.ylabel('Count Rate [ADU/s]')
        plt.grid()
        
    def plot_lightcurve(self, col):
        """
        Plot a lightcurve for each column index given
        
        Parameters
        ----------
        col: int, float, sequence
            The integer column index(es) or float wavelength(s) in microns 
            to plot as a light curve
        """
        # Get the scaled flux in each column
        f = np.nansum(self.tso, axis=1)
        f = f/np.nanmax(f, axis=1)[:,None]
        
        # Make it into an array
        if isinstance(col, (int,float)):
            col = [col]
            
        for c in col:
            
            # If it is an index
            if isinstance(c, int):
                lc = f[:,c]
                label = 'Col {}'.format(c)
                
            # Or assumed to be a wavelength in microns
            elif isinstance(c, float):
                W = np.mean(self.wave[0], axis=0)
                lc = [np.interp(c, W, F) for F in f]
                label = '{} um'.format(c)
                
            else:
                print('Please enter an index, astropy quantity, or array thereof.')
                return
            
            plt.plot(self.time, lc, label=label, marker='.', ls='None')
            
        plt.legend(loc=0, frameon=False)
        
    def plot_spectrum(self, frame=0, order=''):
        """
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
        
        # Get extracted spectrum (Column sum for now)
        wave = np.mean(self.wave[0], axis=0)
        flux = np.sum(tso[frame].data, axis=0)
        
        # Deconvolve with the grism
        throughput = np.genfromtxt(DIR_PATH+'/files/gr700xd_{}_order{}.dat'.format(self.filter,order or 1), unpack=True)
        flux *= np.interp(wave, throughput[0], throughput[-1], left=0, right=0)
        
        # Convert from ADU/s to photon flux density
        wave_int = np.diff(wave)*q.um.to(q.AA)
        flux /= (np.array(list(wave_int)+[wave_int[-1]])*self.primary_mirror*q.cm**2/self.gain).value.flatten()
        
        # Convert from photon flux density to energy flux density
        flux /= wave*503411665111.4543 # [1/erg*um]
        
        # Plot it along with input spectrum
        plt.figure(figsize=(13,5))
        plt.loglog(wave, flux, label='Extracted')
        plt.loglog(*self.star, label='Injected')
        plt.xlim(wave[0]*0.95,wave[-1]*1.05)
        plt.legend()
    
    def save(self, filename='dummy.save'):
        """
        Save the TSO data to file
        
        Parameters
        ----------
        filename: str
            The path of the save file
        """
        print('Saving TSO class dict to {}'.format(filename))
        joblib.dump(self.__dict__, filename)
    
    def load(self, filename):
        """
        Load a previously calculated TSO
        
        Paramaters
        ----------
        filename: str
            The path of the save file
        
        Returns
        -------
        awesim.TSO()
            A TSO class dict
        """
        print('Loading TSO class dict to {}'.format(filename))
        load_dict = joblib.load(filename)
        # for p in [i for i in dir(load_dict)]:
        #     setattr(self, p, getattr(params, p))
        for key in load_dict.keys():
            exec("self." + key + " = load_dict['" + key + "']")
    
    def to_fits(self, outfile):
        """
        Save the data to a JWST pipeline ingestible FITS file
        
        Parameters
        ----------
        outfile: str
            The path of the output file
        """
        # Make the cards
        cards = [('DATE', datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S"), 'Date file created yyyy-mm-ddThh:mm:ss, UTC'),
                ('FILENAME', outfile, 'Name of the file'),
                ('DATAMODL', 'RampModel', 'Type of data model'),
                ('ORIGIN', 'STScI', 'Institution responsible for creating FITS file'),
                ('TIMESYS', 'UTC', 'principal time system for time-related keywords'),
                ('FILETYPE', 'uncalibrated', 'Type of data in the file'),
                ('SDP_VER', '2016_1', 'data processing software version number'),
                ('PRD_VER', 'PRDDEVSOC-D-012', 'S&OC PRD version number used in data processing'),
                ('TELESCOP', 'JWST', 'Telescope used to acquire data'),
                ('RADESYS', 'ICRS', 'Name of the coordinate reference frame'),
                ('', '', ''),
                ('COMMENT', '/ Program information', ''),
                ('TITLE', 'UNKNOWN', 'Proposal title'),
                ('PI_NAME', 'N/A', 'Principal investigator name'),
                ('CATEGORY', 'UNKNOWN', 'Program category'),
                ('SUBCAT', '', 'Program sub-category'),
                ('SCICAT', '', 'Science category assigned during TAC process'),
                ('CONT_ID', 0, 'Continuation of previous program'),
                ('', '', ''),
                ('COMMENT', '/ Observation identifiers', ''),
                ('DATE-OBS', self.obs_date, 'UT date at start of exposure'),
                ('TIME-OBS', self.obs_date, 'UT time at the start of exposure'),
                ('OBS_ID', 'V87600007001P0000000002102', 'Programmatic observation identifier'),
                ('VISIT_ID', '87600007001', 'Visit identifier'),
                ('PROGRAM', '87600', 'Program number'),
                ('OBSERVTN', '001', 'Observation number'),
                ('VISIT', '001', 'Visit number'),
                ('VISITGRP', '02', 'Visit group identifier'),
                ('SEQ_ID', '1', 'Parallel sequence identifier'),
                ('ACT_ID', '02', 'Activity identifier'),
                ('EXPOSURE', '1', 'Exposure request number'),
                ('', '', ''),
                ('COMMENT', '/ Visit information', ''),
                ('TEMPLATE', 'NIRISS SOSS', 'Proposal instruction template used'),
                ('OBSLABEL', 'Observation label', 'Proposer label for the observation'),
                ('VISITYPE', '', 'Visit type'),
                ('VSTSTART', self.obs_date, 'UTC visit start time'),
                ('WFSVISIT', '', 'Wavefront sensing and control visit indicator'),
                ('VISITSTA', 'SUCCESSFUL', 'Status of a visit'),
                ('NEXPOSUR', 1, 'Total number of planned exposures in visit'),
                ('INTARGET', False, 'At least one exposure in visit is internal'),
                ('TARGOOPP', False, 'Visit scheduled as target of opportunity'),
                ('', '', ''),
                ('COMMENT', '/ Target information', ''),
                ('TARGPROP', '', "Proposer's name for the target"),
                ('TARGNAME', self.target, 'Standard astronomical catalog name for tar'),
                ('TARGTYPE', 'FIXED', 'Type of target (fixed, moving, generic)'),
                ('TARG_RA', 175.5546225, 'Target RA at mid time of exposure'),
                ('TARG_DEC', 26.7065694, 'Target Dec at mid time of exposure'),
                ('TARGURA', 0.01, 'Target RA uncertainty'),
                ('TARGUDEC', 0.01, 'Target Dec uncertainty'),
                ('PROP_RA', 175.5546225, 'Proposer specified RA for the target'),
                ('PROP_DEC', 26.7065694, 'Proposer specified Dec for the target'),
                ('PROPEPOC', '2000-01-01 00:00:00', 'Proposer specified epoch for RA and Dec'),
                ('', '', ''),
                ('COMMENT', '/ Exposure parameters', ''),
                ('INSTRUME', 'NIRISS', 'Identifier for niriss used to acquire data'),
                ('DETECTOR', 'NIS', 'ASCII Mnemonic corresponding to the SCA_ID'),
                ('LAMP', 'NULL', 'Internal lamp state'),
                ('FILTER', self.filter, 'Name of the filter element used'),
                ('PUPIL', 'GR700XD', 'Name of the pupil element used'),
                ('FOCUSPOS', 0.0, 'Focus position'),
                ('', '', ''),
                ('COMMENT', '/ Exposure information', ''),
                ('PNTG_SEQ', 2, 'Pointing sequence number'),
                ('EXPCOUNT', 0, 'Running count of exposures in visit'),
                ('EXP_TYPE', 'NIS_SOSS', 'Type of data in the exposure'),
                ('', '', ''),
                ('COMMENT', '/ Exposure times', ''),
                ('EXPSTART', self.time[0], 'UTC exposure start time'),
                ('EXPMID', self.time[len(self.time)//2], 'UTC exposure mid time'),
                ('EXPEND', self.time[-1], 'UTC exposure end time'),
                ('READPATT', 'NISRAPID', 'Readout pattern'),
                ('NINTS', self.nints, 'Number of integrations in exposure'),
                ('NGROUPS', self.ngrps, 'Number of groups in integration'),
                ('NFRAMES', self.nframes, 'Number of frames per group'),
                ('GROUPGAP', 0, 'Number of frames dropped between groups'),
                ('NSAMPLES', 1, 'Number of A/D samples per pixel'),
                ('TSAMPLE', 10.0, 'Time between samples (microsec)'),
                ('TFRAME', FRAME_TIMES[self.subarray], 'Time in seconds between frames'),
                ('TGROUP', FRAME_TIMES[self.subarray], 'Delta time between groups (s)'),
                ('EFFINTTM', 15.8826, 'Effective integration time (sec)'),
                ('EFFEXPTM', 15.8826, 'Effective exposure time (sec)'),
                ('CHRGTIME', 0.0, 'Charge accumulation time per integration (sec)'),
                ('DURATION', self.time[-1]-self.time[0], 'Total duration of exposure (sec)'),
                ('NRSTSTRT', self.nresets, 'Number of resets at start of exposure'),
                ('NRESETS', self.nresets, 'Number of resets between integrations'),
                ('ZEROFRAM', False, 'Zero frame was downlinkws separately'),
                ('DATAPROB', False, 'Science telemetry indicated a problem'),
                ('SCA_NUM', 496, 'Sensor Chip Assembly number'),
                ('DATAMODE', 91, 'post-processing method used in FPAP'),
                ('COMPRSSD', False, 'data compressed on-board (T/F)'),
                ('SUBARRAY', self.subarray, 'Subarray pattern name'),
                ('SUBSTRT1', 1, 'Starting pixel in axis 1 direction'),
                ('SUBSTRT2', 1793, 'Starting pixel in axis 2 direction'),
                ('SUBSIZE1', self.ncols, 'Number of pixels in axis 1 direction'),
                ('SUBSIZE2', self.nrows, 'Number of pixels in axis 2 direction'),
                ('FASTAXIS', -2, 'Fast readout axis direction'),
                ('SLOWAXIS', -1, 'Slow readout axis direction'),
                ('COORDSYS', '', 'Ephemeris coordinate system'),
                ('EPH_TIME', 57403, 'UTC time from ephemeris start time (sec)'),
                ('JWST_X', 1462376.39634336, 'X spatial coordinate of JWST (km)'),
                ('JWST_Y', -178969.457007469, 'Y spatial coordinate of JWST (km)'),
                ('JWST_Z', -44183.7683640854, 'Z spatial coordinate of JWST (km)'),
                ('JWST_DX', 0.147851665036734, 'X component of JWST velocity (km/sec)'),
                ('JWST_DY', 0.352194454527743, 'Y component of JWST velocity (km/sec)'),
                ('JWST_DZ', 0.032553742839182, 'Z component of JWST velocity (km/sec)'),
                ('APERNAME', 'NIS-CEN', 'PRD science aperture used'),
                ('PA_APER', -290.1, 'Position angle of aperture used (deg)'),
                ('SCA_APER', -697.500000000082, 'SCA for intended target'),
                ('DVA_RA', 0.0, 'Velocity aberration correction RA offset (rad)'),
                ('DVA_DEC', 0.0, 'Velocity aberration correction Dec offset (rad)'),
                ('VA_SCALE', 0.0, 'Velocity aberration scale factor'),
                ('BARTDELT', 0.0, 'Barycentric time correction'),
                ('BSTRTIME', 0.0, 'Barycentric exposure start time'),
                ('BENDTIME', 0.0, 'Barycentric exposure end time'),
                ('BMIDTIME', 0.0, 'Barycentric exposure mid time'),
                ('HELIDELT', 0.0, 'Heliocentric time correction'),
                ('HSTRTIME', 0.0, 'Heliocentric exposure start time'),
                ('HENDTIME', 0.0, 'Heliocentric exposure end time'),
                ('HMIDTIME', 0.0, 'Heliocentric exposure mid time'),
                ('WCSAXES', 2, 'Number of WCS axes'),
                ('CRPIX1', 1955.0, 'Axis 1 coordinate of the reference pixel in the'),
                ('CRPIX2', 1199.0, 'Axis 2 coordinate of the reference pixel in the'),
                ('CRVAL1', 175.5546225, 'First axis value at the reference pixel (RA in'),
                ('CRVAL2', 26.7065694, 'Second axis value at the reference pixel (RA in'),
                ('CTYPE1', 'RA---TAN', 'First axis coordinate type'),
                ('CTYPE2', 'DEC--TAN', 'Second axis coordinate type'),
                ('CUNIT1', 'deg', 'units for first axis'),
                ('CUNIT2', 'deg', 'units for second axis'),
                ('CDELT1', 0.065398, 'first axis increment per pixel, increasing east'),
                ('CDELT2', 0.065893, 'Second axis increment per pixel, increasing nor'),
                ('PC1_1', -0.5446390350150271, 'linear transformation matrix element cos(theta)'),
                ('PC1_2', 0.8386705679454239, 'linear transformation matrix element -sin(theta'),
                ('PC2_1', 0.8386705679454239, 'linear transformation matrix element sin(theta)'),
                ('PC2_2', -0.5446390350150271, 'linear transformation matrix element cos(theta)'),
                ('S_REGION', '', 'spatial extent of the observation, footprint'),
                ('GS_ORDER', 0, 'index of guide star within listed of selected g'),
                ('GSSTRTTM', '1999-01-01 00:00:00', 'UTC time when guide star activity started'),
                ('GSENDTIM', '1999-01-01 00:00:00', 'UTC time when guide star activity completed'),
                ('GDSTARID', '', 'guide star identifier'),
                ('GS_RA', 0.0, 'guide star right ascension'),
                ('GS_DEC', 0.0, 'guide star declination'),
                ('GS_URA', 0.0, 'guide star right ascension uncertainty'),
                ('GS_UDEC', 0.0, 'guide star declination uncertainty'),
                ('GS_MAG', 0.0, 'guide star magnitude in FGS detector'),
                ('GS_UMAG', 0.0, 'guide star magnitude uncertainty'),
                ('PCS_MODE', 'COARSE', 'Pointing Control System mode'),
                ('GSCENTX', 0.0, 'guide star centroid x postion in the FGS ideal'),
                ('GSCENTY', 0.0, 'guide star centroid x postion in the FGS ideal'),
                ('JITTERMS', 0.0, 'RMS jitter over the exposure (arcsec).'),
                ('VISITEND', '2017-03-02 15:58:45.36', 'Observatory UTC time when the visit st'),
                ('WFSCFLAG', '', 'Wavefront sensing and control visit indicator'),
                ('BSCALE', 1, ''),
                ('BZERO', 32768, '')]
        
        # Make the header
        prihdr = fits.Header()
        for card in cards:
            prihdr.append(card, end=True)
        
        # Store the header in the object too
        self.header = prihdr
        
        # Make the HDUList
        prihdu  = fits.PrimaryHDU(header=prihdr)
        sci_hdu = fits.ImageHDU(data=self.tso, name='SCI')
        hdulist = fits.HDUList([prihdu, sci_hdu])
        
        # Write the file
        hdulist.writeto(outfile, overwrite=True)
        hdulist.close()
        
        print('File saved as',outfile)

def wave_solutions(subarr, directory=DIR_PATH+'/files/soss_wavelengths_fullframe.fits'):
    """
    Get the wavelength maps for SOSS orders 1, 2, and 3
    This will be obsolete once the apply_wcs step of the JWST pipeline
    is in place.
     
    Parameters
    ==========
    subarr: str
        The subarray to return, accepts '96', '256', or 'full'
    directory: str
        The directory containing the wavelength FITS files
        
    Returns
    =======
    np.ndarray
        An array of the wavelength solutions for orders 1, 2, and 3
    """
    try:
        idx = int(subarr)
    except:
        idx = None
    
    wave = fits.getdata(directory).swapaxes(-2,-1)[:,:idx]
    
    return wave
    
# ================================================================================
# This is WIP code to alternately generate the SOSS trace using 2D webbpsf models
# ================================================================================
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data

def monochromatic(wavelength, filt, star_spectrum, star_params='', model_grid='', planet_spectrum='', planet_params=''):
    """
    For a given wavelength, generate the 2D psf an observer would see
    
    Parameters
    ----------
    wavelength: astropy.units.quantity.Quantity
        The wavelength value
    filt: str
        The filter to use, 'CLEAR' or 'F277W'
    stellar_params: sequence
        The [Teff, logg, FeH] of the simulated star
    model_grid: str, ExoCTK.core.ModelGrid
        The model grid to use
    """
    # Get the psf from webbpsf
    psf2d = generate_psf(filt, wavelength.to(q.um))
    
    # Get the limb darkening coefficients from ExoCTK.ldc
    teff, logg, feh = stellar_params
    ldcs = lf.ldc(teff, logg, feh, model_grid)

def trace_from_webbpsf(psf_file, coeffs=[1.71164931e-11, -9.29379122e-08, 1.91429367e-04, -1.43527531e-01, 7.13727478e+01], plot=True):
    """
    Construct the trace from the 2D psf generated by webbpsf at each wavelength
    
    Parameters
    ----------
    psf_file: str, np.ndarray
        The path to the numpy file with the psf at each wavelength
    offset: int
        A placeholder to set the trace in the center of the image
    plot: bool
        Plot the trace
    
    # TODO: Input a polynomial and then convolve the psf in each column with the trace center
    """
    # Load the psfs
    if isinstance(psf_file, str):
        psfs = np.load(psf_file)
    elif isinstance(psf_file, np.ndarray):
        psfs = psf_file
    else:
        print('Cannot read that data. Please input a .npy file or 3D numpy array.')
        return
        
    # Get the psf center (1/2 the 4x oversampled image = 1/8)
    c = int(psfs.shape[1]/8)
    
    # Empty trace (with padding for overflow)
    final = np.zeros((2048+2*c,256))
    
    # Place wavelength dependent psf in each column (i.e. wavelength)
    for n,p in enumerate(psfs):
        
        # Downsample
        z = zoom(p, 0.25)
        
        # Place the trace center in the correct column
        final[n:n+c*2,:c*2] += z
        
    # Transpose to DMS orientation
    final = final.T
    
    # Trim off padding
    final = final[:,c:-c]
    
    # Add curvature from polynomial
    final = warped(final, coeffs)
    
    # Plot it
    if plot:
        plt.figure(figsize=(13,2))
        plt.imshow(final, origin='lower')
        plt.xlim(0,2048)
        plt.ylim(0,256)
        
    return final
    
def warped(image, coeffs=[1.71164931e-11, -9.29379122e-08, 1.91429367e-04, -1.43527531e-01, 7.13727478e+01], downsample=50, plot=False):
    """
    Warp a 2D image 
    
    Or perhaps use skimage.transform.estimate_transform to find the transformation
    parameters between the wave_map and one where the columns are iso-wavelengths.
    Then use this to transform the input linear image.
    """
    # Get the image dimensions
    rows, cols = image.shape
    
    # Generate the control points
    src_cols = np.linspace(0, cols, cols//downsample)
    src_rows = np.linspace(0, rows, rows//downsample)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]
    
    # Calculate the y-intercept of the curved trace
    y_int = coeffs[-1]+2*76
    
    # Add curvature to control points
    dst_cols = src[:,0]
    dst_rows = src[:,1]+np.polyval(coeffs, dst_cols)-y_int
    dst = np.vstack([dst_cols, dst_rows]).T
    
    # Perform transform
    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out = warp(image, tform, output_shape=(rows, cols))
    
    # DMS coordinates
    out = out[::-1,::-1]
    
    # Fill in background with smallest non-zero value
    out[out==0] = np.min(out[out>0])
    
    if plot:
        plt.figure()
        plt.imshow(out, origin='lower', norm=matplotlib.colors.LogNorm())
        
    return out

# ================================================================================
# ================================================================================
# ================================================================================
