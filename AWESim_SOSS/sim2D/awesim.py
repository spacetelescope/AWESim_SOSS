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
from functools import partial
from sklearn.externals import joblib
from numpy.core.multiarray import interp as compiled_interp

warnings.simplefilter('ignore')

cm = plt.cm
FILTERS = svo.filters()
DIR_PATH = os.path.dirname(os.path.realpath(AWESim_SOSS.__file__))
FRAME_TIMES = {'SUBSTRIP96':2.213, 'SUBSTRIP256':5.491, 'FULL':10.737}

def ADUtoFlux(order):
    """
    Return the wavelength dependent conversion from ADUs to erg s-1 cm-2 
    in SOSS traces 1, 2, and 3
    
    Parameters
    ==========
    order: int
        The trace order, must be 1, 2, or 3
    
    Returns
    =======
    np.ndarray
        Arrays to convert the given order trace from ADUs to units of flux
    """
    ADU2mJy, mJy2erg = 7.586031e-05, 2.680489e-15
    scaling = np.genfromtxt(DIR_PATH+'/files/GR700XD_{}.txt'.format(order), unpack=True)
    scaling[1] *= ADU2mJy*mJy2erg
    
    return scaling

def norm_to_mag(spectrum, magnitude, bandpass):
    """
    Returns the flux of a given *spectrum* [W,F] normalized to the given *magnitude* in the specified photometric *band*
    """
    # Get the current magnitude and convert to flux
    mag, mag_unc = get_mag(spectrum, bandpass, fetch='flux')
    
    # Convert input magnitude to flux
    flx, flx_unc = mag2flux(bandpass.filterID.split('/')[1], magnitude, sig_m='', units=spectrum[1].unit)
    
    # Normalize the spectrum
    spectrum[1] *= np.trapz(bandpass.rsr[1], x=bandpass.rsr[0])*np.sqrt(2)*flx/mag
    
    return spectrum

def flux2mag(bandpass, f, sig_f='', photon=False):
    """
    For given band and flux returns the magnitude value (and uncertainty if *sig_f*)
    """
    eff = bandpass.WavelengthEff
    zp = bandpass.ZeroPoint
    unit = q.erg/q.s/q.cm**2/q.AA
    
    # Convert to f_lambda if necessary
    if f.unit == 'Jy':
        f,  = (ac.c*f/eff**2).to(unit)
        sig_f = (ac.c*sig_f/eff**2).to(unit)
    
    # Convert energy units to photon counts
    if photon:
        f = (f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg), 
        sig_f = (sig_f*(eff/(ac.h*ac.c)).to(1/q.erg)).to(unit/q.erg)
    
    # Calculate magnitude
    m = -2.5*np.log10((f/zp).value)
    sig_m = (2.5/np.log(10))*(sig_f/f).value if sig_f else ''
    
    return [m, sig_m]

def mag2flux(band, mag, sig_m='', units=q.erg/q.s/q.cm**2/q.AA):
    """
    Caluclate the flux for a given magnitude
    
    Parameters
    ----------
    band: str, svo.Filter
        The bandpass
    mag: float, astropy.unit.quantity.Quantity
        The magnitude
    sig_m: float, astropy.unit.quantity.Quantity
        The magnitude uncertainty
    units: astropy.unit.quantity.Quantity
        The unit for the output flux
    """
    try:
        # Get the band info
        filt = FILTERS.loc[band]
        
        # Make mag unitless
        if hasattr(mag,'unit'):
            mag = mag.value
        if hasattr(sig_m,'unit'):
            sig_m = sig_m.value
        
        # Calculate the flux density
        zp = q.Quantity(filt['ZeroPoint'], filt['ZeroPointUnit'])
        f = zp*10**(mag/-2.5)
        
        if isinstance(sig_m,str):
            sig_m = np.nan
        
        sig_f = f*sig_m*np.log(10)/2.5
            
        return [f, sig_f]
        
    except IOError:
        return [np.nan, np.nan]

def rebin_spec(spec, wavnew, oversamp=100, plot=False):
    """
    Rebin a spectrum to a new wavelength array while preserving 
    the total flux
    
    Parameters
    ----------
    spec: array-like
        The wavelength and flux to be binned
    wavenew: array-like
        The new wavelength array
        
    Returns
    -------
    np.ndarray
        The rebinned flux
    
    """
    nlam = len(spec[0])
    x0 = np.arange(nlam, dtype=float)
    x0int = np.arange((nlam-1.)*oversamp + 1., dtype=float)/oversamp
    w0int = np.interp(x0int, x0, spec[0])
    spec0int = np.interp(w0int, spec[0], spec[1])/oversamp
    try:
        err0int = np.interp(w0int, spec[0], spec[2])/oversamp
    except:
        err0int = ''
        
    # Set up the bin edges for down-binning
    maxdiffw1 = np.diff(wavnew).max()
    w1bins = np.concatenate(([wavnew[0]-maxdiffw1], .5*(wavnew[1::]+wavnew[0:-1]), [wavnew[-1]+maxdiffw1]))
    
    # Bin down the interpolated spectrum:
    w1bins = np.sort(w1bins)
    nbins = len(w1bins)-1
    specnew = np.zeros(nbins)
    errnew = np.zeros(nbins)
    inds2 = [[w0int.searchsorted(w1bins[ii], side='left'), w0int.searchsorted(w1bins[ii+1], side='left')] for ii in range(nbins)]

    for ii in range(nbins):
        specnew[ii] = np.sum(spec0int[inds2[ii][0]:inds2[ii][1]])
        try:
            errnew[ii] = np.sum(err0int[inds2[ii][0]:inds2[ii][1]])
        except:
            pass
            
    if plot:
        plt.figure()
        plt.loglog(spec[0], spec[1], c='b')    
        plt.loglog(wavnew, specnew, c='r')
        
    return [wavnew,specnew,errnew]

def get_mag(spectrum, bandpass, exclude=[], fetch='mag', photon=False, Flam=False, plot=False):
    """
    Returns the integrated flux of the given spectrum in the given band
    
    Parameters
    ---------
    spectrum: array-like
        The [w,f,e] of the spectrum with astropy.units
    bandpass: str, svo_filters.svo.Filter
        The bandpass to calculate
    exclude: sequecne
        The wavelength ranges to exclude by linear interpolation between gap edges
    photon: bool
        Use units of photons rather than energy
    Flam: bool
        Use flux units rather than the default flux density units
    plot: bool
        Plot it
    
    Returns
    -------
    list
        The integrated flux of the spectrum in the given band
    """
    # Get the Filter object if necessary
    if isinstance(bandpass, str):
        bandpass = svo.Filter(bandpass)
        
    # Get filter data in order
    unit = q.Unit(bandpass.WavelengthUnit)
    mn = bandpass.WavelengthMin*unit
    mx = bandpass.WavelengthMax*unit
    wav, rsr = bandpass.raw
    wav = wav*unit
    
    # Unit handling
    a = (1 if photon else q.erg)/q.s/q.cm**2/(1 if Flam else q.AA)
    b = (1 if photon else q.erg)/q.s/q.cm**2/q.AA
    c = 1/q.erg
    
    # Test if the bandpass has full spectral coverage
    if np.logical_and(mx < np.max(spectrum[0]), mn > np.min(spectrum[0])) \
    and all([np.logical_or(all([i<mn for i in rng]), all([i>mx for i in rng])) for rng in exclude]):
        
        # Rebin spectrum to bandpass wavelengths
        w, f, sig_f = rebin_spec([i.value for i in spectrum], wav.value)*spectrum[1].unit
        
        # Calculate the integrated flux, subtracting the filter shape
        F = (np.trapz((f*rsr*((wav/(ac.h*ac.c)).to(c) if photon else 1)).to(b), x=wav)/(np.trapz(rsr, x=wav))).to(a)
        
        # Caluclate the uncertainty
        if sig_f:
            sig_F = np.sqrt(np.sum(((sig_f*rsr*np.gradient(wav).value*((wav/(ac.h*ac.c)).to(c) if photon else 1))**2).to(a**2)))
        else:
            sig_F = ''
            
        # Make a plot
        if plot:
            plt.figure()
            plt.step(spectrum[0], spectrum[1], color='k', label='Spectrum')
            plt.errorbar(bandpass.WavelengthEff, F.value, yerr=sig_F.value, marker='o', label='Magnitude')
            try:
                plt.fill_between(spectrum[0], spectrum[1]+spectrum[2], spectrum[1]+spectrum[2], color='k', alpha=0.1)
            except:
                pass
            plt.plot(bandpass.rsr[0], bandpass.rsr[1]*F, label='Bandpass')
            plt.xlabel(unit)
            plt.ylabel(a)
            plt.legend(loc=0, frameon=False)
            
        # Get magnitude from flux
        m, sig_m = flux2mag(bandpass, F, sig_f=sig_F)
        
        return [m, sig_m, F, sig_F] if fetch=='both' else [F, sig_F] if fetch=='flux' else [m, sig_m]
        
    else:
        return ['']*4 if fetch=='both' else ['']*2

def ldc_lookup(ld_profile, grid_point, model_grid, delta_w=0.005, save=''):
    """
    Generate a lookup table of limb darkening coefficients for full SOSS wavelength range
    
    Parameters
    ----------
    ld_profile: str
        A limb darkening profile name supported by `ExoCTK.ldc.ldcfit.ld_profile()`
    grid_point: dict
        The stellar model dictionary from `ExoCTK.core.ModelGrid.get()`
    model_grid: ExoCTK.core.ModelGrid
        The model grid
    delta_w: float
        The width of the wavelength bins in microns
    save: str
        The path to save to file to
    
    Example
    -------
    import os
    from AWESim_SOSS.sim2D import awesim
    from ExoCTK import core
    grid = core.ModelGrid(os.environ['MODELGRID_DIR'], Teff_rng=(3000,4000), logg_rng=(4,5), FeH_rng=(0,0.5), resolution=700)
    model = G.get(3300, 4.5, 0)
    awesim.ldc_lookup('quadratic', model, grid, save='/Users/jfilippazzo/Desktop/')
    """
    print("Go get a coffee! This takes about 5 minutes to run.")
    
    # Initialize the lookup table
    lookup = {}
    
    # Get the full wavelength range
    wave_maps = wave_solutions(256)
    
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
    for order in [1,2,3]:
        
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

def ld_coefficient_map(lookup_file, subarray='SUBSTRIP256', save=True):
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

def trace_polynomial(trace, start=4, end=2040, order=4):
    # Make a scatter plot where the pixels in each column are offset by a small amount
    x, y = [], []
    for n,col in enumerate(trace.T):
        vals = np.where(~col)
        if vals:
            v = list(vals[0])
            y += v
            x += list(np.random.normal(n, 1E-16, size=len(v)))
            
    # Now fit a polynomial to it!
    height, length = trace.shape
    coeffs = np.polyfit(x[start:], y[start:], order)
    X = np.arange(start, length, 1)
    Y = np.polyval(coeffs, X)
    
    return X, Y

def distance_map(order, generate=False, start=4, end=2044, p_order=4, plot=False):
    """
    Generate a map where each pixel is the distance from the trace polynomial
    
    Parameters
    ----------
    plot: bool
        Plot the distance map
    
    Returns
    -------
    np.ndarray
        An array the same shape as masked_data
    
    """   
    # If missing, generate it
    if generate:
        
        print('Generating distance map...')
        
        mask = joblib.load(DIR_PATH+'/files/order{}_mask.save'.format(order)).swapaxes(-1,-2)
        
        # Get the trace polynomial
        X, Y = trace_polynomial(mask, start, end, p_order)
        
        # Get the distance from the pixel to the polynomial
        def dist(p0, Poly):
            return min(np.sqrt((p0[0]-Poly[0])**2 + (p0[1]-Poly[1])**2))
            
        # Make a map of pixel locations
        height, length = mask.shape
        d_map = np.zeros(mask.shape)
        for i in range(length):
            for j in range(height):
                d_map[j,i] = dist((j,i), (Y,X))
                
        joblib.dump(d_map, DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    else:
        d_map = joblib.load(DIR_PATH+'/files/order_{}_distance_map.save'.format(order))
        
    
    if plot:
        plt.figure(figsize=(13,2))
        
        plt.title('Order {}'.format(order))
        
        plt.imshow(d_map, interpolation='none', origin='lower', norm=matplotlib.colors.LogNorm())
        
        plt.colorbar()
    
    return d_map

def psf_position(distance, extend=25, generate=False, filt='CLEAR', plot=False):
    """
    Scale the flux based on the pixel's distance from the center of the cross dispersed psf
    """
    # Generate the PSF from webbpsf
    if generate:
        
        # Get the NIRISS class from webbpsf and set the filter
        ns = webbpsf.NIRISS()
        ns.filter = filt
        ns.pupil_mask = 'GR700XD'
        psf2D = ns.calcPSF(oversample=4)[0].data
        psf1D = np.sum(psf2D, axis=0)
        
    # Or just use this one
    else:
        
        psf1D = np.array([5.83481665e-05,   6.56322048e-05,   7.52470683e-05, \
                         7.02759033e-05,   7.86948234e-05,   7.56720214e-05, \
                         7.16950313e-05,   7.49320549e-05,   8.75974333e-05, \
                         8.49589852e-05,   9.78484741e-05,   9.54325778e-05, \
                         1.04021838e-04,   9.31272313e-05,   9.38132761e-05, \
                         1.06975481e-04,   1.13885427e-04,   1.45214913e-04, \
                         1.25916643e-04,   1.31508590e-04,   1.37958390e-04, \
                         1.56179923e-04,   1.59852140e-04,   1.55946280e-04, \
                         1.69979257e-04,   1.63448538e-04,   1.67324699e-04, \
                         2.02793184e-04,   2.14443303e-04,   2.38094250e-04, \
                         2.52850706e-04,   2.46207776e-04,   3.23323300e-04, \
                         3.22398985e-04,   3.72937770e-04,   3.27714231e-04, \
                         3.90219004e-04,   4.42950638e-04,   5.01561954e-04, \
                         6.25955792e-04,   7.10831339e-04,   7.66708646e-04, \
                         8.28292330e-04,   9.53694152e-04,   1.12110206e-03, \
                         1.56683295e-03,   1.72534924e-03,   2.07805420e-03, \
                         2.28743713e-03,   2.87378788e-03,   3.34490077e-03, \
                         4.37717313e-03,   5.99502637e-03,   8.07971823e-03, \
                         1.00328036e-02,   1.28667718e-02,   1.51740977e-02, \
                         1.63074419e-02,   1.53397374e-02,   1.61010793e-02, \
                         1.54381747e-02,   1.52025943e-02,   1.46467818e-02, \
                         1.36687241e-02,   1.31107948e-02,   1.26643624e-02, \
                         1.35217668e-02,   1.20246384e-02,   9.38761089e-03, \
                         8.20080732e-03,   7.80380494e-03,   7.46099537e-03, \
                         6.72187764e-03,   6.20459022e-03,   6.65443858e-03, \
                         8.48880604e-03,   9.27146992e-03,   8.43853098e-03, \
                         8.93369301e-03,   8.82120736e-03,   9.06529876e-03, \
                         8.40046950e-03,   9.22187873e-03,   9.89323673e-03, \
                         1.10144353e-02,   1.15221573e-02,   1.26997776e-02, \
                         1.41984438e-02,   1.47337816e-02,   1.59085234e-02, \
                         1.86350411e-02,   1.93995664e-02,   1.92512920e-02, \
                         1.77431473e-02,   1.55704866e-02,   1.18367102e-02, \
                         1.01309336e-02,   7.70381621e-03,   5.20222357e-03, \
                         3.92637114e-03,   3.07334265e-03,   2.46599767e-03, \
                         2.11022681e-03,   1.45088608e-03,   1.47390548e-03, \
                         1.21404256e-03,   1.05247860e-03,   8.57261004e-04, \
                         7.41414569e-04,   6.61385824e-04,   5.15329524e-04, \
                         5.62962797e-04,   4.84417475e-04,   4.04049342e-04, \
                         3.45686074e-04,   3.62280810e-04,   3.20793598e-04, \
                         3.17432176e-04,   2.65239236e-04,   2.55608761e-04, \
                         2.11663102e-04,   2.23451940e-04,   2.14970373e-04, \
                         1.87199646e-04,   2.01367252e-04,   1.59298151e-04, \
                         1.73178962e-04,   1.48874838e-04,   1.36001604e-04, \
                         1.43551756e-04,   1.51749658e-04,   1.40357232e-04, \
                         1.08334369e-04,   9.82451511e-05,   1.14485038e-04, \
                         1.05696485e-04,   1.10897103e-04,   9.92508466e-05, \
                         8.17683437e-05,   9.00938135e-05,   7.55619120e-05, \
                         9.22618169e-05,   8.24362262e-05,   8.56524332e-05, \
                         7.17028719e-05,   6.98181765e-05,   7.32711509e-05, \
                         6.02283243e-05,   6.59735326e-05,   6.63745656e-05, \
                         5.65874521e-05,   4.89422753e-05])
                     
    # Function to extend wings
    # def add_wings(a, pts):
    #     w = min(a)*(np.arange(pts)/pts)*50
    #     a = np.concatenate([np.abs(np.random.normal(w,w)),a,np.abs(np.random.normal(w[::-1],w[::-1]))])
    #
    #     return a
        
    # Extend the wings for a nice wide PSF that tapers off rather than ending sharply for bright targets
    # if extend:
    #     lpsf = add_wings(lpsf.copy(), extend)
        
    # Scale the transmission to 1
    psf = psf1D/np.trapz(psf1D)
    
    # Interpolate lpsf to distance
    p0 = len(psf)//2
    val = np.interp(distance, range(len(psf[p0:])), psf[p0:])
    
    if plot:
        plt.plot(range(len(psf[p0:])), psf[p0:])
        plt.scatter(distance, val, c='r', zorder=5)
        
    return val

def lambda_lightcurve(wavelength, response, distance, pfd2adu, ld_coeffs, ld_profile, star, planet, time, params, filt, trace_radius=25, snr=100, floor=2, extend=25, plot=False):
    """
    Generate a lightcurve for a given wavelength
    
    Parameters
    ----------
    wavelength: float
        The wavelength value in microns
    response: float
        The spectral response of the detector at the given wavelength
    distance: float
        The Euclidean distance from the center of the cross-dispersed PSF
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
    
    Returns
    -------
    sequence
        A 1D array of the lightcurve with the same length as *t* 
    """
    nframes = len(time)
    
    # If it's a background pixel, it's just noise
    if distance>trace_radius+extend \
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
        
        # Convert from energy flux density to photon flux density [photons/s/cm2/A]
        # by multiplying by (lambda/h*c)
        flux0 *= wavelength*503411665111.4543 # [1/erg*um]
        
        # Convert from photon flux density to ADU/s by multiplying by the 
        # wavelength interval [um/pixel], primary mirror area [cm2], and gain [ADU/e-]
        flux0 *= pfd2adu
        
        # Expand to shape of time axis and add noise
        # flux0 = np.abs(flux0)
        # flux = np.abs(np.random.normal(loc=flux0, scale=flux0/snr, size=len(time)))
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
        flux *= psf_position(distance, extend=extend)
        
        # Replace very low signal pixels with noise floor
        # flux[flux<floor] += np.random.normal(loc=floor, scale=1, size=len(flux[flux<floor]))
        flux[flux<floor] += np.repeat(floor, len(flux[flux<floor]))
        
        # Plot
        if plot:
            plt.plot(t, flux)
            plt.xlabel("Time from central transit")
            plt.ylabel("Flux Density [photons/s/cm2/A]")
        
    return flux

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
                        extend      = 25, 
                        trace_radius= 50, 
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
        extend: int
            The number of pixels to extend the wings of the pfs
        trace_radius: int
            The radius of the trace
        target: str (optional)
            The name of the target
        """
        # Set instance attributes for the exposure
        self.subarray     = subarray
        self.nrows        = 256 if '256' in subarray else 96 if '96' in subarray else 2048
        self.ncols        = 2048
        self.ngrps        = ngrps
        self.nints        = nints
        self.nresets      = 1
        self.time         = get_frame_times(self.subarray, self.ngrps, self.nints, t0, self.nresets)
        self.nframes      = len(self.time)
        self.target       = target or 'Simulated Target'
        self.obs_date     = ''
        self.filter       = 'CLEAR'
        self.header       = ''
        
        # Set instance attributes for the target
        self.star         = star
        self.planet       = planet
        self.params       = params
        self.ld_coeffs    = ld_coeffs
        self.ld_profile   = ld_profile or 'quadratic'
        self.trace_radius = trace_radius
        self.snr          = snr
        self.extend       = extend
        self.wave         = wave_solutions(str(self.nrows))
        
        # Calculate a map for each order that converts photon flux density to ADU/s
        self.gain = 1.61 # [e-/ADU]
        self.primary_mirror = 253260 # [cm2]
        avg_wave = np.mean(self.wave, axis=1)
        self.pfd2adu = np.ones((3,self.ncols*self.nrows))
        for n,aw in enumerate(avg_wave):
            coeffs = np.polyfit(aw[:-1], np.diff(aw), 1)
            wave_int = (np.polyval(coeffs, self.wave[n])*q.um).to(q.AA)
            self.pfd2adu[n] = (wave_int*self.primary_mirror*q.cm**2/self.gain).value.flatten()
        
        # Add the orbital parameters as attributes
        for p in [i for i in dir(self.params) if not i.startswith('_')]:
            setattr(self, p, getattr(self.params, p))
        
        # Create the empty exposure
        self.tso = np.zeros((self.nframes, self.nrows, self.ncols))
        self.tso_order1 = np.zeros((self.nframes, self.nrows, self.ncols))
        self.tso_order2 = np.zeros((self.nframes, self.nrows, self.ncols))
    
    def run_simulation(self, orders=[1,2], filt='CLEAR'):
        """
        Generate the simulated 2D data given the initialized TSO object
        
        Parameters
        ----------
        orders: sequence
            The orders to simulate
        filt: str
            The element from the filter wheel to use, i.e. 'CLEAR' or 'F277W'
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
            
        # Make dummy array of LDCs if no planet (required for multiprocessing)
        if isinstance(self.planet, str):
            self.ld_coeffs = np.zeros((2, self.nrows*self.ncols, 2))
            
        # Generate simulation for each order
        for order in orders:
            
            # Get the wavelength map
            local_wave = self.wave[order-1].flatten()
            
            # Get the distance map 
            local_distance = distance_map(order=order).flatten()
            
            # Get limb darkening map
            local_ld_coeffs = self.ld_coeffs.copy()[order-1]
            
            # Get relative spectral response map
            throughput = np.genfromtxt(DIR_PATH+'/files/gr700xd_{}_order{}.dat'.format(self.filter,order), unpack=True)
            local_response = np.interp(local_wave, throughput[0], throughput[-1], left=0, right=0)
            
            # Get the wavelength interval per pixel map
            local_pfd2adu = self.pfd2adu[order-1]
            
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
                           trace_radius  = self.trace_radius,
                           snr           = self.snr,
                           extend        = self.extend)
                    
            # Generate the lightcurves at each pixel
            lightcurves = pool.starmap(func, zip(local_wave, local_response, local_distance, local_pfd2adu, local_ld_coeffs))
            
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
        # self.tso += dark_ramps(self.time, self.subarray)
    
    def add_noise_model(self):
        """
        Generate the noise model and add to the simulation
        """
        pass
    
    def plot_frame(self, frame='', scale='linear', order='', cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        scale: str
            Plot in linear or log scale
        order: int (optional)
            The order to isolate
        cmap: str
            The color map to use
        """
        if order:
            tso = getattr(self, 'tso_order{}'.format(order))
        else:
            tso = self.tso
        
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
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        """
        snr  = np.sqrt(self.tso[frame or self.nframes//2].data)
        vmax = int(np.nanmax(snr))
        
        plt.figure(figsize=(13,2))
        plt.imshow(snr, origin='lower', interpolation='none', vmin=1, vmax=vmax, cmap=cmap)
        
        plt.colorbar()
        plt.title('SNR over Spectrum')
        
    def plot_saturation(self, frame='', saturation = 80.0, cmap=cm.jet):
        """
        Plot a frame of the TSO
        
        Parameters
        ----------
        frame: int
            The frame number to plot
        
        fullWell: percentage [0-100] of maximum value, 65536
        """
        
        fullWell    = 65536.0
        
        saturated = np.array(self.tso[frame or self.nframes//2].data) > (saturation/100.0) * fullWell
        
        plt.figure(figsize=(13,2))
        plt.imshow(saturated, origin='lower', interpolation='none', cmap=cmap)
        
        plt.colorbar()
        plt.title('Saturated Pixels')
    
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
        
    def plot_lightcurve(self, col):
        """
        Plot a lightcurve for each column index given
        
        Parameters
        ----------
        col: int, sequence
            The column index(es) to plot a light curve for
        """
        if isinstance(col, int):
            col = [col]
        
        for c in col:
            # ld = self.ldc[c*self.tso.shape[1]]
            w = np.mean(self.wave[0], axis=0)[c]
            f = np.nansum(self.tso[:,:,c], axis=1)
            f *= 1./np.nanmax(f)
            plt.plot(self.time/3000., f, label='Col {}'.format(c), marker='.', ls='None')
            
        # Plot whitelight curve too
        # plt.plot(self.time)
            
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
        
        # Get extracted spectrum
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
        plt.figure(figsize=(13,2))
        plt.plot(wave, flux, label='Extracted')
        plt.plot(*self.star, label='Injected')
    
    def save_tso(self, filename='dummy.save'):
        """
        Save the TSO data to file
        
        Parameters
        ----------
        filename: str
            The path of the save file
        """
        print('Saving TSO class dict to {}'.format(filename))
        joblib.dump(self.__dict__, filename)
    
    def load_tso(self, filename):
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
                ('SUBARRAY', 'SUBSTRIP256', 'Subarray pattern name'),
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

