#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `awesim` module."""

from copy import copy
import unittest

import numpy as np
import astropy.units as q
import astropy.constants as ac
import batman
from hotsoss import STAR_DATA, PLANET_DATA

from awesimsoss import TSO, BlackbodyTSO, ModelTSO, TestTSO


class test_ModelTSO(unittest.TestCase):
    """A test of the ModelTSO class"""
    def setUp(self):
        pass

    def test_run_no_planet(self):
        """A test of the ModelTSO class with no planet"""
        tso = ModelTSO(stellar_model='ATLAS')
        tso = ModelTSO(stellar_model='PHOENIX')

    def test_run_with_planet(self):
        """A test of the ModelTSO class with a planet"""
        tso = ModelTSO(add_planet=True)


class test_BlackbodyTSO(unittest.TestCase):
    """A test of the BlackbodyTSO class"""
    def setUp(self):
        pass

    def test_run_no_planet(self):
        """A test of the BlackbodyTSO class with no planet"""
        tso = BlackbodyTSO()

    def test_run_with_planet(self):
        """A test of the BlackbodyTSO class with a planet"""
        tso = BlackbodyTSO(add_planet=True)


class test_TestTSO(unittest.TestCase):
    """A test of the TestTSO class"""
    def setUp(self):
        pass

    def test_run_no_planet(self):
        """A test of the TestTSO class with no planet"""
        tso = TestTSO()

    def test_run_with_planet(self):
        """A test of the TestTSO class with a planet"""
        tso = TestTSO(add_planet=True)


class test_TSO_validation(unittest.TestCase):
    """Validation tests for the TSO class"""
    def setUp(self):
        """Setup for the tests"""
        # Get data
        self.star = STAR_DATA
        self.planet = PLANET_DATA

        # Make the simulation
        self.tso256 = TSO(ngrps=2, nints=2, star=self.star)
        self.tso256.simulate()

    def test_extract_order1_ideal(self):
        """Test that the column extracted spectrum from the order1 ideal trace matches the input spectrum"""
        # Get the order 1 data
        trace = self.tso256.tso_order1_ideal
        shape = trace.shape

        # Reshape into 3D and NaN reference pixels
        data = trace.reshape((shape[0] * shape[1], shape[2], shape[3]))
        data[:, :, :4] = np.nan
        data[:, :, -4:] = np.nan
        data[:, -4:, :] = np.nan

        # Add up counts
        order1_counts = np.nansum(data, axis=1)

        # Interpolate input spectrum
        input_spec = np.interp(self.tso256.avg_wave[0], self.star[0].value, self.star[1].value, right=np.nan, left=np.nan)

        for n, group in enumerate(order1_counts):

            # Convert counts to flux
            order1_flux = order1_counts[n] / self.tso256.order1_response / (self.tso256.frame_time * (n + 1))

            # Mean residual (away from edges)
            mean_residual = np.nanmean(((order1_flux.value - input_spec) / input_spec)[20:-20])

            # Check that the mean residual is less than 0.01
            self.assertTrue(mean_residual < 0.01)

    def test_extract_order2_ideal(self):
        """Test that the column extracted spectrum from the order2 ideal trace matches the input spectrum"""
        # Get the order 1 data
        trace = self.tso256.tso_order2_ideal
        shape = trace.shape

        # Reshape into 3D and NaN reference pixels
        data = trace.reshape((shape[0] * shape[1], shape[2], shape[3]))
        data[:, :, :4] = np.nan
        data[:, :, -4:] = np.nan
        data[:, -4:, :] = np.nan

        # Add up counts
        order2_counts = np.nansum(data, axis=1)

        # Interpolate input spectrum
        input_spec = np.interp(self.tso256.avg_wave[1], self.star[0].value, self.star[1].value, right=np.nan, left=np.nan)

        for n, group in enumerate(order2_counts):

            # Convert counts to flux
            order2_flux = order2_counts[n] / self.tso256.order2_response / (self.tso256.frame_time * (n + 1))

            # Mean residual (away from edges)
            mean_residual = np.nanmean(((order2_flux.value - input_spec) / input_spec)[20:-20])

            # Check that the mean residual is less than 0.01
            self.assertTrue(mean_residual < 0.01)


class test_TSO_verification(unittest.TestCase):
    """Verification tests for the TSO class"""
    def setUp(self):
        """Setup for the tests"""
        # Get data
        self.star = STAR_DATA
        self.planet = PLANET_DATA

    def test_add_lines(self):
        """Test the add_lines method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        amp = 1e-14*q.erg/q.s/q.cm**2/q.AA
        x_0 = 1.*q.um
        fwhm = 0.01*q.um

        # Add a bad profile name
        kwargs = {'amplitude': 1e-14*q.erg/q.s/q.cm**2/q.AA, 'x_0': 1.*q.um, 'fwhm': 0.01*q.um, 'profile': 'foo'}
        self.assertRaises(ValueError, tso.add_line, **kwargs)

        # Add a Lorentzian line
        kwargs = {'amplitude': amp, 'x_0': x_0, 'fwhm': fwhm, 'profile': 'lorentz'}
        tso.add_line(**kwargs)

        # Add a Gaussian line
        kwargs = {'amplitude': amp, 'x_0': x_0, 'fwhm': fwhm, 'profile': 'gaussian'}
        tso.add_line(**kwargs)

        # Add a Voigt line
        kwargs = {'amplitude': amp, 'x_0': x_0, 'fwhm': (fwhm, fwhm), 'profile': 'voigt'}
        tso.add_line(**kwargs)

        # Add a bad Voigt fwhm
        kwargs = {'amplitude': amp, 'x_0': x_0, 'fwhm': fwhm, 'profile': 'voigt'}
        self.assertRaises(TypeError, tso.add_line, **kwargs)

        # Check that the 3 good lines have been added
        self.assertEqual(len(tso.lines), 3)

    def test_export(self):
        """Test the export method"""
        # Make the TSO object and save
        test_tso = TestTSO(add_planet=True)

        # Good filename
        test_tso.export('outfile_uncal.fits')

        # Bad filename (no '_uncal')
        self.assertRaises(ValueError, test_tso.export, 'outfile.fits')

    def test_init(self):
        """Test that the TSO class is generated properly"""
        # Initialize the FULL frame with two groups and two integrations
        # and the CLEAR filter
        tso2048clear = TSO(ngrps=2, nints=2, star=self.star, subarray='FULL')

        self.assertEqual(tso2048clear.ngrps, 2)
        self.assertEqual(tso2048clear.nints, 2)
        self.assertEqual(tso2048clear.total_groups, 4)
        self.assertEqual(tso2048clear.dims, (2, 2, 2048, 2048))
        self.assertEqual(tso2048clear.subarray, 'FULL')
        self.assertEqual(tso2048clear.filter, 'CLEAR')

        # Initialize the 256 subarray with two groups and two integrations
        # and the CLEAR filter
        tso256clear = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP256')

        self.assertEqual(tso256clear.ngrps, 2)
        self.assertEqual(tso256clear.nints, 2)
        self.assertEqual(tso256clear.total_groups, 4)
        self.assertEqual(tso256clear.dims, (2, 2, 256, 2048))
        self.assertEqual(tso256clear.subarray, 'SUBSTRIP256')
        self.assertEqual(tso256clear.filter, 'CLEAR')

        # Initialize the 96 subarray with two groups and two integrations
        # and the CLEAR filter
        tso96clear = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP96')

        self.assertEqual(tso96clear.ngrps, 2)
        self.assertEqual(tso96clear.nints, 2)
        self.assertEqual(tso96clear.total_groups, 4)
        self.assertEqual(tso96clear.dims, (2, 2, 96, 2048))
        self.assertEqual(tso96clear.subarray, 'SUBSTRIP96')
        self.assertEqual(tso96clear.filter, 'CLEAR')

        # Initialize the FULL frame with two groups and two integrations
        # and the F277W filter
        tso2048f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='FULL', filter='F277W')

        self.assertEqual(tso2048f277w.ngrps, 2)
        self.assertEqual(tso2048f277w.nints, 2)
        self.assertEqual(tso2048f277w.total_groups, 4)
        self.assertEqual(tso2048f277w.dims, (2, 2, 2048, 2048))
        self.assertEqual(tso2048f277w.subarray, 'FULL')
        self.assertEqual(tso2048f277w.filter, 'F277W')

        # Initialize the 256 subarray with two groups and two integrations
        # and the F277W filter
        tso256f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP256', filter='F277W')

        self.assertEqual(tso256f277w.ngrps, 2)
        self.assertEqual(tso256f277w.nints, 2)
        self.assertEqual(tso256f277w.total_groups, 4)
        self.assertEqual(tso256f277w.dims, (2, 2, 256, 2048))
        self.assertEqual(tso256f277w.subarray, 'SUBSTRIP256')
        self.assertEqual(tso256f277w.filter, 'F277W')

        # Initialize the 96 subarray with two groups and two integrations
        # and the F277W filter
        tso96f277w = TSO(ngrps=2, nints=2, star=self.star, subarray='SUBSTRIP96', filter='F277W')

        self.assertEqual(tso96f277w.ngrps, 2)
        self.assertEqual(tso96f277w.nints, 2)
        self.assertEqual(tso96f277w.total_groups, 4)
        self.assertEqual(tso96f277w.dims, (2, 2, 96, 2048))
        self.assertEqual(tso96f277w.subarray, 'SUBSTRIP96')
        self.assertEqual(tso96f277w.filter, 'F277W')

    def test_run_no_planet(self):
        """A test of simulate() with no planet"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()
        tso.subarray = 'SUBSTRIP96'
        tso.simulate()
        tso.subarray = 'FULL'
        tso.simulate()

    def test_run_with_planet(self):
        """A test of simulate() with a planet"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Make orbital params
        params = batman.TransitParams()
        params.t0 = 0.
        params.per = 5.7214742
        params.a = 0.0558*q.AU.to(ac.R_sun)*0.66
        params.inc = 89.8
        params.ecc = 0.
        params.w = 90.
        params.limb_dark = 'quadratic'
        params.u = [0.1, 0.1]
        params.rp = 0.
        tmodel = batman.TransitModel(params, tso.time.jd)
        tmodel.teff = 3500
        tmodel.logg = 5
        tmodel.feh = 0

        # Run the simulation
        tso.simulate(planet=self.planet, tmodel=tmodel)
        tso.subarray = 'SUBSTRIP96'
        tso.simulate(planet=self.planet, tmodel=tmodel)
        tso.subarray = 'FULL'
        tso.simulate(planet=self.planet, tmodel=tmodel)

    def test_lookup(self):
        """Test that coordinates are looked up if given a name"""
        # Make the TSO object
        targ = TSO(ngrps=2, nints=2, star=self.star, target='trappist-1')
        no_targ = TSO(ngrps=2, nints=2, star=self.star)

        # Check target name
        self.assertNotEqual(targ.target, no_targ.target)

        # Check coordinates
        self.assertNotEqual(targ.ra, no_targ.ra)
        self.assertNotEqual(targ.dec, no_targ.dec)

    def test_star(self):
        """Test that errors are thrown for bas star input"""
        # Test that non wavelength units fail
        bad_wave_star = copy(self.star)
        bad_wave_star[0] *= q.Jy
        kwargs = {'nints': 2, 'ngrps': 2, 'star': bad_wave_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that non flux density units fail
        bad_flux_star = copy(self.star)
        bad_flux_star[1] *= q.K
        kwargs = {'nints': 2, 'ngrps': 2, 'star': bad_flux_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that no units fail
        bad_unit_star = copy(self.star)
        bad_unit_star[0] = bad_unit_star[0].value
        kwargs = {'nints': 2, 'ngrps': 2, 'star': bad_unit_star}
        self.assertRaises(ValueError, TSO, **kwargs)

        # Test that spectrum shape
        bad_size_star = [self.star[0]]
        kwargs = {'nints': 2, 'ngrps': 2, 'star': bad_size_star}
        self.assertRaises(ValueError, TSO, **kwargs)

    def test_bad_attrs(self):
        """Test that invalid attributes throw an error"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Bad filter
        self.assertRaises(ValueError, setattr, tso, 'filter', 'foo')

        # Bad ncols
        self.assertRaises(TypeError, setattr, tso, 'ncols', 3)

        # Bad nrows
        self.assertRaises(TypeError, setattr, tso, 'nrows', 3)

        # Bad nints
        self.assertRaises(TypeError, setattr, tso, 'nints', 'three')

        # Bad ngrps
        self.assertRaises(TypeError, setattr, tso, 'ngrps', 'three')

        # Bad nresets
        self.assertRaises(TypeError, setattr, tso, 'nresets', 'three')

        # Bad orders
        tso.orders = 1
        self.assertRaises(ValueError, setattr, tso, 'orders', 'three')

        # Bad subarray
        self.assertRaises(ValueError, setattr, tso, 'subarray', 'three')

        # Bad t0
        self.assertRaises(ValueError, setattr, tso, 'obs_date', 123)

        # Bad target
        self.assertRaises(TypeError, setattr, tso, 'target', 3)

    def test_ldcs(self):
        """Test the limb darkening coefficients"""
        # Create instance
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Set manually
        ldcs = tso.ld_coeffs
        tso.ld_coeffs = np.ones((3, 2048, 2))

        # Bad LDCs (Removed TypeError in favor of print statement)
        # self.assertRaises(TypeError, setattr, tso, 'ld_coeffs', 'foo')

    def test_plot(self):
        """Test plot method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)

        # Test plot with no data
        plt = tso.plot(draw=False)

        # Run simulation
        tso.simulate()

        # Standard plot with traces
        plt = tso.plot(traces=True)

        # Standard plot with one order
        plt = tso.plot(order=1, draw=False)

        # No noise plot
        plt = tso.plot(noise=False, draw=False)

        # Log plot
        plt = tso.plot(scale='log', draw=False)

    def test_plot_ramp(self):
        """Test plot_ramp method"""
        # Make the TSO object
        tso = TSO(ngrps=2, nints=2, star=self.star)
        tso.simulate()

        # Standard plot
        plt = tso.plot_ramp(draw=False)
        tso.plot_ramp()
