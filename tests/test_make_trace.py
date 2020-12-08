#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `make_trace` module."""

import unittest

import numpy as np

from awesimsoss import make_trace as mt


def test_nuke_psfs():
    """Test for nuke_psfs function. This invokes calculate_psf_tilts, generate_SOSS_psfs, and SOSS_psf_cube."""
    # Without multiprocessing
    mt.nuke_psfs(mprocessing=False)

    # With multiprocessing
    mt.nuke_psfs()


def test_generate_SOSS_ldcs():
    """A test of the generate_SOSS_ldcs function"""
    lookup = mt.generate_SOSS_ldcs(np.linspace(1., 2., 3), 'quadratic', [3300, 4.5, 0])

    # Make sure three wavelengths are returned
    assert len(lookup) == 3

    # Make sure 2 coefficients are returned (for quadratic profile)
    assert len(lookup[0]) == 2


def test_get_angle():
    """Test for the get_angle function"""
    coords = 2, 2
    angle = mt.get_angle(coords)

    assert isinstance(angle, float)


def test_get_SOSS_psf():
    """Test for the get_SOSS_psf function"""
    psf = mt.get_SOSS_psf(1)

    assert psf.ndim == 2


def test_psf_tilts():
    """Test for the psf_tilts function"""
    tilts = mt.psf_tilts(1)

    assert len(tilts) == 2048