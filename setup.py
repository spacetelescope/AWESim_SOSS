#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Joe Filippazzo",
    author_email='jfilippazzo@stsci.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
    description="Advanced Webb Exposure SIMulator for SOSS",
    entry_points={
        'console_scripts': [
            'awesimsoss=awesimsoss.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    # long_description=readme + '\n\n' + history,
    long_description="The Advanced Webb Exposure SIMulator for SOSS (awesimsoss) produces simulated time-series observations for the Single Object Slitless Spectroscopy (SOSS) mode of the NIRISS instrument onboard the James Webb Space Telescope.",
    include_package_data=True,
    keywords='awesimsoss',
    name='awesimsoss',
    packages=find_packages(include=['awesimsoss']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/hover2pi/awesimsoss',
    version='0.3.6',
    zip_safe=False,
)
