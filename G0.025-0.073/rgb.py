from spectral_cube import SpectralCube
import warnings
import os
from astropy.convolution import Gaussian1DKernel
from astropy import units as u, constants, visualization
import glob
from scipy.ndimage import binary_dilation, binary_erosion
from scipy import ndimage as ndi
import numpy as np
from tqdm.auto import tqdm

import regions

import pylab as pl
import pyspeckit

cube = SpectralCube.read("SO32_matched_filter_cube.fits")

m0blue = cube.spectral_slab(-100*u.km/u.s, 20*u.km/u.s).moment0()
m0green = cube.spectral_slab(20*u.km/u.s, 80*u.km/u.s).moment0()
m0red = cube.spectral_slab(80*u.km/u.s, 200*u.km/u.s).moment0()

m0blue.write("SO32_matched_filter_m0blue.fits", overwrite=True)
m0green.write("SO32_matched_filter_m0green.fits", overwrite=True)
m0red.write("SO32_matched_filter_m0red.fits", overwrite=True)

