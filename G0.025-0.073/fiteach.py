"""
Using https://radio-astro-tools.github.io/tutorials/fitting_with_spectralcube.html
fit each spectrum with a Gaussian.  Save the resulting peak, width, and centroid images as FITS files.
"""
from spectral_cube import SpectralCube
import warnings
import os
from astropy.convolution import Gaussian1DKernel
from astropy import units as u, constants, visualization
from astropy.modeling import models, fitting
from astropy.io import fits
import glob
from scipy.ndimage import binary_dilation, binary_erosion
from scipy import ndimage as ndi
import numpy as np
from tqdm.auto import tqdm

import regions

import pylab as pl
import pyspeckit

# Define the lines to fit
lines = {
    'CS21': 'spw29.cube.I.pbcor.mublo.CS21.fits',
    'SO2211': 'spw25.cube.I.pbcor.mublo.SO2211.fits',
    'SO32': 'spw31.cube.I.pbcor.mublo.SO32.fits'
}

# Read the region file once
reg = regions.Regions.read('mublo_cutout_circle_tight.reg')[0]

# Initialize the fitter
fit_g = fitting.LevMarLSQFitter()

# Loop through each line
for line_name, cube_file in lines.items():
    print(f"\n{'='*60}")
    print(f"Processing {line_name}")
    print(f"{'='*60}")
    
    # Check if file exists
    if not os.path.exists(cube_file):
        print(f"Warning: {cube_file} not found, skipping {line_name}")
        continue
    
    # Read in the cube
    cube = SpectralCube.read(cube_file)
    scube = cube.subcube_from_regions([reg])
    
    vcube = scube.spectral_slab(-150*u.km/u.s, 250*u.km/u.s)
    
    # Convert to km/s for convenience
    vcube = vcube.with_spectral_unit(u.km / u.s)
    
    # Prepare output arrays for peak (amplitude), centroid (mean), and width (stddev)
    ny, nx = vcube.shape[1], vcube.shape[2]
    amplitude_map = np.zeros((ny, nx)) * vcube.unit
    centroid_map = np.zeros((ny, nx)) * u.km / u.s
    width_map = np.zeros((ny, nx)) * u.km / u.s
    
    print(f"Fitting {ny}x{nx} spectra for {line_name}...")
    
    # Loop through all spatial pixels and fit each spectrum
    for y in tqdm(range(ny), desc=f"{line_name}"):
        for x in range(nx):
            # Extract the spectrum at this position
            spec = vcube[:, y, x]
            
            # Skip if all NaN or all zeros
            if np.all(np.isnan(spec)) or np.all(spec.value == 0):
                amplitude_map[y, x] = np.nan * vcube.unit
                centroid_map[y, x] = np.nan * u.km / u.s
                width_map[y, x] = np.nan * u.km / u.s
                continue
            
            # Estimate initial parameters from the spectrum
            # Peak amplitude
            max_idx = np.nanargmax(spec.value)
            amp_guess = spec.value[max_idx] * vcube.unit
            
            # Centroid guess at the peak location
            mean_guess = spec.spectral_axis[max_idx]
            
            # Width guess (reasonable default for these data)
            stddev_guess = 20.0 * u.km / u.s
            
            # Define initial Gaussian model
            g_init = models.Gaussian1D(
                amplitude=amp_guess,
                mean=mean_guess,
                stddev=stddev_guess
            )
            
            # Fit the model, catching only fitting convergence errors
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    g_fit = fit_g(g_init, spec.spectral_axis, spec)
                
                # Store the fitted parameters
                amplitude_map[y, x] = g_fit.amplitude
                centroid_map[y, x] = g_fit.mean
                width_map[y, x] = g_fit.stddev
                
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                # If fit fails due to convergence issues or invalid values, store NaN
                # but let other exceptions (like AttributeError, KeyError, etc.) propagate
                print(f"Warning: Fit failed at ({y}, {x}): {e}")
                amplitude_map[y, x] = np.nan * vcube.unit
                centroid_map[y, x] = np.nan * u.km / u.s
                width_map[y, x] = np.nan * u.km / u.s
    
    print(f"Fitting complete for {line_name}. Saving results...")
    
    # Save the results as FITS files
    # We'll use the spatial WCS from the original cube
    spatial_wcs = vcube.wcs.celestial
    
    # Create FITS HDUs with proper WCS
    amplitude_hdu = fits.PrimaryHDU(data=amplitude_map.value, header=spatial_wcs.to_header())
    amplitude_hdu.header['BUNIT'] = str(amplitude_map.unit)
    amplitude_hdu.header['COMMENT'] = 'Gaussian fit amplitude (peak intensity)'
    amplitude_hdu.writeto(f'{line_name}_amplitude.fits', overwrite=True)
    
    centroid_hdu = fits.PrimaryHDU(data=centroid_map.value, header=spatial_wcs.to_header())
    centroid_hdu.header['BUNIT'] = str(centroid_map.unit)
    centroid_hdu.header['COMMENT'] = 'Gaussian fit centroid (mean velocity)'
    centroid_hdu.writeto(f'{line_name}_centroid.fits', overwrite=True)
    
    width_hdu = fits.PrimaryHDU(data=width_map.value, header=spatial_wcs.to_header())
    width_hdu.header['BUNIT'] = str(width_map.unit)
    width_hdu.header['COMMENT'] = 'Gaussian fit width (standard deviation)'
    width_hdu.writeto(f'{line_name}_width.fits', overwrite=True)
    
    print(f"Saved for {line_name}:")
    print(f"  - {line_name}_amplitude.fits (peak intensity)")
    print(f"  - {line_name}_centroid.fits (mean velocity)")
    print(f"  - {line_name}_width.fits (standard deviation)")

print("\n" + "="*60)
print("All lines processed!")
print("="*60)