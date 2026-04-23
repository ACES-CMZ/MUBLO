"""
"matched filter" based on SO32
"""
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

cube = SpectralCube.read('spw31.cube.I.pbcor.mublo.SO32.fits')
reg = regions.Regions.read('mublo_cutout_circle_tight.reg')[0]
scube = cube.subcube_from_regions([reg])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    std = scube.std()
mask1 = binary_erosion((scube > 0.25*std).include(), iterations=1)
mask2 = (scube > 3.*std).include()
mask3 = binary_dilation(input=mask2, mask=mask1)

# Remove small islands in mask3 before making mask4.
# "Small" = islands with size < 75th percentile, but
# only apply this rule when there are more than 10 islands.
labeled, n_islands = ndi.label(mask3)
if n_islands > 10:
    # compute sizes of each island (exclude background bin 0)
    sizes = np.bincount(labeled.ravel())
    if sizes.size > 1:
        island_sizes = sizes[1:]
        labels = np.arange(1, len(island_sizes) + 1)
        threshold = np.percentile(island_sizes, 50)
        print("Threshold size for island removal:", threshold)
        # labels to remove: strictly smaller than threshold
        small_labels = labels[island_sizes < threshold]
        if small_labels.size > 0:
            remove_mask = np.isin(labeled, small_labels)
            #mask3 = mask3.copy()
            mask3[remove_mask] = False

mask4 = binary_dilation(input=mask3, iterations=3)

mcube = scube.with_mask(mask4)
mcube.write("SO32_matched_filter_cube.fits", overwrite=True)


def matched_filter(scube, mcube):
    """
    Apply matched filter to produce a spectrum.

    The spectrum is the weighted sum using mcube as the weight,
    centered at each spectral location. For each output channel i,
    we extract a spectral window from scube centered at i with the
    same size as mcube, multiply element-wise by mcube, and sum.

    This follows spectral-cube best practices by:
    - Using filled_data to respect masks with fill_value=0
    - Minimizing data copying and memory usage
    - Using vectorized operations for maximum efficiency

    Parameters
    ----------
    scube : SpectralCube
        The input spectral cube to filter (can have any number of spectral channels)
    mcube : SpectralCube
        The matched filter cube (defines the 3D weight pattern).
        Should have the same spatial dimensions as scube.

    Returns
    -------
    spectrum : OneDSpectrum
        The filtered spectrum with the same spectral axis as scube

    Notes
    -----
    Edge channels where the mcube filter extends beyond scube are handled
    by only using the valid overlapping portion of the filter.
    """
    from spectral_cube import OneDSpectrum

    # Get the filled data arrays (respecting masks)
    # Using fill_value=0 so masked regions don't contribute
    sdata = scube.filled_data[:].value  # shape: (nspec_s, ny, nx)
    mdata = mcube.filled_data[:].value  # shape: (nspec_m, ny, nx)

    # Get the filled data arrays (respecting masks)
    # Using fill_value=0 so masked regions don't contribute
    sdata = scube.filled_data[:].value  # shape: (nspec_s, ny, nx)
    mdata = mcube.filled_data[:].value  # shape: (nspec_m, ny, nx)

    # Replace any remaining NaNs with 0
    sdata = np.where(np.isfinite(sdata), sdata, 0)
    mdata = np.where(np.isfinite(mdata), mdata, 0)

    nspec_s = sdata.shape[0]  # Number of channels in scube
    nspec_m = mdata.shape[0]  # Number of channels in mcube (filter length)

    # The filter will be centered at each channel
    half_filter = nspec_m // 2

    # Flatten spatial dimensions for efficiency
    sdata_flat = sdata.reshape(nspec_s, -1)  # (nspec_s, npix)
    mdata_flat = mdata.reshape(nspec_m, -1)  # (nspec_m, npix)

    # Initialize output spectrum
    spectrum = np.zeros(nspec_s)

    # For each output channel, extract the corresponding window and compute dot product
    # This is more efficient than nested loops
    for i in tqdm(range(nspec_s)):
        # Determine the range of channels in scube to use for this output channel
        start_chan = i - half_filter
        end_chan = start_chan + nspec_m

        # Determine the valid overlap between the filter and the data
        # Handle edge cases where filter extends beyond data
        filter_start = max(0, -start_chan)  # Where to start in the filter
        filter_end = nspec_m - max(0, end_chan - nspec_s)  # Where to end in the filter

        data_start = max(0, start_chan)  # Where to start in the data
        data_end = min(nspec_s, end_chan)  # Where to end in the data

        # Extract the valid portions and compute the weighted sum
        if filter_end > filter_start and data_end > data_start:
            data_window = sdata_flat[data_start:data_end, :]  # (n_valid, npix)
            filter_window = mdata_flat[filter_start:filter_end, :]  # (n_valid, npix)

            # Sum over both spectral and spatial dimensions
            spectrum[i] = np.sum(data_window * filter_window) / np.sum(filter_window)

    # Create OneDSpectrum with proper units and spectral axis
    # OneDSpectrum accepts wcs but not spectral_axis directly
    spectrum_1d = OneDSpectrum(spectrum * scube.unit,
                               wcs=scube.wcs.spectral)

    return spectrum_1d


# Apply matched filter to other cubes
for fn in glob.glob("*cube.I.pbcor.10kms.fits"):
    cube = SpectralCube.read(fn)
    scube = cube.subcube_from_regions([reg])

    # scube and mcube should have the same spatial shape
    assert scube.shape[1:] == mcube.shape[1:]

    meanspec = scube.mean(axis=(1, 2))
    meanspec.write(fn.replace('.fits', '.mean_spectrum.fits'), overwrite=True)

    spectrum = matched_filter(scube, mcube)

    # Save the spectrum
    outname = fn.replace('.fits', '.matchfilter_spectrum.fits')
    spectrum.write(outname, overwrite=True)
    print(f"Saved matched filter spectrum to {outname}")


    sp = pyspeckit.Spectrum(outname)
    sp.plotter()
    sp.plotter.savefig(outname.replace('.fits', '.png'))
    #sp.plotter.axis.plot(meanspec.spectral_axis.to(u.Hz), meanspec.value, label='Mean Spectrum', linewidth=0.5, alpha=0.7, zorder=-5)
