from spectral_cube import SpectralCube, DaskSpectralCube
import warnings
import os
from astropy.convolution import Gaussian1DKernel
from astropy import units as u, constants, visualization
import glob
from scipy.ndimage import binary_dilation, binary_erosion
from scipy import ndimage as ndi
import numpy as np
    
import regions

# Configure dask to use 8 processors for parallel processing
import dask
dask.config.set(scheduler='threads', num_workers=8)

lines = {'SO2211': 86.09395, 'H13CN10': 86.33992, 'H13CO+10': 86.7543, 'SiO21': 86.84696, 'HNCO4-3':87.925238, 'CS21': 97.98095, 'SO2735-826': 97.70234, 'OCS8-7': 97.3012085, 'HCN1-0': 88.6316023, 'SO32': 99.29987, 'H40a': 99.02295, 'CS76': 342.88285, 'SO8877': 344.31061, '13CO32': 330.58796, '12CO32': 345.796}

reg = regions.Regions.read('mublo_cutout_square.reg')[0]
for fn in glob.glob('b9/*cube.I.selfcal.pbcor.fits') + (glob.glob("*cube.I.pbcor.fits") + glob.glob("*cube.I.selfcal.pbcor.fits"))[::-1]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = SpectralCube.read(fn, use_dask=True)
    scube = cube.subcube_from_regions([reg])

    outfile = fn.replace(".fits", f".10kms.fits")
    if not os.path.exists(outfile):
        print(f'{fn}')
        # assume intrinsic width is 1.2 channels
        pixwidth = np.abs(cube.with_spectral_unit(u.km/u.s, velocity_convention='radio').spectral_axis.diff()[0])
        kernelwidth = (((10*u.km/u.s / 2.35)**2 - (1.2 * pixwidth )**2)**0.5 / pixwidth).value
        downsample_factor = int(np.floor(kernelwidth))
        print(f"Smoothing to 10 km/s with kernel width {kernelwidth} from pixel width {pixwidth} and downsample_factor={downsample_factor}", flush=True)
        smcube = scube.spectral_smooth(Gaussian1DKernel(kernelwidth), verbose=1, num_cores=8)[::downsample_factor]
        smcube.write(outfile)
    else:
        smcube = SpectralCube.read(outfile)
    print(fn, cube, scube, smcube)

    for linename, freq in lines.items():
        freq = freq*u.GHz if freq < 900 else freq*u.MHz
        vcube = scube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=freq)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vscube = vcube.spectral_slab(-300*u.km/u.s, 300*u.km/u.s)
        if vscube.shape[0] > 1:
            outfile = fn.replace(".fits", f".mublo.{linename}.fits")
            if not os.path.exists(outfile):
                print(f'{fn}: {linename} - creating smoothed cube {outfile}')
                vsmcube = vscube.spectral_smooth(Gaussian1DKernel(3), verbose=1, num_cores=8)[::3]
                vsmcube.write(outfile)
            else:
                vsmcube = SpectralCube.read(outfile, use_dask=True)

            vsmcube = vsmcube.spectral_slab(-120*u.km/u.s, 220*u.km/u.s)

            outfile = fn.replace(".fits", f".mublo.{linename}.mom0.fits")
            if True: #not os.path.exists(outfile):
                print(f'{fn}: {linename} mom0')
                mom0 = vscube.moment0()
                mom0.write(outfile, overwrite=True)

            outfile = fn.replace(".fits", f".mublo.{linename}.mom1.fits")
            if True: #not os.path.exists(outfile):
                print(f'{fn}: {linename} mom1')
                mom1 = vscube.moment1()
                mom1.write(outfile, overwrite=True)


            outfile = fn.replace(".fits", f".mublo.{linename}.masked.mom0.fits")
            if True: #not os.path.exists(outfile):
                print(f'{fn}: {linename} mom0 and mom1 masked [cube.shape={vsmcube.shape}]', end='')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    std = vsmcube.std()
                mask1 = binary_erosion((vsmcube > 0.25*std).include(), iterations=1)
                mask2 = (vsmcube > 3.*std).include()
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
                        threshold = np.percentile(island_sizes, 70)
                        print("Threshold size for island removal:", threshold)
                        # labels to remove: strictly smaller than threshold
                        small_labels = labels[island_sizes < threshold]
                        if small_labels.size > 0:
                            remove_mask = np.isin(labeled, small_labels)
                            #mask3 = mask3.copy()
                            mask3[remove_mask] = False

                mask4 = binary_dilation(input=mask3, iterations=3)

                print(f" Included in mask3: {mask3.sum()}, in mask4: {mask4.sum()}")
                mom0 = vsmcube.with_mask(mask4).moment0()
                mom0.write(outfile, overwrite=True)

                mom1 = vsmcube.with_mask(mask4).moment1()
                mom1.write(outfile.replace(".mom0.fits", ".mom1.fits"), overwrite=True)
