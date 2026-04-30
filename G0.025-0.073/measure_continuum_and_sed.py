"""
Measure continuum flux from B3, B7, and B9 ALMA bands and overplot on SED.
Borrows SED plotting code from MUBLO_MultiwavelengthCutouts.ipynb.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import radio_beam

# Source coordinates (MUBLO)
coord = SkyCoord(ra=266.4905821*u.deg, dec=-28.9529931*u.deg, frame='icrs')

# Continuum file paths
B3_CONT = 'b3.spw25_27_29_31.cont.I.tt0.pbcor.fits'
B7_CONT = 'b7.spw25_27_29_31.cont.I.selfcal.pbcor.fits'
B9_CONT = 'b9/member.uid___A001_X3845_Xa31._G0.02467-0.0727__sci.spw45_47_49_51_53_55_57_59.cont.I.selfcal.pbcor.fits'

def measure_flux_from_image(image_path, coord, beam=None):
    """Measure flux at a given coordinate from a FITS image."""
    with fits.open(image_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header).celestial

        # Get beam info if available
        if beam is None:
            try:
                beam = radio_beam.Beam.from_fits_header(header)
            except Exception:
                print(f"  Warning: Could not get beam from {os.path.basename(image_path)}")
                beam = None

    # Convert coordinate to pixel
    x, y = wcs.world_to_pixel(coord)
    x, y = int(np.round(x)), int(np.round(y))

    # Handle potential extra dimensions (Stokes axis, etc.)
    while data.ndim > 2:
        data = data[0]

    # Check bounds
    if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
        print(f"  Warning: Coordinate ({x}, {y}) outside image bounds {data.shape}")
        return np.nan, beam

    flux_value = data[y, x]

    return flux_value, beam

print("Measuring continuum flux from ALMA bands...")
print()

# B3: 102 GHz (from header info)
print(f"B3 ({B3_CONT}):")
b3_flux_jy_beam, b3_beam = measure_flux_from_image(B3_CONT, coord)
b3_freq = 102.0 * u.GHz
b3_wl = b3_freq.to(u.um, u.spectral())
b3_flux_jy = b3_flux_jy_beam  # Use Jy/beam value for SED (like notebook does)
if b3_beam is not None:
    print(f"  Flux: {b3_flux_jy_beam:.6f} Jy/beam, Beam: {b3_beam}")
    print(f"  Wavelength: {b3_wl:.3f}")
else:
    print(f"  Flux: {b3_flux_jy_beam:.6f} Jy/beam")
print()

# B7: 350 GHz (from header info)
print(f"B7 ({B7_CONT}):")
b7_flux_jy_beam, b7_beam = measure_flux_from_image(B7_CONT, coord)
b7_freq = 350.0 * u.GHz
b7_wl = b7_freq.to(u.um, u.spectral())
b7_flux_jy = b7_flux_jy_beam  # Use Jy/beam value for SED (like notebook does)
if b7_beam is not None:
    print(f"  Flux: {b7_flux_jy_beam:.6f} Jy/beam, Beam: {b7_beam}")
    print(f"  Wavelength: {b7_wl:.3f}")
else:
    print(f"  Flux: {b7_flux_jy_beam:.6f} Jy/beam")
print()

# B9: Determine frequency from header
print(f"B9 ({B9_CONT}):")
with fits.open(B9_CONT) as hdul:
    header = hdul[0].header
    # Try to get center frequency from header
    if 'CRVAL3' in header and 'CUNIT3' in header and 'CTYPE3' in header:
        if 'FREQ' in header['CTYPE3']:
            b9_freq = header['CRVAL3'] * u.Hz
            b9_freq = b9_freq.to(u.GHz)
        else:
            # Fallback: assume typical Band 9 frequency (~640 GHz for spw45-59)
            b9_freq = 640.0 * u.GHz
    else:
        # Fallback
        b9_freq = 640.0 * u.GHz

b9_flux_jy_beam, b9_beam = measure_flux_from_image(B9_CONT, coord)
b9_wl = b9_freq.to(u.um, u.spectral())
b9_flux_jy = b9_flux_jy_beam  # Use Jy/beam value for SED (like notebook does)
if b9_beam is not None:
    print(f"  Flux: {b9_flux_jy_beam:.6f} Jy/beam, Beam: {b9_beam}")
    print(f"  Frequency: {b9_freq:.1f}, Wavelength: {b9_wl:.3f}")
else:
    print(f"  Flux: {b9_flux_jy_beam:.6f} Jy/beam")
    print(f"  Frequency: {b9_freq:.1f}, Wavelength: {b9_wl:.3f}")
print()

# Load or create full SED table with multi-wavelength data
print("Building full SED table...")
sed_ecsv = 'SED.ecsv'
if os.path.exists(sed_ecsv):
    ulimtbl = Table.read(sed_ecsv)
    print(f"  Loaded {len(ulimtbl)} SED entries from {sed_ecsv}")
else:
    print(f"  Creating SED table from notebook data...")
    # Data from MUBLO_MultiwavelengthCutouts.ipynb - external survey data
    # These are mostly upper limits from Herschel/Spitzer/VLA/SMA
    herschelspitzer = {
        '3.6um': '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SPITZER_stolovy_I1_13368832_0000_6_E8709676_maic.fits',
        '4.5um': '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SPITZER_stolovy_I2_13368832_0000_6_E8709929_maic.fits',
        '5.8um': '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SPITZER_stolovy_I3_13368832_0000_6_E8709933_maic.fits',
        '8.0um': '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SPITZER_stolovy_I4_13368832_0000_6_E8709940_maic.fits',
        '24um': '/orange/adamginsburg/cmz/mipsgal_24micron_data/gc_mosaic_MIPSGAL_gal.fits',
        '70um': '/orange/adamginsburg/cmz/herschel/destripe_l000_blue_wgls_rcal.fits',
        '160um': '/orange/adamginsburg/cmz/herschel/destripe_l000_red_wgls_rcal.fits',
        '250um': '/orange/adamginsburg/cmz/herschel/destripe_l000_PSW_wgls_rcal.fits',
        '350um': '/orange/adamginsburg/cmz/herschel/destripe_l000_PMW_wgls_rcal.fits',
        '500um': '/orange/adamginsburg/cmz/herschel/destripe_l000_PLW_wgls_rcal.fits',
        '60000um': '/orange/adamginsburg/cmz/xinglu/SgrA_CONT_tclean_nterm2.image.tt0.fits',
        '200000um': '/orange/adamginsburg/cmz/meerkat/MeerKAT_Galactic_Centre_1284MHz-StokesI.fits'
    }

    herschelspitzer_resolution = {
        '3.6um': radio_beam.Beam(2*u.arcsec),
        '4.5um': radio_beam.Beam(2*u.arcsec),
        '5.8um': radio_beam.Beam(2*u.arcsec),
        '8.0um': radio_beam.Beam(2*u.arcsec),
        '24um': radio_beam.Beam(6*u.arcsec),
        '70um': radio_beam.Beam(10.7*u.arcsec, 9.7*u.arcsec),
        '160um': radio_beam.Beam(13.9*u.arcsec, 13.2*u.arcsec),
        '250um': radio_beam.Beam(23.9*u.arcsec, 22.8*u.arcsec),
        '350um': radio_beam.Beam(31.3*u.arcsec, 29.3*u.arcsec),
        '500um': radio_beam.Beam(43.8*u.arcsec, 41.1*u.arcsec),
    }

    # Try to add radio beams if files exist
    try:
        herschelspitzer_resolution['60000um'] = radio_beam.Beam.from_fits_header(fits.getheader('/orange/adamginsburg/cmz/xinglu/SgrA_CONT_tclean_nterm2.image.tt0.fits'))
    except:
        pass
    try:
        herschelspitzer_resolution['200000um'] = radio_beam.Beam.from_fits_header(fits.getheader('/orange/adamginsburg/cmz/meerkat/MeerKAT_Galactic_Centre_1284MHz-StokesI.fits'))
    except:
        pass

    ulimtbl = []
    for wl, fn in herschelspitzer.items():
        if not os.path.exists(fn):
            print(f"    Skipping {wl} (file not found: {fn})")
            continue
        try:
            fh = fits.open(fn)
            ww = WCS(fh[0].header).celestial
            xx, yy = np.array(list(map(int, ww.world_to_pixel(coord))))
            val = fh[0].data[yy, xx]
            fh.close()
            if u.Quantity(wl) < 1*u.mm:
                ulimtbl.append((u.Quantity(wl.strip('um'), u.um), val*u.MJy/u.sr,
                                herschelspitzer_resolution[wl].sr,
                                (u.Quantity(val, u.MJy/u.sr)*herschelspitzer_resolution[wl].sr).to(u.Jy)))
            else:
                beam = herschelspitzer_resolution[wl]
                ulimtbl.append([u.Quantity(wl.strip('um'), u.um),
                                val*u.Jy/beam.sr,
                                beam.sr,
                                u.Quantity(val, u.Jy)])
            print(f"    Added {wl}")
        except Exception as e:
            print(f"    Error loading {wl}: {e}")

    # Convert to table
    if ulimtbl:
        ulimtbl = Table(rows=ulimtbl, names=['Wavelength', 'Surface Brightness',
                                             'Beam Area', 'Flux'])
        ulimtbl.sort('Wavelength')
        print(f"  Created table with {len(ulimtbl)} entries")
    else:
        print(f"  Warning: Could not load external SED data")
        ulimtbl = Table(names=['Wavelength', 'Surface Brightness', 'Beam Area', 'Flux'])

# Create updated SED plot with new ALMA continuum points
print("\nCreating updated SED plot...")
fig, ax = plt.subplots(figsize=(9, 4.5))

# Plot existing upper limits and detections
if len(ulimtbl) > 0:
    ulwl = ulimtbl['Wavelength']
    ax.plot(ulwl[ulwl < 800*u.um], ulimtbl['Flux'][ulwl < 800*u.um], 'v',
            markerfacecolor='none', markeredgecolor='k', label='Upper limits (short wavelength)')
    ax.plot(ulwl[ulwl > 1*u.cm], ulimtbl['Flux'][ulwl > 1*u.cm], 'v',
            markerfacecolor='none', markeredgecolor='k', label='Upper limits (long wavelength)')

# Plot new ALMA continuum detections
colors = ['b', 'g', 'purple']  # Different color for B9
labels = ['B3 (102 GHz)', 'B7 (350 GHz)', 'B9 ({:.0f} GHz)'.format(b9_freq.value)]
fluxes = [b3_flux_jy, b7_flux_jy, b9_flux_jy]
wavelengths = [b3_wl, b7_wl, b9_wl]

for wl, flux, color, label in zip(wavelengths, fluxes, colors, labels):
    ax.plot(wl, flux, 's', markeredgecolor='k', markerfacecolor=color,
            markersize=8, label=label, zorder=10)

ax.loglog()
ax.axis([1, 1e5, 5e-4, 300])
ax.set_xlabel("Wavelength [$\mu$m]")
ax.set_ylabel("Flux Density $S_\\nu$ [Jy]")
ax.legend(loc='best', fontsize=10)
ax.grid(True, which='both', alpha=0.3)

outname_png = 'png_figures/SED_with_continuum_measurements.png'
outname_pdf = 'png_figures/SED_with_continuum_measurements.pdf'
os.makedirs(os.path.dirname(outname_png), exist_ok=True)
fig.savefig(outname_png, bbox_inches='tight', dpi=150)
fig.savefig(outname_pdf, bbox_inches='tight')
print(f"  Saved {outname_png}")
print(f"  Saved {outname_pdf}")

# Print summary
print("\n" + "="*60)
print("ALMA Continuum Measurements Summary")
print("="*60)
print(f"{'Band':<10} {'Frequency':<15} {'Wavelength':<20} {'Flux (Jy/beam)':<20}")
print("-"*65)
print(f"{'B3':<10} {str(b3_freq):<15} {str(b3_wl):<20} {b3_flux_jy:>15.6f}")
print(f"{'B7':<10} {str(b7_freq):<15} {str(b7_wl):<20} {b7_flux_jy:>15.6f}")
print(f"{'B9':<10} {str(b9_freq):<15} {str(b9_wl):<20} {b9_flux_jy:>15.6f}")
print("="*65)

plt.close('all')
print("\nDone!")
