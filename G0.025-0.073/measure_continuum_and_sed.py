"""
<<<<<<< HEAD
Measure MUBLO continuum flux from ALMA B3, B7, and B9 and build the SED.

Supersedes the earlier peak-pixel version of this script.  Three corrections
matter for the spectral index:

 1. **Reference frequencies come from the headers**, not hard-coded values.
    The old numbers (102 / 350 / 640 GHz) were wrong by +10% / +3.5% / -0.3%;
    the true MFS reference frequencies are 92.7 / 338.1 / 641.9 GHz.  The B3
    error alone moved alpha(B3-B7) by ~0.15.

 2. **Bands are convolved to a common beam before photometry.**  Native beams
    span a factor of ~30 in solid angle, so peak Jy/beam was not a flux
    density and was not comparable between bands.

 3. **Configurations are matched.**  Each band has (or now has) both a
    compact-configuration and a long-baseline execution:

        B3  ACES 12m                          0.180 x 0.136"
        B7  compact   X67e4                   0.253 x 0.183"
        B7  long      X67e2                   0.061 x 0.048"
        B9  compact   Xa31                    0.368 x 0.258"
        B9  long      Xa2f   <- NEW           0.067 x 0.046"

    Mixing a long-baseline B7 with a compact B9 compares different maximum
    recoverable scales and biases alpha.  We therefore build two internally
    matched SEDs:

      TOTAL  (compact configs)     -> total flux density, the SED to quote
      COMPACT-CORE (long baseline) -> the unresolved core only; the new B9
                                      Xa2f execution is what makes this
                                      possible at 642 GHz for the first time.

Photometry is aperture-integrated (Jy) with a local background subtracted
from an off-source annulus.  The aperture radius is set where the curve of
growth flattens; the curve of growth itself is written out for inspection.

Outputs:
  MUBLO_continuum_SED.ecsv                       photometry, both sets
  MUBLO_continuum_curveofgrowth.ecsv             curve of growth per image
  png_figures/SED_with_continuum_measurements.{png,pdf}
  png_figures/MUBLO_continuum_commonbeam.png     common-beam cutouts
  png_figures/MUBLO_curve_of_growth.png
=======
Measure continuum flux from the 2025.1.00021.S ALMA B3, B7, and B9 images and
overplot on the MUBLO SED.

Includes both Band 9 executions:
  * uid://A001/X3845/Xa31 = TM2 (compact, ~0.37x0.26")
  * uid://A001/X3845/Xa2f = TM1 (extended, ~0.067x0.046") -- released 2026-07

The source is resolved at TM1 resolution (it breaks up into an arc/shell), so
peak Jy/beam is not the flux density.  Fluxes here are aperture-integrated
(Jy/beam summed over the aperture, divided by pixels-per-beam) with a background
taken from a surrounding annulus.  A curve of growth is printed so the aperture
choice can be checked.

Non-ALMA SED points (Spitzer/Herschel/VLA/MeerKAT upper limits) follow
MUBLO_MultiwavelengthCutouts.ipynb.
>>>>>>> e40e19ee3feb8b1f27b10e23c39fffb7ced2875d
"""
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy import units as u
from astropy.convolution import convolve_fft
import radio_beam
from reproject import reproject_interp

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

coord = SkyCoord(ra=266.4905821 * u.deg, dec=-28.9529931 * u.deg, frame='icrs')

# ALMA absolute flux calibration uncertainty by band (fractional).
CALERR = {'B3': 0.05, 'B7': 0.10, 'B9': 0.20}

IMAGES = [
    dict(band='B3', config='compact', color='tab:blue',
         path='b3.spw25_27_29_31.cont.I.tt0.pbcor.fits'),
    dict(band='B7', config='compact', color='tab:green',
         path='b7/member.uid___A001_X3833_X67e4._G0.02467-0.0727__sci'
              '.spw25_27_29_31.cont.I.selfcal.pbcor.fits'),
    dict(band='B9', config='compact', color='tab:purple',
         path='b9/member.uid___A001_X3845_Xa31._G0.02467-0.0727__sci'
              '.spw45_47_49_51_53_55_57_59.cont.I.selfcal.pbcor.fits'),
    dict(band='B3', config='long', color='tab:blue',
         path='b3.spw25_27_29_31.cont.I.tt0.pbcor.fits'),
    dict(band='B7', config='long', color='tab:green',
         path='b7.spw25_27_29_31.cont.I.selfcal.pbcor.fits'),
    dict(band='B9', config='long', color='tab:purple',
         path='member.uid___A001_X3845_Xa2f._G0.02467-0.0727__sci'
              '.spw45_47_49_51_53_55_57_59.cont.I.selfcal.pbcor.fits'),
]

<<<<<<< HEAD
CUTOUT_SIZE = 8.0 * u.arcsec
PIXSCALE = 0.02 * u.arcsec

# Aperture per configuration set.  The compact-config curve of growth flattens
# cleanly at 1.0".  The long-baseline maps suffer a negative bowl from missing
# short spacings -- their curve of growth peaks near 0.7" and then declines
# (B9 goes negative by 2"), so no radius is truly convergent; 0.7" is the
# maximum-recovery point and its flux is best read as a lower bound.
APER = {'compact': 1.0 * u.arcsec, 'long': 0.7 * u.arcsec}
ANN_IN, ANN_OUT = 2.0 * u.arcsec, 3.5 * u.arcsec
COG_RADII = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85,
                      1.0, 1.25, 1.5, 2.0]) * u.arcsec


def load_2d(path):
    """Return (data2d, header2d, beam, reffreq) with degenerate axes dropped."""
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        beam = radio_beam.Beam.from_fits_header(header)

    if str(header.get('CTYPE3', '')).startswith('FREQ'):
        reffreq = (header['CRVAL3'] * u.Hz).to(u.GHz)
    elif 'RESTFRQ' in header:
        reffreq = (header['RESTFRQ'] * u.Hz).to(u.GHz)
    else:
        raise ValueError(f"No frequency information in {path}")

    while data.ndim > 2:
        data = data[0]

    wcs2d = WCS(header).celestial
    header2d = wcs2d.to_header()
    header2d['NAXIS'] = 2
    header2d['NAXIS1'] = data.shape[1]
    header2d['NAXIS2'] = data.shape[0]
    return data, header2d, beam, reffreq


def common_grid_header(center, size, pixscale):
    npix = int(np.round((size / pixscale).decompose().value))
    if npix % 2 == 0:
        npix += 1
    hdr = fits.Header()
    hdr['NAXIS'], hdr['NAXIS1'], hdr['NAXIS2'] = 2, npix, npix
    hdr['CTYPE1'], hdr['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    hdr['CRPIX1'] = hdr['CRPIX2'] = (npix + 1) / 2
    hdr['CRVAL1'], hdr['CRVAL2'] = center.ra.deg, center.dec.deg
    hdr['CDELT1'] = -pixscale.to(u.deg).value
    hdr['CDELT2'] = pixscale.to(u.deg).value
    hdr['CUNIT1'] = hdr['CUNIT2'] = 'deg'
    hdr['RADESYS'] = 'ICRS'
    return hdr


def convolve_to_common(data, beam, target_beam, pixscale_deg):
    """Convolve a Jy/beam map to `target_beam`, conserving flux density.

    Convolution conserves surface brightness; since the map is in Jy/beam and
    the beam grows, rescale by (target area / native area) to hold Jy fixed.
    """
    kernel = target_beam.deconvolve(beam).as_kernel(pixscale_deg * u.deg)
    smoothed = convolve_fft(data, kernel, nan_treatment='interpolate',
                            preserve_nan=True, allow_huge=True)
    return smoothed * (target_beam.sr / beam.sr).decompose().value


def radius_map(header, center):
    wcs2d = WCS(header)
    ny, nx = header['NAXIS2'], header['NAXIS1']
    yy, xx = np.mgrid[:ny, :nx]
    x0, y0 = wcs2d.world_to_pixel(center)
    pixscale = np.abs(header['CDELT1']) * u.deg
    return np.hypot(xx - x0, yy - y0) * pixscale.to(u.arcsec), pixscale


def photometry(data, header, beam, center, radius, bkg=None):
    """Background-subtracted aperture flux (Jy) on a Jy/beam map."""
    rr, pixscale = radius_map(header, center)
    pix_per_beam = (beam.sr / (pixscale.to(u.rad) ** 2)).decompose().value
    if bkg is None:
        ann = (rr > ANN_IN) & (rr < ANN_OUT) & np.isfinite(data)
        bkg = np.nanmedian(data[ann])
    mask = (rr <= radius) & np.isfinite(data)
    flux = np.nansum(data[mask] - bkg) / pix_per_beam
    return flux, mask.sum() / pix_per_beam, bkg


print("=" * 78)
print("MUBLO continuum photometry -- common beam, matched configurations")
print("=" * 78)

# ---- load everything, then set a common beam PER CONFIGURATION SET ---------
for im in IMAGES:
    data, header, beam, reffreq = load_2d(os.path.join(BASE, im['path']))
    im.update(data=data, header=header, beam=beam, reffreq=reffreq)

target_header = common_grid_header(coord, CUTOUT_SIZE, PIXSCALE)

target_beams = {}
for config in ('compact', 'long'):
    members = [im for im in IMAGES if im['config'] == config]
    common = radio_beam.Beams(beams=[im['beam'] for im in members]).common_beam()
    target_beams[config] = radio_beam.Beam(major=common.major * 1.1,
                                           minor=common.minor * 1.1,
                                           pa=common.pa)
    print(f"\n{config.upper()} set -> common beam "
          f"{target_beams[config].major.to(u.arcsec):.4f} x "
          f"{target_beams[config].minor.to(u.arcsec):.4f} "
          f"PA {target_beams[config].pa.to(u.deg):.1f}")
    for im in members:
        print(f"    {im['band']}  nu={im['reffreq']:8.2f}  native beam "
              f"{im['beam'].major.to(u.arcsec):.4f} x "
              f"{im['beam'].minor.to(u.arcsec):.4f}   "
              f"{os.path.basename(im['path'])[:52]}")

# ---- convolve, reproject, measure -----------------------------------------
cog_rows = []
print("\n" + "=" * 78)
print("Photometry")
print("=" * 78)

for im in IMAGES:
    tb = target_beams[im['config']]
    smoothed = convolve_to_common(im['data'], im['beam'], tb,
                                  np.abs(im['header']['CDELT1']))
    regridded, _ = reproject_interp((smoothed, im['header']), target_header)
    im['map'] = regridded
    im['target_beam'] = tb

    rr, _ = radius_map(target_header, coord)
    ann = (rr > ANN_IN) & (rr < ANN_OUT) & np.isfinite(regridded)
    bkg = np.nanmedian(regridded[ann])
    rms = np.nanstd(regridded[ann])

    cog = []
    for R in COG_RADII:
        f, nb, _ = photometry(regridded, target_header, tb, coord, R, bkg=bkg)
        cog.append(f)
        cog_rows.append([im['band'], im['config'], R.to(u.arcsec).value, f])
    im['cog'] = np.array(cog)

    aper = APER[im['config']]
    flux, n_beams, _ = photometry(regridded, target_header, tb, coord, aper,
                                  bkg=bkg)
    stat_err = rms * np.sqrt(n_beams)
    cal_err = CALERR[im['band']] * flux
    im.update(flux=flux, stat_err=stat_err, cal_err=cal_err, rms=rms, bkg=bkg,
              tot_err=np.hypot(stat_err, cal_err),
              peak=np.nanmax(regridded[rr <= 0.5 * u.arcsec]) - bkg)

    print(f"  {im['band']} {im['config']:<8} nu={im['reffreq']:8.2f}  "
          f"S = {flux*1e3:8.2f} +/- {stat_err*1e3:5.2f} (stat) "
          f"+/- {cal_err*1e3:6.2f} (cal) mJy   "
          f"[peak {im['peak']*1e3:7.2f} mJy/bm, rms {rms*1e3:.3f}]")


def spectral_index(subset, label):
    nu = np.array([im['reffreq'].to(u.GHz).value for im in subset])
    S = np.array([im['flux'] for im in subset])
    Serr = np.array([im['stat_err'] for im in subset])
    order = np.argsort(nu)
    nu, S, Serr, subset = nu[order], S[order], Serr[order], [subset[i] for i in order]

    print(f"\n  --- {label} ---")
    for i in range(len(nu) - 1):
        a = np.log(S[i + 1] / S[i]) / np.log(nu[i + 1] / nu[i])
        # statistical only
        da = (np.hypot(Serr[i + 1] / S[i + 1], Serr[i] / S[i])
              / np.abs(np.log(nu[i + 1] / nu[i])))
        # including per-band absolute calibration
        dcal = (np.hypot(CALERR[subset[i + 1]['band']], CALERR[subset[i]['band']])
                / np.abs(np.log(nu[i + 1] / nu[i])))
        print(f"    alpha({subset[i]['band']}-{subset[i+1]['band']}) = "
              f"{a:.2f} +/- {da:.2f} (stat) +/- {dcal:.2f} (cal)")

    coef = np.polyfit(np.log(nu), np.log(S), 1, w=S / Serr)
    print(f"    alpha(all bands, weighted) = {coef[0]:.2f}   "
          f"=> beta = alpha - 2 = {coef[0]-2:.2f} (optically thin, R-J)")
    return coef


print("\n" + "=" * 78)
print("Spectral index   S_nu ~ nu^alpha")
print("=" * 78)
compact_set = [im for im in IMAGES if im['config'] == 'compact']
long_set = [im for im in IMAGES if im['config'] == 'long']
coef_compact = spectral_index(compact_set, "TOTAL flux (compact configs) -- quote this")
coef_long = spectral_index(long_set, "COMPACT CORE (long baselines) -- "
                           "SYSTEMATICS-LIMITED, see note")

print("\n  Fraction of total flux in the compact core:")
for b in ('B3', 'B7', 'B9'):
    c = [im for im in compact_set if im['band'] == b][0]
    l = [im for im in long_set if im['band'] == b][0]
    print(f"    {b}: {100*l['flux']/c['flux']:5.1f}%   "
          f"({l['flux']*1e3:.2f} / {c['flux']*1e3:.2f} mJy)")

# ---- outputs ---------------------------------------------------------------
tbl = QTable(rows=[[im['band'], im['config'],
                    im['reffreq'].to(u.GHz).value,
                    im['reffreq'].to(u.um, u.spectral()).value,
                    im['flux'], im['stat_err'], im['cal_err'], im['peak'],
                    im['rms'], os.path.basename(im['path'])]
                   for im in IMAGES],
             names=['Band', 'Config', 'Frequency', 'Wavelength', 'Flux',
                    'Flux_staterr', 'Flux_calerr', 'Peak', 'RMS', 'File'])
tbl['Frequency'].unit = u.GHz
tbl['Wavelength'].unit = u.um
for c in ('Flux', 'Flux_staterr', 'Flux_calerr'):
    tbl[c].unit = u.Jy
for c in ('Peak', 'RMS'):
    tbl[c].unit = u.Jy / u.beam
tbl.meta['aperture_radius_arcsec'] = {k: v.to(u.arcsec).value
                                      for k, v in APER.items()}
tbl.meta['alpha_total'] = float(coef_compact[0])
tbl.meta['alpha_core'] = float(coef_long[0])
for k, v in target_beams.items():
    tbl.meta[f'commonbeam_{k}_arcsec'] = [v.major.to(u.arcsec).value,
                                          v.minor.to(u.arcsec).value,
                                          v.pa.to(u.deg).value]
tbl.write(os.path.join(BASE, 'MUBLO_continuum_SED.ecsv'), overwrite=True)
print(f"\nWrote MUBLO_continuum_SED.ecsv")

cog_tbl = QTable(rows=cog_rows, names=['Band', 'Config', 'Radius', 'Flux'])
cog_tbl['Radius'].unit = u.arcsec
cog_tbl['Flux'].unit = u.Jy
cog_tbl.write(os.path.join(BASE, 'MUBLO_continuum_curveofgrowth.ecsv'),
              overwrite=True)
print("Wrote MUBLO_continuum_curveofgrowth.ecsv")

figdir = os.path.join(BASE, 'png_figures')
os.makedirs(figdir, exist_ok=True)

# --- SED figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for subset, coef, marker, ls, tag in [
        (compact_set, coef_compact, 's', '--', 'total (compact config)'),
        (long_set, coef_long, 'o', ':', 'compact core (long baseline)')]:
    nu = np.array([im['reffreq'].to(u.GHz).value for im in subset])
    wl = np.array([im['reffreq'].to(u.um, u.spectral()).value for im in subset])
    S = np.array([im['flux'] for im in subset])
    E = np.array([im['tot_err'] for im in subset])
    for im, w_, s_, e_ in zip(subset, wl, S, E):
        ax.errorbar(w_, s_ * 1e3, yerr=e_ * 1e3, fmt=marker, markersize=9,
                    markeredgecolor='k', markerfacecolor=im['color'],
                    ecolor='k', zorder=10)
    wgrid = np.logspace(np.log10(wl.min() * .6), np.log10(wl.max() * 1.6), 50)
    nugrid = (wgrid * u.um).to(u.GHz, u.spectral()).value
    ax.plot(wgrid, np.exp(np.polyval(coef, np.log(nugrid))) * 1e3, 'k',
            ls=ls, lw=1.3, zorder=5,
            label=rf'{tag}: $\alpha={coef[0]:.2f}$')

for im in compact_set:
    ax.annotate(f"{im['band']}\n{im['reffreq'].value:.0f} GHz",
                (im['reffreq'].to(u.um, u.spectral()).value, im['flux'] * 1e3),
                textcoords='offset points', xytext=(6, -18), fontsize=9,
                color=im['color'])

ax.loglog()
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"Flux Density $S_\nu$ [mJy]")
ax.set_title("MUBLO ALMA continuum SED\n"
             f"r={APER['compact'].value:.1f}\"/{APER['long'].value:.1f}\" "
             "aperture, common beam per configuration set",
             fontsize=11)
ax.legend(loc='upper right', fontsize=10)
=======
# Photometry setup: the curve of growth flattens by ~0.75-1.0" in all bands.
APERTURE = 1.0 * u.arcsec
BKG_ANNULUS = (3.0, 4.0) * u.arcsec
CURVE_RADII = np.array([0.25, 0.375, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0]) * u.arcsec

_SG = '2025.1.00021.S/science_goal.uid___A001_X38'

# Absolute flux calibration uncertainty by band (ALMA Technical Handbook)
FLUXCAL_FRAC = {'B3': 0.05, 'B7': 0.10, 'B9': 0.20}

# band, array config, path.  'primary' marks the image used for the SED point:
# the compact (TM2) configurations recover the total flux; TM1 resolves it out.
ALMA_IMAGES = [
    dict(name='B3 TM1', band='B3', config='TM1', primary=True,
         path=_SG + '33_X67e6/group.uid___A001_X3833_X67e7/member.uid___A001_X3833_X67e8/'
                    'product/member.uid___A001_X3833_X67e8._G0.02467-0.0727__sci.'
                    'spw25_27_29_31.cont.I.tt0.pbcor.fits'),
    dict(name='B7 TM1', band='B7', config='TM1', primary=False,
         path=_SG + '33_X67e0/group.uid___A001_X3833_X67e1/member.uid___A001_X3833_X67e2/'
                    'product/member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.'
                    'spw25_27_29_31.cont.I.selfcal.pbcor.fits'),
    dict(name='B7 TM2', band='B7', config='TM2', primary=True,
         path=_SG + '33_X67e0/group.uid___A001_X3833_X67e1/member.uid___A001_X3833_X67e4/'
                    'product/member.uid___A001_X3833_X67e4._G0.02467-0.0727__sci.'
                    'spw25_27_29_31.cont.I.selfcal.pbcor.fits'),
    dict(name='B9 TM1', band='B9', config='TM1', primary=False,
         path=_SG + '45_Xa2d/group.uid___A001_X3845_Xa2e/member.uid___A001_X3845_Xa2f/'
                    'product/member.uid___A001_X3845_Xa2f._G0.02467-0.0727__sci.'
                    'spw45_47_49_51_53_55_57_59.cont.I.selfcal.pbcor.fits'),
    dict(name='B9 TM2', band='B9', config='TM2', primary=True,
         path=_SG + '45_Xa2d/group.uid___A001_X3845_Xa2e/member.uid___A001_X3845_Xa31/'
                    'product/member.uid___A001_X3845_Xa31._G0.02467-0.0727__sci.'
                    'spw45_47_49_51_53_55_57_59.cont.I.selfcal.pbcor.fits'),
]


def aperture_photometry(image_path, coord, aperture=APERTURE, annulus=BKG_ANNULUS):
    """Aperture-integrated flux density from a Jy/beam image.

    Returns a dict with frequency, beam, peak surface brightness, integrated
    flux density, statistical uncertainty, and the curve of growth.
    """
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        data = np.squeeze(hdul[0].data)
        wcs = WCS(header).celestial
        beam = radio_beam.Beam.from_fits_header(header)

    if header.get('CTYPE3', '').startswith('FREQ'):
        freq = (header['CRVAL3'] * u.Hz).to(u.GHz)
    else:
        freq = (header['RESTFRQ'] * u.Hz).to(u.GHz)

    pixscale = np.abs(header['CDELT1']) * 3600 * u.arcsec
    pix_per_beam = float(beam.sr / (pixscale**2).to(u.sr))

    x, y = wcs.world_to_pixel(coord)
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    rad = np.hypot(xx - float(x), yy - float(y)) * pixscale

    bkg_mask = (rad > annulus[0]) & (rad < annulus[1])
    background = np.nanmedian(data[bkg_mask])
    rms = np.nanstd(data[bkg_mask])

    curve = np.array([np.nansum(data[rad < r] - background) / pix_per_beam
                      for r in CURVE_RADII])

    ap_mask = rad < aperture
    flux = np.nansum(data[ap_mask] - background) / pix_per_beam
    npix = np.count_nonzero(ap_mask & np.isfinite(data))
    nbeams = npix / pix_per_beam
    # correlated-noise error: rms per beam times sqrt(number of beams in aperture)
    eflux = rms * np.sqrt(nbeams)

    return dict(freq=freq, wavelength=freq.to(u.um, u.spectral()), beam=beam,
                pixscale=pixscale, pix_per_beam=pix_per_beam,
                peak=np.nanmax(data[ap_mask]) * u.Jy / u.beam,
                background=background, rms=rms * u.Jy / u.beam,
                flux=flux * u.Jy, eflux_stat=eflux * u.Jy,
                nbeams=nbeams, curve=curve)


print("Measuring ALMA continuum flux densities...")
print(f"  aperture radius {APERTURE}, background annulus {BKG_ANNULUS[0]}-{BKG_ANNULUS[1]}")
print()

results = []
for entry in ALMA_IMAGES:
    if not os.path.exists(entry['path']):
        print(f"{entry['name']}: MISSING {entry['path']}")
        continue
    meas = aperture_photometry(entry['path'], coord)
    meas.update(entry)
    meas['eflux'] = np.hypot(meas['eflux_stat'].to_value(u.Jy),
                             FLUXCAL_FRAC[entry['band']] *
                             meas['flux'].to_value(u.Jy)) * u.Jy
    results.append(meas)

    print(f"{entry['name']} ({meas['freq']:.2f} = {meas['wavelength']:.1f}):")
    print(f"  beam {meas['beam'].major.to(u.arcsec):.3f} x "
          f"{meas['beam'].minor.to(u.arcsec):.3f}, "
          f"{meas['pix_per_beam']:.1f} pix/beam")
    print(f"  peak {meas['peak'].to_value(u.Jy/u.beam)*1e3:.3f} mJy/beam, "
          f"annulus rms {meas['rms'].to_value(u.Jy/u.beam)*1e3:.3f} mJy/beam")
    print(f"  S_nu = {meas['flux'].to_value(u.Jy)*1e3:.2f} +/- "
          f"{meas['eflux'].to_value(u.Jy)*1e3:.2f} mJy "
          f"(stat {meas['eflux_stat'].to_value(u.Jy)*1e3:.2f}, "
          f"{FLUXCAL_FRAC[entry['band']]*100:.0f}% cal)")
    print("  curve of growth [mJy]: " +
          " ".join(f"{r.value:.2f}\"={f*1e3:.1f}"
                   for r, f in zip(CURVE_RADII, meas['curve'])))
    print()

# Flux recovered by the extended configuration relative to the compact one
for band in ('B7', 'B9'):
    tm1 = [r for r in results if r['band'] == band and r['config'] == 'TM1']
    tm2 = [r for r in results if r['band'] == band and r['config'] == 'TM2']
    if tm1 and tm2:
        ratio = tm1[0]['flux'] / tm2[0]['flux']
        print(f"{band}: TM1 recovers {ratio.to_value(u.dimensionless_unscaled)*100:.0f}% "
              f"of the TM2 flux -> the source is resolved out at "
              f"{tm1[0]['beam'].major.to(u.arcsec):.3f} resolution")
print()

alma_tbl = Table(dict(
    Name=[r['name'] for r in results],
    Band=[r['band'] for r in results],
    Config=[r['config'] for r in results],
    Frequency=u.Quantity([r['freq'] for r in results]),
    Wavelength=u.Quantity([r['wavelength'] for r in results]),
    BeamMajor=u.Quantity([r['beam'].major.to(u.arcsec) for r in results]),
    BeamMinor=u.Quantity([r['beam'].minor.to(u.arcsec) for r in results]),
    Peak=u.Quantity([r['peak'].to_value(u.Jy/u.beam) for r in results], u.Jy),
    RMS=u.Quantity([r['rms'].to_value(u.Jy/u.beam) for r in results], u.Jy),
    Flux=u.Quantity([r['flux'] for r in results]),
    eFlux=u.Quantity([r['eflux'] for r in results]),
    Primary=[r['primary'] for r in results],
))
alma_tbl.write('alma_continuum_fluxes.ecsv', overwrite=True)
print("Wrote alma_continuum_fluxes.ecsv")

# ---------------------------------------------------------------------------
# Full SED: non-ALMA points (mostly upper limits) from the multiwavelength
# cutouts notebook.
# ---------------------------------------------------------------------------
print("\nBuilding full SED table...")
sed_ecsv = 'SED.ecsv'
if os.path.exists(sed_ecsv):
    ulimtbl = Table.read(sed_ecsv)
    print(f"  Loaded {len(ulimtbl)} SED entries from {sed_ecsv}")
else:
    print("  Creating SED table from notebook data...")
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
    for wl in ('60000um', '200000um'):
        herschelspitzer_resolution[wl] = radio_beam.Beam.from_fits_header(
            fits.getheader(herschelspitzer[wl]))

    ulimtbl = []
    for wl, fn in herschelspitzer.items():
        if not os.path.exists(fn):
            print(f"    Skipping {wl} (file not found: {fn})")
            continue
        fh = fits.open(fn)
        ww = WCS(fh[0].header).celestial
        xx, yy = np.array(list(map(int, ww.world_to_pixel(coord))))
        val = np.squeeze(fh[0].data)[yy, xx]
        fh.close()
        beam = herschelspitzer_resolution[wl]
        if u.Quantity(wl) < 1*u.mm:
            ulimtbl.append((u.Quantity(wl.strip('um'), u.um), val*u.MJy/u.sr,
                            beam.sr,
                            (u.Quantity(val, u.MJy/u.sr)*beam.sr).to(u.Jy)))
        else:
            ulimtbl.append([u.Quantity(wl.strip('um'), u.um),
                            (val*u.Jy/beam.sr).to(u.MJy/u.sr),
                            beam.sr,
                            u.Quantity(val, u.Jy)])
        print(f"    Added {wl}")

    ulimtbl = Table(rows=ulimtbl,
                    names=['Wavelength', 'Surface Brightness', 'Beam Area', 'Flux'])
    ulimtbl.sort('Wavelength')
    ulimtbl.write(sed_ecsv, overwrite=True)
    print(f"  Created table with {len(ulimtbl)} entries -> {sed_ecsv}")

# Published ALMA/SMA detections & limits from the MUBLO paper (arXiv 2404.07808)
literature = Table(dict(
    Wavelength=u.Quantity([850., 1303.4, 2939.1], u.um),
    Flux=u.Quantity([0.0982, 0.0166, 0.00177], u.Jy),
    Label=['ALMA 7m 850um', 'SMA 1.3mm (limit)', 'ALMA 3mm'],
))

# ---------------------------------------------------------------------------
# SED figure
# ---------------------------------------------------------------------------
print("\nCreating updated SED plot...")
fig, ax = plt.subplots(figsize=(9, 5))

ulwl = ulimtbl['Wavelength']
short = ulwl < 800*u.um
long = ulwl > 1*u.cm
ax.plot(ulwl[short], ulimtbl['Flux'][short], 'v', markerfacecolor='none',
        markeredgecolor='k', label='Upper limits (IR)')
ax.plot(ulwl[long], ulimtbl['Flux'][long], 'v', markerfacecolor='none',
        markeredgecolor='gray', label='Upper limits (cm)')
ax.plot(literature['Wavelength'], literature['Flux'], 'o', markerfacecolor='none',
        markeredgecolor='C0', markersize=7, label='Previous ALMA/SMA (2404.07808)')

style = {
    ('B3', 'TM1'): dict(color='tab:blue', marker='s'),
    ('B7', 'TM2'): dict(color='tab:green', marker='s'),
    ('B7', 'TM1'): dict(color='tab:green', marker='x'),
    ('B9', 'TM2'): dict(color='tab:purple', marker='s'),
    ('B9', 'TM1'): dict(color='tab:red', marker='D'),
}
for r in results:
    st = style[(r['band'], r['config'])]
    lbl = (f"{r['band']} {r['config']} ({r['freq'].to_value(u.GHz):.0f} GHz)"
           + ('' if r['primary'] else ', resolved out'))
    ax.errorbar(r['wavelength'].to_value(u.um), r['flux'].to_value(u.Jy),
                yerr=r['eflux'].to_value(u.Jy), marker=st['marker'],
                markeredgecolor='k' if r['primary'] else st['color'],
                markerfacecolor=st['color'] if r['primary'] else 'none',
                color=st['color'], markersize=9 if r['primary'] else 7,
                linestyle='none', label=lbl, zorder=10,
                alpha=1.0 if r['primary'] else 0.6)

ax.loglog()
ax.axis([1, 1e5, 5e-4, 300])
ax.set_xlabel("Wavelength [$\\mu$m]")
ax.set_ylabel("Flux Density $S_\\nu$ [Jy]")
ax.legend(loc='upper left', fontsize=8, ncol=2)
>>>>>>> e40e19ee3feb8b1f27b10e23c39fffb7ced2875d
ax.grid(True, which='both', alpha=0.3)
fig.savefig(os.path.join(figdir, 'SED_with_continuum_measurements.png'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(figdir, 'SED_with_continuum_measurements.pdf'),
            bbox_inches='tight')
print("Wrote png_figures/SED_with_continuum_measurements.png")

<<<<<<< HEAD
# --- curve of growth --------------------------------------------------------
figc, axc = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for axi, config in zip(axc, ('compact', 'long')):
    for im in [i for i in IMAGES if i['config'] == config]:
        norm = im['cog'][np.argmin(np.abs(COG_RADII - APER[config]))]
        axi.plot(COG_RADII.value, im['cog'] / norm, 'o-',
                 color=im['color'], label=f"{im['band']}")
    axi.axvline(APER[config].value, color='k', ls='--', lw=1,
                label='adopted aperture')
    axi.axhline(1.0, color='grey', ls=':', lw=1)
    axi.set_title(f'{config} configuration')
    axi.set_xlabel('aperture radius ["]')
    axi.legend(fontsize=9)
    axi.grid(alpha=0.3)
axc[0].set_ylabel('flux / flux(adopted aperture)')
figc.suptitle('MUBLO continuum curve of growth (background-subtracted)')
figc.savefig(os.path.join(figdir, 'MUBLO_curve_of_growth.png'),
             bbox_inches='tight', dpi=150)
print("Wrote png_figures/MUBLO_curve_of_growth.png")
=======
os.makedirs('png_figures', exist_ok=True)
for ext in ('png', 'pdf'):
    outname = f'png_figures/SED_with_continuum_measurements.{ext}'
    fig.savefig(outname, bbox_inches='tight', dpi=150)
    print(f"  Saved {outname}")

# Zoom on the ALMA (sub)mm points, where the new B9 data live
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(literature['Wavelength'], literature['Flux'], 'o', markerfacecolor='none',
         markeredgecolor='C0', markersize=8, label='Previous ALMA/SMA')
for r in results:
    st = style[(r['band'], r['config'])]
    ax2.errorbar(r['wavelength'].to_value(u.um), r['flux'].to_value(u.Jy),
                 yerr=r['eflux'].to_value(u.Jy), marker=st['marker'],
                 markeredgecolor='k' if r['primary'] else st['color'],
                 markerfacecolor=st['color'] if r['primary'] else 'none',
                 color=st['color'], markersize=10 if r['primary'] else 8,
                 linestyle='none', zorder=10,
                 label=f"{r['band']} {r['config']}"
                       + ('' if r['primary'] else ' (resolved out)'))

# spectral index through the primary (flux-recovering) points
prim = sorted([r for r in results if r['primary']], key=lambda r: r['freq'].value)
nu = np.array([r['freq'].to_value(u.GHz) for r in prim])
sn = np.array([r['flux'].to_value(u.Jy) for r in prim])
alpha, logA = np.polyfit(np.log10(nu), np.log10(sn), 1)
nugrid = np.logspace(np.log10(nu.min()/1.3), np.log10(nu.max()*1.3), 50)
ax2.plot((nugrid*u.GHz).to_value(u.um, u.spectral()), 10**logA * nugrid**alpha,
         'k--', alpha=0.6, label=rf'$S_\nu \propto \nu^{{{alpha:.2f}}}$')
for i in range(len(prim)-1):
    a = (np.log10(sn[i+1]/sn[i])) / (np.log10(nu[i+1]/nu[i]))
    print(f"  spectral index {prim[i]['band']}->{prim[i+1]['band']}: alpha = {a:.2f}")
print(f"  overall power-law fit through primary points: alpha = {alpha:.2f}")

ax2.loglog()
ax2.set_xlabel("Wavelength [$\\mu$m]")
ax2.set_ylabel("Flux Density $S_\\nu$ [Jy]")
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, which='both', alpha=0.3)
for ext in ('png', 'pdf'):
    outname = f'png_figures/SED_mm_zoom_with_B9.{ext}'
    fig2.savefig(outname, bbox_inches='tight', dpi=150)
    print(f"  Saved {outname}")

# LaTeX table for the paper
with open('alma_continuum.tex', 'w') as fh:
    fh.write("\\begin{table}[htp]\n\\centering\n")
    fh.write("\\caption{ALMA continuum flux densities of the MUBLO "
             "(2025.1.00021.S), measured in a "
             f"{APERTURE.value:.1f}\\arcsec-radius aperture.}}\n")
    fh.write("\\label{tab:almacont}\n")
    fh.write("\\begin{tabular}{lccccc}\n")
    fh.write("Band & Config & $\\nu$ & $\\theta_{maj}\\times\\theta_{min}$ "
             "& $S_\\nu$ & rms \\\\\n")
    fh.write(" &  & GHz & \\arcsec & mJy & mJy beam$^{-1}$ \\\\\n\\hline\n")
    for r in results:
        fh.write(f"{r['band']} & {r['config']} & {r['freq'].to_value(u.GHz):.1f} & "
                 f"{r['beam'].major.to_value(u.arcsec):.3f}$\\times$"
                 f"{r['beam'].minor.to_value(u.arcsec):.3f} & "
                 f"{r['flux'].to_value(u.Jy)*1e3:.1f} $\\pm$ "
                 f"{r['eflux'].to_value(u.Jy)*1e3:.1f} & "
                 f"{r['rms'].to_value(u.Jy/u.beam)*1e3:.3f} \\\\\n")
    fh.write("\\hline\n\\end{tabular}\n\\end{table}\n")
print("  Wrote alma_continuum.tex")

print("\n" + "="*78)
print("ALMA Continuum Measurements Summary "
      f"(aperture r = {APERTURE.value:.1f}\")")
print("="*78)
print(f"{'Band':<8}{'Cfg':<6}{'Freq [GHz]':>12}{'Wl [um]':>10}"
      f"{'S_nu [mJy]':>14}{'peak [mJy/bm]':>16}")
print("-"*78)
for r in results:
    print(f"{r['band']:<8}{r['config']:<6}{r['freq'].to_value(u.GHz):>12.2f}"
          f"{r['wavelength'].to_value(u.um):>10.1f}"
          f"{r['flux'].to_value(u.Jy)*1e3:>9.2f} +/-{r['eflux'].to_value(u.Jy)*1e3:>5.2f}"
          f"{r['peak'].to_value(u.Jy/u.beam)*1e3:>16.3f}")
print("="*78)
>>>>>>> e40e19ee3feb8b1f27b10e23c39fffb7ced2875d

# --- common-beam cutouts ----------------------------------------------------
fig2, axes = plt.subplots(2, 3, figsize=(13, 8.4))
for row, config in enumerate(('compact', 'long')):
    for axi, im in zip(axes[row], [i for i in IMAGES if i['config'] == config]):
        n = target_header['NAXIS1']
        half = int(2.0 / PIXSCALE.value)
        sl = slice(n // 2 - half, n // 2 + half)
        cut = im['map'][sl, sl]
        pcm = axi.imshow(cut * 1e3, origin='lower', cmap='inferno')
        axi.set_title(f"{im['band']} {config}  {im['reffreq'].value:.0f} GHz\n"
                      f"{im['flux']*1e3:.2f} mJy", fontsize=10)
        plt.colorbar(pcm, ax=axi, label='mJy/beam')
        axi.set_xticks([]); axi.set_yticks([])
fig2.suptitle('MUBLO continuum, convolved to the common beam of each '
              'configuration set (central 4")')
fig2.savefig(os.path.join(figdir, 'MUBLO_continuum_commonbeam.png'),
             bbox_inches='tight', dpi=150)
print("Wrote png_figures/MUBLO_continuum_commonbeam.png")
plt.close('all')

print("\n" + "=" * 78)
print("NOTE: the two sets answer different questions.  The compact-config set")
print("gives total flux density and is the SED to quote.  The long-baseline")
print("set measures only the unresolved core; comparing the two gives the")
print("compact fraction printed above.  Do not mix configurations within one")
print("spectral index -- that was the bug in the previous version.")
print("=" * 78)
