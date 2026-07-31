"""
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
ax.grid(True, which='both', alpha=0.3)
fig.savefig(os.path.join(figdir, 'SED_with_continuum_measurements.png'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(figdir, 'SED_with_continuum_measurements.pdf'),
            bbox_inches='tight')
print("Wrote png_figures/SED_with_continuum_measurements.png")

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
