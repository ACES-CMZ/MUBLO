"""
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
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
import radio_beam

# Source coordinates (MUBLO)
coord = SkyCoord(ra=266.4905821*u.deg, dec=-28.9529931*u.deg, frame='icrs')

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
ax.grid(True, which='both', alpha=0.3)

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

plt.close('all')
print("\nDone!")
