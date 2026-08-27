"""
Measure the CS J=13-12 line (636.53246 GHz) in the Band 9 cubes of
2025.1.00021.S toward the MUBLO.

CS 13-12 falls only in spw45.  That spw covers roughly -645 to +236 km/s
relative to the CS 13-12 rest frequency, so for a MUBLO-width line
(v_cen ~ 45 km/s, sigma ~ 160 km/s from the 3 mm fits in spectral_fits.tex)
the blue half is fully covered and the red wing is truncated beyond ~+236 km/s.

Both executions are measured:
  Xa2f = TM1 (extended, 0.072x0.048")
  Xa31 = TM2 (compact,  0.371x0.258")

The pipeline `cube` products are continuum-subtracted, so the spectra sit on a
zero baseline.

Significance is assessed with a matched filter: the aperture spectrum is
correlated with a Gaussian template fixed at the 3 mm line parameters.

The noise on that statistic is measured EMPIRICALLY, by running the identical
aperture + filter at 24 source-free positions on rings 2.5" and 3.5" from the
source in the same cube.  This is deliberate: the line is nearly as wide as the
band, so the matched filter is sensitive to baseline curvature and to the
negative bowls around resolved-out emission, and a thermal-noise-only error bar
badly underestimates the true uncertainty.  The off-position scatter captures
thermal noise, bowls, and continuum-subtraction residuals together.

Spectra are extracted at several aperture radii because the optimal aperture is
not obvious: the source is ~0.6" across, so a 1" aperture (used for the
continuum photometry) adds noise without adding signal.

Extracted spectra are cached in b9_cs1312_spectra.npz; delete that file to
force re-extraction (the cubes are 20 GB each).
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
from astropy.table import Table
from astropy.modeling import models, fitting
from astropy import units as u
from astropy import constants
import radio_beam

warnings.filterwarnings('ignore')

coord = SkyCoord(ra=266.4905821*u.deg, dec=-28.9529931*u.deg, frame='icrs')

CS1312 = 636.53246 * u.GHz  # JPL, v=0
APERTURES = np.array([0.25, 0.5, 0.75, 1.0])   # arcsec
# template from the 3 mm line fits (spectral_fits.tex): CS 2-1 v_cen 47,
# sigma 164 km/s; SO 3-2 v_cen 40, sigma 161 km/s
V_TEMPLATE = 45.0
SIGMA_TEMPLATE = 160.0
BIN_KMS = 20.0        # channel width for display and fitting
LINEFREE_V = 350.0    # |v - v_template| beyond this is treated as line-free

CACHE = 'b9_cs1312_spectra.npz'

_GRP = ('2025.1.00021.S/science_goal.uid___A001_X3845_Xa2d/'
        'group.uid___A001_X3845_Xa2e/member.uid___A001_X3845_')
CUBES = {
    'TM1': _GRP + ('Xa2f/product/member.uid___A001_X3845_Xa2f._G0.02467-0.0727__sci.'
                   'spw45.cube.I.selfcal.pbcor.fits'),
    'TM2': _GRP + ('Xa31/product/member.uid___A001_X3845_Xa31._G0.02467-0.0727__sci.'
                   'spw45.cube.I.selfcal.pbcor.fits'),
}


CUTOUT_RADIUS = 4.0          # arcsec half-size of the extracted region
OFFSET_RINGS = (2.5, 3.5)    # arcsec, radii of the source-free test positions
N_PER_RING = 12
CHUNK = 128                  # channels read at a time


def extract(cube_path, coord, apertures=APERTURES, restfreq=CS1312):
    """Aperture spectra (Jy) at the source and at source-free test positions."""
    hdul = fits.open(cube_path, memmap=True)
    header = hdul[0].header
    cwcs = WCS(header).celestial
    beam = radio_beam.Beam.from_fits_header(header)

    pixscale = np.abs(header['CDELT1']) * 3600
    pix_per_beam = float(beam.sr / ((pixscale*u.arcsec)**2).to(u.sr))

    nchan = header['NAXIS3']
    freq = ((np.arange(nchan) + 1 - header['CRPIX3']) * header['CDELT3']
            + header['CRVAL3']) * u.Hz
    vel = (constants.c * (1 - freq / restfreq)).to_value(u.km/u.s)

    x, y = cwcs.world_to_pixel(coord)
    rpix = int(np.ceil(CUTOUT_RADIUS / pixscale))
    xi, yi = int(round(float(x))), int(round(float(y)))

    yy, xx = np.mgrid[:2*rpix+1, :2*rpix+1]
    # source position within the cutout
    cx, cy = float(x) - (xi - rpix), float(y) - (yi - rpix)

    # aperture masks: index 0 is the source, the rest are the offset positions
    centres = [(cx, cy)]
    for rring in OFFSET_RINGS:
        for k in range(N_PER_RING):
            th = 2*np.pi*k/N_PER_RING + (0.5*np.pi/N_PER_RING if rring > 3 else 0)
            centres.append((cx + rring*np.cos(th)/pixscale,
                            cy + rring*np.sin(th)/pixscale))
    masks = {}
    nbeams = {}
    for ap in apertures:
        ms = [np.hypot(xx-px, yy-py)*pixscale < ap for px, py in centres]
        masks[ap] = ms
        nbeams[ap] = ms[0].sum() / pix_per_beam

    specs = {ap: np.zeros((len(centres), nchan)) for ap in apertures}
    noise_chan = []

    data = hdul[0].data
    if data.ndim == 4:
        data = data[0]
    for c0 in range(0, nchan, CHUNK):
        c1 = min(c0+CHUNK, nchan)
        blk = np.array(data[c0:c1, yi-rpix:yi+rpix+1, xi-rpix:xi+rpix+1],
                       dtype='float32')
        blk[~np.isfinite(blk)] = 0
        for ap in apertures:
            for i, m in enumerate(masks[ap]):
                specs[ap][i, c0:c1] = blk[:, m].sum(axis=1) / pix_per_beam
        if c0 == 0 or c0 >= nchan//2 and len(noise_chan) < 2:
            ann = (np.hypot(xx-cx, yy-cy)*pixscale > 1.5) & \
                  (np.hypot(xx-cx, yy-cy)*pixscale < 3.0)
            v = blk[blk.shape[0]//2][ann]
            noise_chan.append(np.median(np.abs(v-np.median(v)))*1.4826)
    hdul.close()

    return dict(vel=vel, specs={ap: specs[ap][0] for ap in apertures},
                offspecs={ap: specs[ap][1:] for ap in apertures},
                nbeams=nbeams, beam=beam, pixscale=pixscale,
                pix_per_beam=pix_per_beam, rms_beam=float(np.median(noise_chan)),
                chanwidth=float(np.abs(np.median(np.diff(vel)))))


def rebin(vel, spec, width):
    """Bin a spectrum to approximately `width` km/s channels."""
    n = max(1, int(round(width / np.abs(np.median(np.diff(vel))))))
    nkeep = (len(vel) // n) * n
    v = vel[:nkeep].reshape(-1, n).mean(axis=1)
    s = spec[:nkeep].reshape(-1, n).mean(axis=1)
    return v, s, n


def mf_stat(vel, spec, vcen, sigma):
    """Optimally weighted line flux estimate (Jy km/s) for a fixed template."""
    w = np.exp(-0.5*((vel - vcen)/sigma)**2)
    dv = np.abs(np.median(np.diff(vel)))
    # least-squares amplitude for template w:  A = sum(w*s)/sum(w^2)
    amp = np.sum(w*spec)/np.sum(w**2)
    return amp * sigma * np.sqrt(2*np.pi), amp


def channel_rms(vel, spec, vcen=V_TEMPLATE, exclude=LINEFREE_V):
    """Robust per-channel noise from the line-free part of the band.

    The band only extends ~1.2 sigma redward of the line, so the line-free
    window is one-sided (the blue end).  MAD over the full band is reported as
    a cross-check: a real line would inflate it, so it is the conservative one.
    """
    linefree = np.abs(vel - vcen) > exclude
    s = spec[linefree]
    rms_lf = np.median(np.abs(s - np.median(s))) * 1.4826 if s.size > 5 else np.nan
    rms_all = np.median(np.abs(spec - np.median(spec))) * 1.4826
    return rms_lf, rms_all, int(linefree.sum())


def mf_uncertainty(vel, sigma, rms_ch, vcen=V_TEMPLATE):
    """Analytic uncertainty on the matched-filter line flux.

    For A = sum(w*s)/sum(w^2) with independent per-channel noise rms_ch,
    var(A) = rms_ch^2 / sum(w^2), and S_int = A*sigma*sqrt(2*pi).
    """
    w = np.exp(-0.5*((vel - vcen)/sigma)**2)
    return rms_ch / np.sqrt(np.sum(w**2)) * sigma * np.sqrt(2*np.pi)


def mf_offsets(vel, spec, sigma, vcen=V_TEMPLATE, exclude=300.0):
    """Empirical cross-check: filter statistic at off-line velocity offsets."""
    trials = [mf_stat(vel, spec, v0, sigma)[0]
              for v0 in np.arange(vel.min() + sigma, vel.max() - sigma, 20.0)
              if abs(v0 - vcen) > exclude]
    return (np.std(trials), len(trials)) if len(trials) > 3 else (np.nan, len(trials))


# ---------------------------------------------------------------------------
if os.path.exists(CACHE):
    print(f"Loading cached spectra from {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    data = {k: z[k].item() for k in z.files}
else:
    data = {}
    for cfg, path in CUBES.items():
        if not os.path.exists(path):
            print(f"{cfg}: MISSING {path}")
            continue
        print(f"Extracting {cfg} from {os.path.basename(path)} ...", flush=True)
        data[cfg] = extract(path, coord)
    np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in data.items()})
    print(f"Cached spectra -> {CACHE}")

print(f"\nCS J=13-12, rest {CS1312:.5f}")
print(f"Template fixed at v_cen = {V_TEMPLATE:.0f}, sigma = {SIGMA_TEMPLATE:.0f} km/s "
      "(from the 3 mm CS 2-1 / SO 3-2 fits)\n")

rows = []
for cfg, ex in data.items():
    vel = ex['vel']
    print(f"=== B9 {cfg}: beam {ex['beam'].major.to(u.arcsec):.3f} x "
          f"{ex['beam'].minor.to(u.arcsec):.3f}, {ex['pixscale']:.4f}\"/pix")
    print(f"    coverage {vel.min():.0f} to {vel.max():.0f} km/s, "
          f"native channel {ex['chanwidth']:.2f} km/s")
    print(f"    per-beam rms {ex['rms_beam']*1e3:.1f} mJy/beam per native channel "
          f"-> {ex['rms_beam']*1e3*np.sqrt(ex['chanwidth']/10.):.1f} mJy/beam at 10 km/s")
    frac_covered = ((vel.max() - V_TEMPLATE) / SIGMA_TEMPLATE)
    print(f"    red wing truncated at +{frac_covered:.1f} sigma")
    for ap in APERTURES:
        v, s, nbin = rebin(vel, ex['specs'][ap], BIN_KMS)
        flux, amp = mf_stat(v, s, V_TEMPLATE, SIGMA_TEMPLATE)
        rms_lf, rms_all, nlf = channel_rms(v, s)
        thermal = mf_uncertainty(v, SIGMA_TEMPLATE, rms_lf)

        # empirical: same filter at the source-free positions
        offvals = []
        for off in ex['offspecs'][ap]:
            vo, so, _ = rebin(vel, off, BIN_KMS)
            offvals.append(mf_stat(vo, so, V_TEMPLATE, SIGMA_TEMPLATE)[0])
        offvals = np.array(offvals)
        noise = offvals.std()
        bias = offvals.mean()
        snr = (flux - bias)/noise

        rows.append(dict(Config=cfg, Aperture=ap, NBeams=ex['nbeams'][ap],
                         ChanRMS=rms_lf, Amp=amp, Flux=flux, Bias=bias,
                         eFlux=noise, eFluxThermal=thermal, SNR=snr))
        print(f"    r={ap:.2f}\" ({ex['nbeams'][ap]:5.1f} beams): "
              f"amp {amp*1e3:7.2f} mJy | "
              f"S_int {flux:7.2f} Jy km/s, off-position mean {bias:6.2f} "
              f"+/- {noise:5.2f} (thermal-only would be {thermal:.2f}) "
              f"-> {snr:5.1f} sigma")
    print()

tbl = Table(rows)
tbl.write('b9_cs1312_matchedfilter.ecsv', overwrite=True)
print("Wrote b9_cs1312_matchedfilter.ecsv\n")

# free Gaussian fit at the most sensitive aperture (smallest error bar)
best = {}
for cfg in data:
    sub = tbl[tbl['Config'] == cfg]
    best[cfg] = float(sub['Aperture'][np.argmin(sub['eFlux'])])

fitres = {}
for cfg, ex in data.items():
    ap = best[cfg]
    v, s, _ = rebin(ex['vel'], ex['specs'][ap], BIN_KMS)
    init = (models.Gaussian1D(amplitude=np.nanmax(s), mean=V_TEMPLATE,
                              stddev=SIGMA_TEMPLATE)
            + models.Linear1D(slope=0, intercept=0))
    init[0].mean.bounds = (-150, 250)
    init[0].stddev.bounds = (40, 350)
    fitter = fitting.LevMarLSQFitter()
    ok = np.isfinite(s)
    fit = fitter(init, v[ok], s[ok], maxiter=5000)
    resid = s - fit(v)
    linefree = np.abs(v - V_TEMPLATE) > LINEFREE_V
    rms = np.std(resid[linefree]) if linefree.sum() > 5 else np.std(resid)
    fitres[cfg] = dict(ap=ap, v=v, s=s, fit=fit, rms=rms,
                       amp=fit[0].amplitude.value, vcen=fit[0].mean.value,
                       sigma=abs(fit[0].stddev.value))
    print(f"B9 {cfg} free fit (r={ap}\", {BIN_KMS:.0f} km/s channels): "
          f"amp {fit[0].amplitude.value*1e3:.1f} mJy, v_cen {fit[0].mean.value:.0f}, "
          f"sigma {abs(fit[0].stddev.value):.0f} km/s; "
          f"binned rms {rms*1e3:.1f} mJy -> peak SNR "
          f"{fit[0].amplitude.value/rms:.1f}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(len(data), 1, figsize=(9, 4*len(data)), squeeze=False)
for ax, (cfg, ex) in zip(axs[:, 0], data.items()):
    fr = fitres[cfg]
    ap = fr['ap']
    ax.step(fr['v'], fr['s']*1e3, where='mid', color='k', lw=1.0,
            label=f'data, r={ap}" aperture, {BIN_KMS:.0f} km/s')
    ax.axhspan(-fr['rms']*1e3, fr['rms']*1e3, color='C0', alpha=0.15,
               label=f"$\\pm1\\sigma$ = {fr['rms']*1e3:.0f} mJy")
    ax.plot(fr['v'], fr['fit'](fr['v'])*1e3, 'r-', lw=1.5,
            label=(f"free fit: $v$={fr['vcen']:.0f}, "
                   f"$\\sigma$={fr['sigma']:.0f} km/s"))
    row = tbl[(tbl['Config'] == cfg) & (tbl['Aperture'] == ap)][0]
    tmpl = row['Amp']*np.exp(-0.5*((fr['v']-V_TEMPLATE)/SIGMA_TEMPLATE)**2)
    ax.plot(fr['v'], tmpl*1e3, 'g--', lw=1.5,
            label=(f"3 mm template: {row['Flux']:.1f}$\\pm${row['eFlux']:.1f} "
                   f"Jy km/s ({row['SNR']:.1f}$\\sigma$)"))
    ax.axvline(V_TEMPLATE, color='gray', ls=':', lw=1)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('Flux density [mJy]')
    ax.set_title(f"CS J=13-12, B9 {cfg} "
                 f"({ex['beam'].major.to_value(u.arcsec):.3f}\"x"
                 f"{ex['beam'].minor.to_value(u.arcsec):.3f}\")", fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
axs[-1, 0].set_xlabel('$V_{LSR}$ [km s$^{-1}$] (CS 13-12, 636.53246 GHz)')
fig.tight_layout()
os.makedirs('png_figures', exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(f'png_figures/CS1312_B9_spectra.{ext}', bbox_inches='tight', dpi=150)
    print(f"Saved png_figures/CS1312_B9_spectra.{ext}")

# ---------------------------------------------------------------------------
# Context: what would CS 13-12 look like if it tracked the 3 mm CS 2-1 line?
# CS 2-1 from spectral_fits.tex (ACES): amplitude 1.032 K, sigma 164.5 km/s,
# in a 1.626" x 1.440" beam at 97.98095 GHz.
# ---------------------------------------------------------------------------
cs21_amp_K = 1.032 * u.K
cs21_sigma = 164.5 * u.km/u.s
cs21_freq = 97.98095 * u.GHz
cs21_beam = radio_beam.Beam(1.626*u.arcsec, 1.440*u.arcsec)
jy_per_k = (1*u.K).to(u.Jy, u.brightness_temperature(cs21_freq, cs21_beam))
cs21_amp = (cs21_amp_K.value * jy_per_k)
cs21_int = cs21_amp * cs21_sigma.value * np.sqrt(2*np.pi)
print(f"\n3 mm CS 2-1 for reference: {cs21_amp.to_value(u.Jy)*1e3:.1f} mJy peak "
      f"({jy_per_k.to_value(u.Jy):.4f} Jy/K in a "
      f"{cs21_beam.major.to_value(u.arcsec):.2f}\"x"
      f"{cs21_beam.minor.to_value(u.arcsec):.2f}\" beam), "
      f"S_int = {cs21_int.to_value(u.Jy):.2f} Jy km/s")
# optically thick, thermalised gas filling the same solid angle would scale as nu^2
thick = cs21_int.to_value(u.Jy) * (CS1312/cs21_freq).to_value(u.dimensionless_unscaled)**2
print(f"  optically thick + thermalised (S ~ nu^2) would give "
      f"CS 13-12 S_int ~ {thick:.0f} Jy km/s")

print("\n" + "="*72)
print("CS J=13-12 detection summary (matched filter, 3 mm line template)")
print("="*72)
for cfg in data:
    sub = tbl[tbl['Config'] == cfg]
    j = int(np.argmin(sub['eFlux']))
    snr = sub['SNR'][j]
    verdict = ("DETECTED" if snr > 5 else "marginal" if snr > 3 else "NOT DETECTED")
    print(f"B9 {cfg}: {verdict} -- most sensitive aperture r={sub['Aperture'][j]:.2f}\", "
          f"S_int = {sub['Flux'][j]:.2f} +/- {sub['eFlux'][j]:.2f} Jy km/s "
          f"({snr:.1f} sigma)")
    if snr < 3:
        lim = 3*sub['eFlux'][j]
        print(f"          3-sigma upper limit: S_int < {lim:.2f} Jy km/s "
              f"= {lim/cs21_int.to_value(u.Jy):.2f} x the CS 2-1 integrated flux, "
              f"{lim/thick:.4f} x the optically-thick thermalised expectation")
    print(f"          (all apertures: "
          + ", ".join(f"r={a:.2f}\":{s:+.1f}sig" for a, s in
                      zip(sub['Aperture'], sub['SNR'])) + ")")
print("="*72)
