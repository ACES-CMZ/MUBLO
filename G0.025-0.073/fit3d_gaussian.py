"""
Fit a 3D (elliptical-spatial, Gaussian-spectral) Gaussian model to the
SO(3-2) MUBLO cube, write model and residual cubes, and use the analytic
model as a matched-filter template on B3 and B7 cubes.

Avoids spectral-cube because dask/distributed are broken in this env.
"""
import glob
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit

BASE = os.path.dirname(os.path.abspath(__file__))
SO32_CUBE = os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits")


def spectral_axis_kms(header, nchan=None):
    """Return velocity axis in km/s from a FITS header, regardless of
    whether WCS reports SI m/s or header-declared km/s."""
    if nchan is None:
        nchan = header["NAXIS3"]
    crpix = header["CRPIX3"]
    crval = header["CRVAL3"]
    cdelt = header["CDELT3"]
    unit = str(header.get("CUNIT3", "")).strip().lower()
    scale = 1.0
    if unit in ("m/s", "m s-1", "m s^-1"):
        scale = 1e-3
    vel = crval + cdelt * (np.arange(nchan) + 1 - crpix)
    return vel * scale  # km/s


def sky_axes_deg(header):
    """Return 1D RA and Dec arrays in degrees for the cube's spatial grid
    at the image reference Dec (good enough near the equator of the tangent
    plane; the cube is tiny, 250x250 pix at ~0.024"/pix)."""
    wcs_cel = WCS(header).celestial
    ny = header["NAXIS2"]
    nx = header["NAXIS1"]
    y, x = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs_cel.wcs_pix2world(x, y, 0)
    return ra, dec  # 2D arrays in degrees


def gauss3d(coords, amp, x0, y0, sx, sy, v0, sv, offset):
    """3D Gaussian: elliptical axis-aligned in sky, Gaussian in velocity.

    coords: (ra_deg_flat, dec_deg_flat, vel_kms_flat) - each shape (N,)
    Returns flattened model.
    """
    ra, dec, vel = coords
    # Use tangent-plane offsets in arcsec relative to (x0, y0)
    cosd = np.cos(np.deg2rad(y0))
    dx = (ra - x0) * cosd * 3600.0
    dy = (dec - y0) * 3600.0
    dv = vel - v0
    arg = 0.5 * (dx ** 2 / sx ** 2 + dy ** 2 / sy ** 2 + dv ** 2 / sv ** 2)
    return offset + amp * np.exp(-arg)


def gauss3d_on_grid(params, ra2d, dec2d, vel1d):
    """Evaluate gauss3d on a (nv, ny, nx) grid without flattening."""
    amp, x0, y0, sx, sy, v0, sv, offset = params
    cosd = np.cos(np.deg2rad(y0))
    dx = (ra2d - x0) * cosd * 3600.0  # (ny, nx)
    dy = (dec2d - y0) * 3600.0
    spatial_arg = 0.5 * (dx ** 2 / sx ** 2 + dy ** 2 / sy ** 2)  # (ny, nx)
    dv = vel1d - v0  # (nv,)
    spec_arg = 0.5 * (dv ** 2 / sv ** 2)  # (nv,)
    arg = spatial_arg[None, :, :] + spec_arg[:, None, None]
    return offset + amp * np.exp(-arg)


def fit_so32():
    print(f"Loading {SO32_CUBE}")
    data = fits.getdata(SO32_CUBE).astype(np.float64)
    header = fits.getheader(SO32_CUBE)
    # Drop degenerate Stokes axis if present
    while data.ndim > 3:
        data = data[0]
    nv, ny, nx = data.shape
    print(f"  cube shape (v,y,x) = {data.shape}")

    vel = spectral_axis_kms(header, nv)  # km/s
    ra2d, dec2d = sky_axes_deg(header)  # deg
    print(f"  velocity range: {vel.min():.1f} to {vel.max():.1f} km/s, dv={vel[1]-vel[0]:.3f} km/s")

    # Robust noise estimate from channels far from MUBLO (|v-45|>200 km/s)
    off_signal = np.abs(vel - 45.0) > 200.0
    if off_signal.sum() < 4:
        off_signal = np.abs(vel - 45.0) > 150.0
    rms = np.nanstd(data[off_signal])
    print(f"  off-signal rms = {rms:.3e} Jy/beam  (n_off_chans={off_signal.sum()})")

    # Initial guesses
    valid = np.isfinite(data)
    data_for_init = np.where(valid, data, 0.0)
    # moment-0 near MUBLO velocity window
    near = np.abs(vel - 45.0) <= 150.0
    m0 = data_for_init[near].sum(axis=0)
    # peak spatial location
    iy, ix = np.unravel_index(np.nanargmax(m0), m0.shape)
    x0_init = float(ra2d[iy, ix])
    y0_init = float(dec2d[iy, ix])
    # spectrum at peak
    spec = data_for_init[:, iy, ix]
    v0_init = float(np.nansum(vel * np.clip(spec, 0, None)) / max(np.nansum(np.clip(spec, 0, None)), 1e-30))
    sv_init = 160.0 / 2.355  # from paper FWHM
    # spatial size from m0 FWHM-ish
    peak_m0 = m0.max()
    above = m0 > 0.5 * peak_m0
    if above.sum() > 3:
        ys, xs = np.where(above)
        sx_arcsec = np.std(ra2d[ys, xs] * np.cos(np.deg2rad(y0_init)) * 3600.0)
        sy_arcsec = np.std(dec2d[ys, xs] * 3600.0)
    else:
        sx_arcsec = 0.3
        sy_arcsec = 0.3
    amp_init = float(np.nanmax(data))
    offset_init = 0.0

    p0 = [amp_init, x0_init, y0_init, max(sx_arcsec, 0.1), max(sy_arcsec, 0.1),
          v0_init, sv_init, offset_init]
    print(f"  initial guess: A={p0[0]:.3e}, x0={p0[1]:.6f} deg, y0={p0[2]:.6f} deg, "
          f"sx={p0[3]:.3f}\", sy={p0[4]:.3f}\", v0={p0[5]:.1f}, sv={p0[6]:.1f} km/s, off={p0[7]:.3e}")

    # Flatten for curve_fit, and restrict to a fitting sub-box around the peak
    # to keep the fit fast and robust; include plenty of off-source voxels too.
    # Use a half-width of 40 pixels spatially and all spectral channels.
    box = 40
    y_lo, y_hi = max(0, iy - box), min(ny, iy + box + 1)
    x_lo, x_hi = max(0, ix - box), min(nx, ix + box + 1)
    sub = data[:, y_lo:y_hi, x_lo:x_hi]
    ra_sub = ra2d[y_lo:y_hi, x_lo:x_hi]
    dec_sub = dec2d[y_lo:y_hi, x_lo:x_hi]
    print(f"  fitting subcube (v,y,x) = {sub.shape}")

    # Build flattened coordinate arrays, mask NaNs
    V, Y, X = np.meshgrid(vel, np.arange(sub.shape[1]), np.arange(sub.shape[2]), indexing='ij')
    ra_flat = ra_sub[Y, X].ravel()
    dec_flat = dec_sub[Y, X].ravel()
    vel_flat = V.ravel()
    data_flat = sub.ravel()
    finite = np.isfinite(data_flat)
    ra_flat = ra_flat[finite]
    dec_flat = dec_flat[finite]
    vel_flat = vel_flat[finite]
    data_flat = data_flat[finite]
    print(f"  fitting npts = {data_flat.size}")

    popt, pcov = curve_fit(
        gauss3d, (ra_flat, dec_flat, vel_flat), data_flat, p0=p0,
        sigma=np.full_like(data_flat, rms), absolute_sigma=True,
        maxfev=20000,
    )
    amp, x0, y0, sx, sy, v0, sv, offset = popt
    print("=== Best-fit parameters ===")
    print(f"  A    = {amp:.4e} Jy/beam")
    print(f"  RA   = {x0:.7f} deg")
    print(f"  Dec  = {y0:.7f} deg")
    print(f"  sx   = {abs(sx):.4f} arcsec  (FWHM = {2.355*abs(sx):.3f}\")")
    print(f"  sy   = {abs(sy):.4f} arcsec  (FWHM = {2.355*abs(sy):.3f}\")")
    print(f"  v0   = {v0:.3f} km/s")
    print(f"  sv   = {abs(sv):.3f} km/s    (FWHM = {2.355*abs(sv):.2f} km/s)")
    print(f"  off  = {offset:.3e} Jy/beam")

    # Build model on the full SO32 cube grid
    params = (amp, x0, y0, sx, sy, v0, sv, offset)
    model = gauss3d_on_grid(params, ra2d, dec2d, vel).astype(np.float32)
    residual = data.astype(np.float32) - model

    # Preserve original header for output (single Stokes axis collapsed)
    out_header = header.copy()
    # Make sure NAXIS matches 3D output
    for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
        if k in out_header:
            del out_header[k]
    out_header["NAXIS"] = 3

    model_path = os.path.join(BASE, "SO32_model3D_gaussian.fits")
    resid_path = os.path.join(BASE, "SO32_residual3D_gaussian.fits")
    fits.writeto(model_path, model, out_header, overwrite=True)
    fits.writeto(resid_path, residual, out_header, overwrite=True)
    print(f"Wrote {model_path}")
    print(f"Wrote {resid_path}")

    # Quick residual diagnostics
    res_rms_on_box = np.nanstd(residual[:, y_lo:y_hi, x_lo:x_hi])
    res_rms_off = np.nanstd(residual[off_signal])
    print(f"  residual rms (fit box, all channels): {res_rms_on_box:.3e}")
    print(f"  residual rms (off-signal channels): {res_rms_off:.3e}")
    print(f"  input off-signal rms: {rms:.3e}")
    print(f"  ratio resid_box / off_rms = {res_rms_on_box / rms:.3f}")

    return params


def build_template_on_cube(cube_path, params, output_path=None):
    """Evaluate the analytic 3D Gaussian model on a target cube's spatial+
    spectral grid. Template is normalized so peak=1, so the matched-filter
    flux estimator sum(d*t)/sum(t^2) returns the peak intensity of a
    MUBLO-shaped emitter in the target cube's brightness units.

    Returns (template, vel, ok). ok=False if the cube axis isn't velocity.
    """
    hdr = fits.getheader(cube_path)
    ctype3 = str(hdr.get("CTYPE3", "")).upper()
    if "FREQ" in ctype3:
        return None, None, False
    nv = hdr["NAXIS3"]
    vel = spectral_axis_kms(hdr, nv)
    wcs_cel = WCS(hdr).celestial
    ny = hdr["NAXIS2"]
    nx = hdr["NAXIS1"]
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    ra2d, dec2d = wcs_cel.wcs_pix2world(x_idx, y_idx, 0)
    amp, x0, y0, sx, sy, v0, sv, _offset = params
    template = gauss3d_on_grid((1.0, x0, y0, sx, sy, v0, sv, 0.0),
                                ra2d, dec2d, vel).astype(np.float32)
    if output_path:
        out_hdr = hdr.copy()
        for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
            if k in out_hdr:
                del out_hdr[k]
        out_hdr["NAXIS"] = 3
        fits.writeto(output_path, template, out_hdr, overwrite=True)
    return template, vel, True


def matched_filter_spectrum(cube_data, template, off_mask=None):
    """Proper matched-filter flux estimator per output channel, with the
    template spectrally shifted to that channel.

    For output channel i, shift the template along the spectral axis by
    (i - i_center) and compute sum(d*t) / sum(t^2).

    i_center is the template's signal-weighted center channel.
    """
    nv, ny, nx = cube_data.shape
    assert template.shape == cube_data.shape, \
        f"shape mismatch: cube {cube_data.shape} vs template {template.shape}"

    # Zero out template where it is below 0.1% of its peak to avoid numerical
    # junk dominating the denominator far from the line center.
    peak = np.nanmax(template)
    t = np.where(template > 1e-3 * peak, template, 0.0)
    d = np.where(np.isfinite(cube_data), cube_data, 0.0)

    # Center channel = peak channel of the integrated-spatial template
    template_spec = t.sum(axis=(1, 2))
    i_center = int(np.argmax(template_spec))

    # t_squared_sum: for each shift s, compute sum(t^2) where t has been
    # translated by s channels. Since the template is just shifted (circularly
    # we want to avoid), we will compute directly via roll + sum, skipping
    # wrap-around by masking.
    out = np.zeros(nv, dtype=np.float64)
    denom = np.zeros(nv, dtype=np.float64)

    # Flatten spatial dims for speed
    t_flat = t.reshape(nv, -1)
    d_flat = d.reshape(nv, -1)

    # Precompute cumulative arrays are nontrivial for shifts; do direct loop
    for i in range(nv):
        shift = i - i_center
        if shift > 0:
            t_shift = t_flat[:-shift] if shift != 0 else t_flat
            d_shift = d_flat[shift:]
        elif shift < 0:
            t_shift = t_flat[-shift:]
            d_shift = d_flat[:shift] if shift != 0 else d_flat
        else:
            t_shift = t_flat
            d_shift = d_flat
        if t_shift.size == 0:
            continue
        num = np.sum(d_shift * t_shift)
        den = np.sum(t_shift * t_shift)
        if den > 0:
            out[i] = num / den
            denom[i] = den

    # Estimate noise from off-signal channels
    if off_mask is not None and off_mask.sum() > 3:
        sigma = np.std(out[off_mask])
    else:
        sigma = np.std(out)
    return out, sigma, i_center


def _safe_hdr_val(v):
    """Return a value safe to put in a FITS header (no NaN)."""
    if v is None:
        return 0.0
    try:
        if not np.isfinite(v):
            return 0.0
    except Exception:
        pass
    return float(v)


def run_on_cube(cube_path, params, label):
    print(f"\n=== Matched filter on {os.path.basename(cube_path)} ({label}) ===")
    try:
        hdr = fits.getheader(cube_path)
    except Exception as e:
        print(f"  CANNOT READ header: {e}")
        return
    ctype3 = str(hdr.get("CTYPE3", "")).upper()
    if "FREQ" in ctype3:
        print(f"  SKIP (CTYPE3={ctype3} — broadband cube; run per-line cubes instead)")
        return

    # Build template on this cube's grid (also returns velocity axis in km/s)
    tpl_path = os.path.join(
        os.path.dirname(cube_path),
        os.path.basename(cube_path).replace('.fits', '.gauss3d_template.fits')
    )
    template, vel, ok = build_template_on_cube(cube_path, params, output_path=tpl_path)
    if not ok:
        print("  SKIP (not a velocity axis)")
        return
    nv = template.shape[0]
    print(f"  nchan={nv}, dv={vel[1]-vel[0]:.3f} km/s, v range {vel.min():.1f}..{vel.max():.1f}")

    # Load cube data (skip degenerate Stokes)
    data = fits.getdata(cube_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    if data.shape != template.shape:
        print(f"  shape mismatch cube vs template: {data.shape} vs {template.shape}")
        return

    off_mask = np.abs(vel - params[5]) > 250.0
    if off_mask.sum() < 5:
        off_mask = np.abs(vel - params[5]) > 150.0
    if off_mask.sum() < 5:
        # fallback: outermost 20% of channels
        n_edge = max(2, nv // 10)
        off_mask = np.zeros(nv, dtype=bool)
        off_mask[:n_edge] = True
        off_mask[-n_edge:] = True

    spec, sigma, i_center = matched_filter_spectrum(data, template, off_mask=off_mask)
    v_at_peak = vel[i_center]
    snr_at_peak = spec[i_center] / sigma if sigma > 0 else np.nan
    print(f"  template peak at v={v_at_peak:.2f} km/s (channel {i_center})")
    print(f"  flux estimator at peak: {spec[i_center]:.3e} (same units as cube)")
    print(f"  sigma (off-signal): {sigma:.3e}")
    print(f"  SNR at peak: {snr_at_peak:.2f}")
    snr_spec = spec / sigma if sigma > 0 else spec * np.nan
    imax = int(np.nanargmax(np.where(np.isfinite(snr_spec), snr_spec, -np.inf)))
    print(f"  peak SNR anywhere: {snr_spec[imax]:.2f} at v={vel[imax]:.2f} km/s")

    out_path = os.path.join(
        os.path.dirname(cube_path),
        os.path.basename(cube_path).replace('.fits', '.gauss3d_mf_spectrum.fits')
    )
    col = fits.Column(name='flux', format='D', array=spec)
    col_v = fits.Column(name='velocity_kms', format='D', array=vel)
    col_snr = fits.Column(name='snr', format='D', array=snr_spec)
    hdu = fits.BinTableHDU.from_columns([col_v, col, col_snr])
    hdu.header['SIGMA'] = (_safe_hdr_val(sigma), 'off-signal stdev of flux spectrum')
    hdu.header['VPEAK'] = (_safe_hdr_val(v_at_peak), 'template peak velocity, km/s')
    hdu.header['SNRPEAK'] = (_safe_hdr_val(snr_at_peak), 'SNR at template peak channel')
    hdu.header['SNRMAX'] = (_safe_hdr_val(snr_spec[imax]), 'peak SNR in spectrum')
    hdu.header['VSNRMAX'] = (_safe_hdr_val(vel[imax]), 'velocity of peak SNR, km/s')
    hdu.writeto(out_path, overwrite=True)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    params = fit_so32()

    # Per-line B3 MUBLO cubes (all in VRAD, already spatially cut around MUBLO)
    b3_line_cubes = sorted(glob.glob(os.path.join(BASE, "b3.spw*.cube.I.pbcor.mublo.*.fits")))
    b3_line_cubes = [c for c in b3_line_cubes if 'masked' not in c and 'mom' not in c
                     and 'template' not in c and 'mf_spectrum' not in c]
    for cp in b3_line_cubes:
        run_on_cube(cp, params, "B3 per-line")

    # Per-line B7 MUBLO cubes — includes 12CO(3-2), 13CO(3-2), CS(7-6), SO(8_8-7_7)
    b7_line_cubes = sorted(glob.glob(os.path.join(BASE, "b7",
                                                   "*.cube.I.selfcal.pbcor.mublo.*.fits")))
    b7_line_cubes = [c for c in b7_line_cubes if 'masked' not in c and 'mom' not in c
                     and 'template' not in c and 'mf_spectrum' not in c]
    for cp in b7_line_cubes:
        run_on_cube(cp, params, "B7 per-line")
