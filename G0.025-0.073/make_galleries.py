"""
PNG gallery for MUBLO analysis:
 - matched-filter spectra: ONLY for the 8 full spectral windows
   (4 B3 spws + 4 B7 spws), annotated with all catalog lines at
   MUBLO's v_LSR.
 - per-line moment maps (moment 0, peak, moment 1, peak-v), for every
   per-line MUBLO cutout cube.
 - MF-weighted moment maps (moment 0 and moment 1 weighted by the
   SO32-derived Gaussian velocity template, so noise channels are
   down-weighted).

Outputs under ./galleries/.
"""
import glob
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.io import fits
from astropy.wcs import WCS

from fit3d_gaussian import matched_filter_freq_cube  # FREQ-axis MF

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(BASE, "galleries")
os.makedirs(GAL, exist_ok=True)

# Best-fit 3D Gaussian parameters (from fit3d_gaussian.py output)
PARAMS = (2.1432e-03,           # amp (unused for moment weighting)
          266.4905821,          # x0 deg
          -28.9529931,          # y0 deg
          0.3545,               # sx arcsec
          0.3620,               # sy arcsec
          38.962,               # v0 km/s
          79.813,               # sv km/s (sigma)
          0.0)                  # offset
_, RA_MUBLO, DEC_MUBLO, SX, SY, V0, SIGMA_V, _ = PARAMS
FWHM_V = 2.355 * SIGMA_V
V_LO = V0 - 1.5 * FWHM_V
V_HI = V0 + 1.5 * FWHM_V

C_KMS = 299792.458

# Load bow-shock 1D velocity profile if present (from combined SO+CS fit)
BOW_PROFILE = None
_bow_path = os.path.join(BASE, "SOCS_bowshock_profile.fits")
if os.path.exists(_bow_path):
    with fits.open(_bow_path) as _hdul:
        _tbl = _hdul[1].data
        _phdr = _hdul[1].header
    _v = np.array(_tbl["velocity_kms"])
    _p = np.array(_tbl["profile"])
    # np.interp requires ascending xp; the SO cube's velocity axis is
    # descending, so flip if needed.
    if _v[0] > _v[-1]:
        _v = _v[::-1]
        _p = _p[::-1]
    BOW_PROFILE = {
        "velocity_kms": _v,
        "profile": _p,
        "fwhm": _phdr.get("FWHM", np.nan),
        "centroid": _phdr.get("VCENT", V0),
    }
    print(f"[gallery] loaded bow-shock profile: FWHM={BOW_PROFILE['fwhm']:.1f} km/s, "
          f"centroid={BOW_PROFILE['centroid']:.1f} km/s, "
          f"n_nonzero={(BOW_PROFILE['profile']>1e-3).sum()}")

# Line catalog: rest frequencies in GHz (any line present at MUBLO's v0
# will be annotated if it falls inside a cube's FREQ range). Feel free to
# extend this list.
LINE_CATALOG = [
    ("SO2 2(2,1)-1(1,1)",  86.09395,  "SO2"),
    ("H13CN 1-0",          86.33992,  "H13CN"),
    ("H13CO+ 1-0",         86.75430,  "H13CO+"),
    ("SiO 2-1",            86.84696,  "SiO"),
    ("HNCO 4(0,4)-3(0,3)", 87.92524,  "HNCO"),
    ("HCN 1-0",            88.63160,  "HCN"),
    ("HCO+ 1-0",           89.18852,  "HCO+"),
    ("HNC 1-0",            90.66356,  "HNC"),
    ("OCS 8-7",            97.30121,  "OCS"),
    ("SO 2(7,3,5)-8(2,6)", 97.70234,  "SO27"),
    ("CS 2-1",             97.98095,  "CS(2-1)"),
    ("H40a",               99.02295,  "H40a"),
    ("SO 3(2)-2(1)",       99.29987,  "SO(3-2)"),
    # Band 7 lines of interest
    ("13CO 3-2",          330.58796,  "13CO(3-2)"),
    ("C18O 3-2",          329.33055,  "C18O(3-2)"),
    ("C17O 3-2",          337.06108,  "C17O(3-2)"),
    ("CS 7-6",            342.88285,  "CS(7-6)"),
    ("SO 8(8)-7(7)",      344.31061,  "SO(8-7)"),
    ("12CO 3-2",          345.79600,  "12CO(3-2)"),
    ("H13CN 4-3",         345.33977,  "H13CN(4-3)"),
    ("HCN 4-3",           354.50547,  "HCN(4-3)"),
    ("HCO+ 4-3",          356.73424,  "HCO+(4-3)"),
]


# ---------- shared helpers ----------
def spectral_axis_kms(header, nchan=None):
    if nchan is None:
        nchan = header["NAXIS3"]
    crpix = header["CRPIX3"]
    crval = header["CRVAL3"]
    cdelt = header["CDELT3"]
    unit = str(header.get("CUNIT3", "")).strip().lower()
    scale = 1e-3 if unit in ("m/s", "m s-1", "m s^-1") else 1.0
    return (crval + cdelt * (np.arange(nchan) + 1 - crpix)) * scale


def load_cube_3d(path):
    d = fits.getdata(path).astype(np.float32)
    while d.ndim > 3:
        d = d[0]
    return d


def safe_hdr(h, key, default=None):
    v = h.get(key, default)
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return default
    except Exception:
        return default
    return v


# ---------- full-window matched-filter spectra ----------
def plot_fullwindow_spectrum(cube_path, outdir):
    """Run the FREQ-axis MF and plot SNR vs frequency, with all catalog
    lines marked at MUBLO v_LSR = V0."""
    result = matched_filter_freq_cube(cube_path, PARAMS)
    if result is None:
        return None
    freq_ghz = result["freq_hz"] / 1e9
    snr = result["snr"]
    flux = result["flux"]
    sigma = result["sigma"]

    f_lo = float(np.min(freq_ghz))
    f_hi = float(np.max(freq_ghz))

    # Shift each rest line to its expected observed frequency at V0
    doppler = 1.0 - V0 / C_KMS  # obs/rest
    lines_in_range = []
    for (lbl, rest_ghz, short) in LINE_CATALOG:
        obs = rest_ghz * doppler
        if f_lo <= obs <= f_hi:
            lines_in_range.append((lbl, short, obs))

    base = os.path.basename(cube_path).replace(".fits", "")
    fig, (ax_f, ax_s) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                     gridspec_kw=dict(hspace=0.04))
    ax_f.plot(freq_ghz, flux, color="black", lw=0.6, drawstyle="steps-mid")
    ax_f.axhline(0, color="gray", lw=0.4)
    ax_f.set_ylabel("MF flux")
    ax_f.set_title(f"{base}\nfull-window matched filter (σ={sigma:.2e}, "
                   f"N_lines_in_range={len(lines_in_range)})",
                   fontsize=10)

    ax_s.plot(freq_ghz, snr, color="navy", lw=0.6, drawstyle="steps-mid")
    ax_s.axhline(0, color="gray", lw=0.4)
    for thr in (3, 5, 10):
        ax_s.axhline(thr, color="orange", lw=0.5, ls=":", alpha=0.7)
        ax_s.axhline(-thr, color="orange", lw=0.5, ls=":", alpha=0.7)
    ax_s.set_xlabel("frequency (GHz)")
    ax_s.set_ylabel("MF SNR")

    # line markers + labels
    ymin_f, ymax_f = ax_f.get_ylim()
    ymin_s, ymax_s = ax_s.get_ylim()
    for (lbl, short, obs) in lines_in_range:
        ax_f.axvline(obs, color="red", lw=0.6, alpha=0.7)
        ax_s.axvline(obs, color="red", lw=0.6, alpha=0.7)
        ax_f.text(obs, ymax_f, f" {short}", color="red", fontsize=8,
                  rotation=90, va="top", ha="left")
    ax_f.set_xlim(f_lo, f_hi)

    outname = os.path.join(outdir, base + ".fullwindow_mf.png")
    fig.savefig(outname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return outname


# ---------- moment maps (standard + MF-weighted) ----------
def plot_moment_maps(cube_path, outdir, v_lo=V_LO, v_hi=V_HI,
                     pos_ra=RA_MUBLO, pos_dec=DEC_MUBLO):
    """9-panel figure (3 rows):
     - row 1 (standard, fixed window): mom0, peak, mom1
     - row 2 (Gaussian MF weighting, FWHM ~188 km/s): MF-M0, MF-M0 SNR, MF-M1
     - row 3 (bow-shock velocity-profile weighting, narrower): bow-M0,
        bow-M0 SNR, bow-M1
    """
    hdr = fits.getheader(cube_path)
    if "FREQ" in str(hdr.get("CTYPE3", "")).upper():
        return None
    data = load_cube_3d(cube_path)
    nv, ny, nx = data.shape
    vel = spectral_axis_kms(hdr, nv)
    dv = abs(vel[1] - vel[0])

    sel = (vel >= v_lo) & (vel <= v_hi)
    if sel.sum() < 2:
        center = np.argmin(np.abs(vel - V0))
        half = max(2, int(0.75 * FWHM_V / dv))
        lo = max(0, center - half)
        hi = min(nv, center + half + 1)
        sel = np.zeros(nv, dtype=bool)
        sel[lo:hi] = True
        if sel.sum() < 2:
            return None
    sub = data[sel]
    sub_vel = vel[sel]
    n_sel = int(sub.shape[0])

    # --- standard moment 0 (unweighted window integration)
    total = np.nansum(sub, axis=0)
    m0 = total * dv
    peak = np.nanmax(sub, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.nansum(sub * sub_vel[:, None, None], axis=0) / total
    peakarg = np.argmax(np.where(np.isfinite(sub), sub, -np.inf), axis=0)
    vpeak = sub_vel[peakarg]

    off = data[~sel] if (~sel).sum() > 3 else data
    rms = np.nanstd(off)
    m1_mask = total < 3 * rms * np.sqrt(n_sel)
    m1_plot = np.where(m1_mask, np.nan, m1)
    vpeak_plot = np.where(m1_mask, np.nan, vpeak)

    # --- MF-weighted moments (1D velocity Gaussian weight over the full cube)
    w_v = np.exp(-0.5 * ((vel - V0) / SIGMA_V) ** 2).astype(np.float32)  # peak=1
    # Clip weights below 0.1% of peak to avoid numerical noise buildup
    w_v = np.where(w_v > 1e-3, w_v, 0.0)
    data_finite = np.where(np.isfinite(data), data, 0.0)
    # moment 0 weighted: sum_v d(x,y,v) * w(v) * dv
    m0_mf = np.einsum("i,ijk->jk", w_v, data_finite) * dv
    # denominator for mom1: sum_v d * w (no *dv, cancels out in ratio)
    num_mf = np.einsum("i,ijk->jk", w_v * vel.astype(np.float32), data_finite)
    den_mf = np.einsum("i,ijk->jk", w_v, data_finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1_mf = num_mf / den_mf
    # noise estimate for MF m0 using same weighting on off-signal channels
    off_mask = np.abs(vel - V0) > 300.0
    if off_mask.sum() < 5:
        off_mask = np.abs(vel - V0) > 200.0
    if off_mask.sum() >= 5:
        off_rms = np.nanstd(data_finite[off_mask])
        expected_mf_sigma = off_rms * np.sqrt((w_v ** 2).sum()) * dv
    else:
        expected_mf_sigma = np.nan
    snr_map = m0_mf / expected_mf_sigma if np.isfinite(expected_mf_sigma) else np.full_like(m0_mf, np.nan)
    m1_mf_mask = (np.abs(m0_mf) < 3 * expected_mf_sigma) if np.isfinite(expected_mf_sigma) else np.ones_like(m0_mf, dtype=bool)
    m1_mf_plot = np.where(m1_mf_mask, np.nan, m1_mf)

    # --- spatial cutout around MUBLO
    wcs_cel = WCS(hdr).celestial
    cdelt1 = abs(hdr["CDELT1"])
    halfpix = int(round(1.5 / 3600.0 / cdelt1))
    x_src, y_src = wcs_cel.wcs_world2pix(pos_ra, pos_dec, 0)
    ix = int(round(float(x_src)))
    iy = int(round(float(y_src)))
    x_lo = max(0, ix - halfpix)
    x_hi = min(nx, ix + halfpix + 1)
    y_lo = max(0, iy - halfpix)
    y_hi = min(ny, iy + halfpix + 1)

    def cut(a):
        return a[y_lo:y_hi, x_lo:x_hi]

    m0c = cut(m0); peakc = cut(peak); m1c = cut(m1_plot); vpeakc = cut(vpeak_plot)
    m0mfc = cut(m0_mf); m1mfc = cut(m1_mf_plot); snrc = cut(snr_map)

    # --- Bow-shock weighted moments (narrower 1D profile from SO+CS residual)
    if BOW_PROFILE is not None:
        bow_w_v = np.interp(vel, BOW_PROFILE["velocity_kms"], BOW_PROFILE["profile"],
                            left=0.0, right=0.0).astype(np.float32)
        bow_w_v = np.where(bow_w_v > 1e-3, bow_w_v, 0.0)
    else:
        # Fallback: narrow Gaussian centered at V0+25 with FWHM~100 km/s
        bow_w_v = np.exp(-0.5 * ((vel - (V0 + 25.0)) / (100.0 / 2.355)) ** 2).astype(np.float32)

    m0_bow = np.einsum("i,ijk->jk", bow_w_v, data_finite) * dv
    num_bow = np.einsum("i,ijk->jk", bow_w_v * vel.astype(np.float32), data_finite)
    den_bow = np.einsum("i,ijk->jk", bow_w_v, data_finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1_bow = num_bow / den_bow
    if off_mask.sum() >= 5:
        expected_bow_sigma = off_rms * np.sqrt((bow_w_v ** 2).sum()) * dv
    else:
        expected_bow_sigma = np.nan
    if np.isfinite(expected_bow_sigma):
        snr_bow = m0_bow / expected_bow_sigma
        m1_bow_mask = np.abs(m0_bow) < 3 * expected_bow_sigma
    else:
        snr_bow = np.full_like(m0_bow, np.nan)
        m1_bow_mask = np.ones_like(m0_bow, dtype=bool)
    m1_bow_plot = np.where(m1_bow_mask, np.nan, m1_bow)
    m0bowc = cut(m0_bow); snrbowc = cut(snr_bow); m1bowc = cut(m1_bow_plot)

    vmin_m1 = V0 - FWHM_V / 2
    vmax_m1 = V0 + FWHM_V / 2
    # Tighter v range for bow-shock m1 panel based on the profile
    if BOW_PROFILE is not None and np.isfinite(BOW_PROFILE["fwhm"]):
        bow_cent = float(BOW_PROFILE["centroid"])
        bow_fwhm = float(BOW_PROFILE["fwhm"]) if BOW_PROFILE["fwhm"] > 0 else 100.0
        vmin_bow = bow_cent - 1.0 * bow_fwhm
        vmax_bow = bow_cent + 1.0 * bow_fwhm
    else:
        vmin_bow, vmax_bow = V0 - 60, V0 + 90

    base = os.path.basename(cube_path).replace(".fits", "")
    fig, axes = plt.subplots(3, 3, figsize=(13, 12.5))
    # Top row: standard moment products
    im = axes[0, 0].imshow(m0c, origin="lower", cmap="viridis",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[0, 0].set_title("Moment 0 (window)", fontsize=10)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(peakc, origin="lower", cmap="magma",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[0, 1].set_title("Peak (Jy/beam)", fontsize=10)
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(m1c, origin="lower", cmap="RdBu_r",
                           vmin=vmin_m1, vmax=vmax_m1,
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[0, 2].set_title(f"Moment 1 (km/s)\n[{vmin_m1:.0f},{vmax_m1:.0f}]", fontsize=10)
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Middle row: Gaussian MF-weighted (FWHM~188 km/s)
    im = axes[1, 0].imshow(m0mfc, origin="lower", cmap="viridis",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[1, 0].set_title("Gauss-MF Moment 0", fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(snrc, origin="lower", cmap="inferno",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[1, 1].set_title(f"Gauss-MF SNR  (σ={expected_mf_sigma:.2e})", fontsize=10)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(m1mfc, origin="lower", cmap="RdBu_r",
                           vmin=vmin_m1, vmax=vmax_m1,
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[1, 2].set_title("Gauss-MF Moment 1", fontsize=10)
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Bottom row: bow-shock velocity-profile weighted (narrower)
    if BOW_PROFILE is not None:
        bow_fwhm = float(BOW_PROFILE["fwhm"])
        bow_cent = float(BOW_PROFILE["centroid"])
        bow_lbl = f"bow-shock (FWHM~{bow_fwhm:.0f}, v̄={bow_cent:.0f})"
    else:
        bow_lbl = "bow-shock (fallback narrow Gaussian)"
    im = axes[2, 0].imshow(m0bowc, origin="lower", cmap="viridis",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[2, 0].set_title(f"bow-shock Moment 0", fontsize=10)
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im = axes[2, 1].imshow(snrbowc, origin="lower", cmap="inferno",
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[2, 1].set_title(f"bow-shock SNR  (σ={expected_bow_sigma:.2e})", fontsize=10)
    plt.colorbar(im, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im = axes[2, 2].imshow(m1bowc, origin="lower", cmap="RdBu_r",
                           vmin=vmin_bow, vmax=vmax_bow,
                           extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
    axes[2, 2].set_title(f"bow-shock Moment 1\n[{vmin_bow:.0f},{vmax_bow:.0f}]",
                          fontsize=10)
    plt.colorbar(im, ax=axes[2, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.plot(x_src, y_src, "x", color="cyan", ms=7, mew=1.3)
        ax.set_xlim(x_lo - 0.5, x_hi - 0.5)
        ax.set_ylim(y_lo - 0.5, y_hi - 0.5)
        ax.set_xticks([]); ax.set_yticks([])

    # beam ellipse in top-left
    bmaj = safe_hdr(hdr, "BMAJ"); bmin = safe_hdr(hdr, "BMIN"); bpa = safe_hdr(hdr, "BPA", 0.0)
    if bmaj and bmin:
        ex = x_lo + 0.1 * (x_hi - x_lo)
        ey = y_lo + 0.1 * (y_hi - y_lo)
        beam = Ellipse((ex, ey), width=bmin / cdelt1, height=bmaj / cdelt1,
                       angle=bpa, fill=False, edgecolor="white", lw=1.2)
        axes[0, 0].add_patch(beam)

    fig.suptitle(f"{base}\nstd window: [{v_lo:.0f},{v_hi:.0f}] km/s (n={n_sel})   "
                 f"Gauss weight: FWHM={FWHM_V:.0f}   "
                 f"bow-shock weight: {bow_lbl}",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outname = os.path.join(outdir, base + ".moments.png")
    fig.savefig(outname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return outname


def plot_resid_fullwin(mf_path, cube_path, outdir):
    """Plot the residual-template matched-filter spectrum for a full
    spectral window (FREQ axis), with catalog lines annotated at their
    MUBLO-shifted observed frequencies."""
    with fits.open(mf_path) as hdul:
        tbl = hdul[1].data
        hdr = hdul[1].header
    freq_hz = tbl["frequency_hz"]
    flux = tbl["flux"]
    snr = tbl["snr"]
    sigma = safe_hdr(hdr, "SIGMA", 0.0)
    snr_max = safe_hdr(hdr, "SNRMAX", np.nan)
    freq_ghz = freq_hz / 1e9

    f_lo = float(np.min(freq_ghz)); f_hi = float(np.max(freq_ghz))
    doppler = 1.0 - V0 / C_KMS
    lines_in_range = []
    for (lbl, rest_ghz, short) in LINE_CATALOG:
        obs = rest_ghz * doppler
        if f_lo <= obs <= f_hi:
            lines_in_range.append((lbl, short, obs))

    base = os.path.basename(cube_path).replace(".fits", "")
    fig, (ax_f, ax_s) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                     gridspec_kw=dict(hspace=0.04))
    ax_f.plot(freq_ghz, flux, color="black", lw=0.6, drawstyle="steps-mid")
    ax_f.axhline(0, color="gray", lw=0.4)
    ax_f.set_ylabel("residMF flux")
    ax_f.set_title(f"{base}   [residual template, bow-shock]\n"
                   f"σ={sigma:.2e}  peak SNR={snr_max:.1f}   "
                   f"N_lines_in_range={len(lines_in_range)}",
                   fontsize=10)

    ax_s.plot(freq_ghz, snr, color="navy", lw=0.6, drawstyle="steps-mid")
    ax_s.axhline(0, color="gray", lw=0.4)
    for thr in (3, 5, 10):
        ax_s.axhline(thr, color="orange", lw=0.5, ls=":", alpha=0.7)
        ax_s.axhline(-thr, color="orange", lw=0.5, ls=":", alpha=0.7)
    ax_s.set_xlabel("frequency (GHz)")
    ax_s.set_ylabel("residMF SNR")
    ymin_f, ymax_f = ax_f.get_ylim()
    for (lbl, short, obs) in lines_in_range:
        ax_f.axvline(obs, color="red", lw=0.6, alpha=0.7)
        ax_s.axvline(obs, color="red", lw=0.6, alpha=0.7)
        ax_f.text(obs, ymax_f, f" {short}", color="red", fontsize=8,
                  rotation=90, va="top", ha="left")
    ax_f.set_xlim(f_lo, f_hi)

    outname = os.path.join(outdir, base + ".residfullwin_mf.png")
    fig.savefig(outname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return outname


def write_index_html(spectrum_pngs, moment_pngs, residmf_pngs, mask_diag_pngs, outdir,
                      rgb_png=None, rgb_bow_png=None, dmr_png=None, radial_png=None,
                      band9_pngs=None):
    index = os.path.join(outdir, "index.html")
    with open(index, "w") as fh:
        fh.write("<html><head><title>MUBLO gallery</title>")
        fh.write("<style>body{font-family:sans-serif;background:#111;color:#eee;} "
                 "img{max-width:96%;margin:4px 0;} "
                 "h2{border-bottom:1px solid #555;} "
                 "section{margin-bottom:32px;}</style></head><body>")
        fh.write("<h1>MUBLO gallery</h1>")
        fh.write(f"<p>v₀ = {V0:.2f} km/s &nbsp; FWHM_v = {FWHM_V:.2f} km/s &nbsp; "
                 f"window = [{V_LO:.0f},{V_HI:.0f}] km/s</p>")
        if rgb_png and os.path.exists(rgb_png):
            fh.write("<h2>RGB composite (bright lines) + bow-shock whitening + "
                     "faint-line 4σ/5σ contours</h2>")
            rel = os.path.basename(rgb_png)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if rgb_bow_png and os.path.exists(rgb_bow_png):
            fh.write("<h2>RGB composite — bow-shock emphasis (greyscale = SO Gaussian "
                     "model, RGB = residual M0 of SO/CS/SO₂)</h2>")
            rel = os.path.basename(rgb_bow_png)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if dmr_png and os.path.exists(dmr_png):
            fh.write("<h2>SO(3-2) and CS(2-1): data, Gaussian model, residual M0</h2>")
            rel = os.path.basename(dmr_png)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if radial_png and os.path.exists(radial_png):
            fh.write("<h2>Radial profiles — Gaussian vs bow-shock weighting, "
                     "pre- and post-subtraction</h2>")
            rel = os.path.basename(radial_png)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if mask_diag_pngs:
            fh.write("<h2>Residual-template mask construction &amp; bow-shock profile</h2>")
            for p in mask_diag_pngs:
                rel = os.path.basename(p)
                fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        fh.write("<h2>Full-window matched-filter spectra (8 SPWs, Gaussian template)</h2>")
        for p in spectrum_pngs:
            rel = os.path.basename(p)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if band9_pngs:
            fh.write("<h2>Band 9 matched-filter spectra (Gaussian template)</h2>")
            for p in band9_pngs:
                rel = os.path.basename(p)
                fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        if residmf_pngs:
            fh.write("<h2>Residual-template matched-filter spectra (bow-shock)</h2>")
            for p in residmf_pngs:
                rel = os.path.basename(p)
                fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        fh.write("<h2>Per-line moment maps (standard + MF-weighted)</h2>")
        for p in moment_pngs:
            rel = os.path.basename(p)
            fh.write(f'<section><h3>{rel}</h3><img src="{rel}"/></section>')
        fh.write("</body></html>")
    return index


if __name__ == "__main__":
    # 1) Full-window spectra: 4 B3 + 4 B7 "10kms" cubes
    full_window_cubes = (
        sorted(glob.glob(os.path.join(BASE, "b3.spw*.cube.I.pbcor.10kms.fits")))
        + sorted(glob.glob(os.path.join(BASE, "b7", "*.cube.I.selfcal.pbcor.10kms.fits")))
    )
    print(f"Full-window cubes: {len(full_window_cubes)}")
    spectrum_pngs = []
    for cp in full_window_cubes:
        try:
            out = plot_fullwindow_spectrum(cp, GAL)
            if out:
                spectrum_pngs.append(out)
                print(f"  spectrum -> {os.path.basename(out)}")
        except Exception as e:
            print(f"  spectrum FAILED for {cp}: {e}")

    # 2) Per-line moment maps for all per-line cubes
    line_cubes = []
    for pat in [
        os.path.join(BASE, "b3.spw*.cube.I.pbcor.mublo.*.fits"),
        os.path.join(BASE, "b7", "*.cube.I.selfcal.pbcor.mublo.*.fits"),
        os.path.join(BASE, "b9.spw*.cube.I.selfcal.pbcor.mublo.*.fits"),
    ]:
        for p in sorted(glob.glob(pat)):
            bn = os.path.basename(p)
            if any(x in bn for x in (
                "masked", ".mom", "template", "mf_spectrum",
                "fullwin_mf", "model3D", "residual3D", "residtpl",
                "resid_mf", "solocked",
            )):
                continue
            line_cubes.append(p)
    print(f"Per-line cubes: {len(line_cubes)}")
    moment_pngs = []
    for cube in line_cubes:
        try:
            out = plot_moment_maps(cube, GAL)
            if out:
                moment_pngs.append(out)
                print(f"  moments -> {os.path.basename(out)}")
        except Exception as e:
            print(f"  moments FAILED for {cube}: {e}")

    # Also moments of model and residual cubes (both SO and CS) for inspection
    for extra in [
        os.path.join(BASE, "SO32_model3D_gaussian.fits"),
        os.path.join(BASE, "SO32_residual3D_gaussian.fits"),
        os.path.join(BASE, "CS21_model3D_gaussian_solocked.fits"),
        os.path.join(BASE, "CS21_residual3D_gaussian_solocked.fits"),
    ]:
        if os.path.exists(extra):
            try:
                out = plot_moment_maps(extra, GAL)
                if out:
                    moment_pngs.append(out)
                    print(f"  moments -> {os.path.basename(out)}")
            except Exception as e:
                print(f"  moments FAILED for {extra}: {e}")

    # 3) Residual-template matched-filter spectra — ONLY for the 8
    # full spectral windows (same as the Gaussian-template section).
    residmf_pngs = []
    for cube in full_window_cubes:
        mf_path = cube.replace(".fits", ".resid_fullwin_mf.fits")
        if os.path.exists(mf_path):
            try:
                out = plot_resid_fullwin(mf_path, cube, GAL)
                residmf_pngs.append(out)
                print(f"  resid_mf -> {os.path.basename(out)}")
            except Exception as e:
                print(f"  resid_mf FAILED for {cube}: {e}")
        else:
            print(f"  no resid fullwin MF for {os.path.basename(cube)} — run residual_matched_filter.py first")

    # 4) Mask diagnostic PNGs + bow-shock 1D profile PNG
    mask_diag_pngs = sorted(glob.glob(os.path.join(GAL, "*mask_diagnostic*.png")))
    # include the bow-shock 1D profile (not a mask, but related)
    profile_png = os.path.join(GAL, "SOCS_bowshock_profile.png")
    if os.path.exists(profile_png):
        mask_diag_pngs.append(profile_png)

    # 5) RGB composites — rebuild so they stay in sync with template/profile.
    try:
        import subprocess
        subprocess.run(["python", os.path.join(BASE, "make_rgb_overlay.py")],
                        check=True)
    except Exception as e:
        print(f"  RGB overlay FAILED: {e}")
    try:
        subprocess.run(["python", os.path.join(BASE, "make_rgb_bowshock.py")],
                        check=True)
    except Exception as e:
        print(f"  RGB bow-shock-emphasis FAILED: {e}")
    try:
        subprocess.run(["python", os.path.join(BASE, "make_data_model_residual.py")],
                        check=True)
    except Exception as e:
        print(f"  data/model/residual FAILED: {e}")
    try:
        subprocess.run(["python", os.path.join(BASE, "make_radial_profiles.py")],
                        check=True)
    except Exception as e:
        print(f"  radial profiles FAILED: {e}")
    rgb_png = os.path.join(GAL, "MUBLO_RGB_bowshock.png")
    rgb_bow_png = os.path.join(GAL, "MUBLO_RGB_bowshock_emphasis.png")
    dmr_png = os.path.join(GAL, "MUBLO_data_model_residual.png")
    radial_png = os.path.join(GAL, "MUBLO_radial_profiles.png")

    # Clean up old per-line spectrum PNGs from earlier runs
    for old in glob.glob(os.path.join(GAL, "*.spectrum.png")):
        try:
            os.remove(old)
            print(f"  removed stale {os.path.basename(old)}")
        except Exception:
            pass

    # 7) Band 9 matched-filter spectra (if process_band9.py has been run)
    band9_pngs = sorted(glob.glob(os.path.join(GAL, "B9_matched_filter_*.png")))
    print(f"Band 9 matched-filter spectra: {len(band9_pngs)}")

    idx = write_index_html(spectrum_pngs, moment_pngs, residmf_pngs,
                           mask_diag_pngs, GAL, rgb_png=rgb_png,
                           rgb_bow_png=rgb_bow_png, dmr_png=dmr_png,
                           radial_png=radial_png, band9_pngs=band9_pngs)
    print(f"\nWrote {len(spectrum_pngs)} full-window spectra, {len(moment_pngs)} moment panels, "
          f"{len(residmf_pngs)} residual-MF spectra, {len(mask_diag_pngs)} mask diagnostics, "
          f"{len(band9_pngs)} Band 9 spectra")
    print(f"Open {idx}")
