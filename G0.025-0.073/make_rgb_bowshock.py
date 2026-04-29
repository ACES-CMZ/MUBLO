"""
Bow-shock-emphasis composite for MUBLO.

Greyscale base: SO 3D Gaussian *model* moment 0 (the smooth central
                Gaussian component, no residual structure).
Color layers (additive bow-shock): residual moment 0 of three sulfur-
                bearing lines, each fit with the SO shape locked and
                only amplitude+offset free, then SUBTRACTED to leave
                the bow-shock structure:
                  R = SO(3-2)   residual M0
                  G = CS(2-1)   residual M0
                  B = SO2(2_2,1-1_1,1) residual M0
Faint-line 4σ/5σ contours: SiO(2-1), HCN(1-0), CS(7-6).

Outputs galleries/MUBLO_RGB_bowshock_emphasis.png.
"""
import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(BASE, "galleries")
os.makedirs(GAL, exist_ok=True)

V0 = 38.962
FWHM_V = 187.96
SIGMA_V_SO = FWHM_V / 2.355
RA_MUBLO = 266.4905821
DEC_MUBLO = -28.9529931
SX = 0.3545; SY = 0.3620   # arcsec (SO fit)

# Cubes
SO32_CUBE   = os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits")
CS21_CUBE   = os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.CS21.fits")
SO2_CUBE    = os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SO2211.fits")
SIO_CUBE    = os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SiO21.fits")
HCN_CUBE    = os.path.join(BASE, "b3.spw27.cube.I.pbcor.mublo.HCN1-0.fits")
CS76_CUBE   = os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw29.cube.I.selfcal.pbcor.mublo.CS76.fits")

SO_MODEL    = os.path.join(BASE, "SO32_model3D_gaussian.fits")
SO_RESID    = os.path.join(BASE, "SO32_residual3D_gaussian.fits")
CS_RESID    = os.path.join(BASE, "CS21_residual3D_gaussian_solocked.fits")
SO2_RESID   = os.path.join(BASE, "SO2211_residual3D_gaussian_solocked.fits")
SO2_MODEL   = os.path.join(BASE, "SO2211_model3D_gaussian_solocked.fits")
BOW_PROFILE = os.path.join(BASE, "SOCS_bowshock_profile.fits")


def spectral_axis_kms(header, nchan=None):
    if nchan is None:
        nchan = header["NAXIS3"]
    crpix = header["CRPIX3"]; crval = header["CRVAL3"]; cdelt = header["CDELT3"]
    unit = str(header.get("CUNIT3", "")).strip().lower()
    scale = 1e-3 if unit in ("m/s", "m s-1", "m s^-1") else 1.0
    return (crval + cdelt * (np.arange(nchan) + 1 - crpix)) * scale


def load_cube(path):
    d = fits.getdata(path).astype(np.float32)
    while d.ndim > 3:
        d = d[0]
    h = fits.getheader(path)
    v = spectral_axis_kms(h, d.shape[0])
    return d, h, v


def fit_so2_shape_locked():
    """Amplitude+offset-only fit of SO2(2_2,1-1_1,1) with SO's spatial+
    spectral shape; produces model and residual cubes if not already done."""
    if os.path.exists(SO2_RESID) and os.path.exists(SO2_MODEL):
        return  # cached
    print("[so2 fit] amplitude-only refit of SO2 with SO32 shape")
    data = fits.getdata(SO2_CUBE).astype(np.float64)
    while data.ndim > 3:
        data = data[0]
    hdr = fits.getheader(SO2_CUBE)
    nv, ny, nx = data.shape
    vel = spectral_axis_kms(hdr, nv)
    wcs_cel = WCS(hdr).celestial
    yi, xi = np.mgrid[0:ny, 0:nx]
    ra2d, dec2d = wcs_cel.wcs_pix2world(xi, yi, 0)
    cosd = np.cos(np.deg2rad(DEC_MUBLO))
    dx = (ra2d - RA_MUBLO) * cosd * 3600.0
    dy = (dec2d - DEC_MUBLO) * 3600.0
    spatial_arg = 0.5 * (dx ** 2 / SX ** 2 + dy ** 2 / SY ** 2)
    spec_arg = 0.5 * ((vel - V0) / SIGMA_V_SO) ** 2
    unit_tpl = (np.exp(-spatial_arg)[None, :, :] *
                np.exp(-spec_arg)[:, None, None]).astype(np.float32)
    tpl_flat = unit_tpl.ravel().astype(np.float64)
    d_flat = np.where(np.isfinite(data), data, 0.0).ravel()
    ones = np.ones_like(tpl_flat)
    M = np.array([[np.dot(tpl_flat, tpl_flat), np.dot(tpl_flat, ones)],
                  [np.dot(ones, tpl_flat), np.dot(ones, ones)]])
    y = np.array([np.dot(tpl_flat, d_flat), np.dot(ones, d_flat)])
    A, B = np.linalg.solve(M, y)
    print(f"   SO2 peak amp = {A:.4e} Jy/beam, offset = {B:.3e}")
    model = (A * unit_tpl + B).astype(np.float32)
    residual = (data.astype(np.float32) - model)
    out_hdr = hdr.copy()
    for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
        if k in out_hdr:
            del out_hdr[k]
    out_hdr["NAXIS"] = 3
    out_hdr["HISTORY"] = "SO2 model: SO32 shape, amp+offset only refit"
    fits.writeto(SO2_MODEL, model, out_hdr, overwrite=True)
    fits.writeto(SO2_RESID, residual, out_hdr, overwrite=True)


def cube_mom0(cube_path, v_lo=V0 - 1.2 * FWHM_V, v_hi=V0 + 1.2 * FWHM_V,
              positive_only=False):
    d, h, v = load_cube(cube_path)
    sel = (v >= v_lo) & (v <= v_hi)
    dv = abs(v[1] - v[0])
    sub = d[sel]
    if positive_only:
        sub = np.where(sub > 0, sub, 0.0)
    return np.nansum(sub, axis=0) * dv, h


def load_bow_profile():
    with fits.open(BOW_PROFILE) as hd:
        t = hd[1].data
    v = np.array(t["velocity_kms"]); p = np.array(t["profile"])
    if v[0] > v[-1]:
        v = v[::-1]; p = p[::-1]
    return v, p


def bow_weighted_m0_snr(cube_path):
    d, h, v = load_cube(cube_path)
    dv = abs(v[1] - v[0])
    v_ref, p_ref = load_bow_profile()
    w = np.interp(v, v_ref, p_ref, left=0.0, right=0.0).astype(np.float32)
    w = np.where(w > 1e-3, w, 0.0)
    data_f = np.where(np.isfinite(d), d, 0.0)
    m0 = np.einsum("i,ijk->jk", w, data_f) * dv
    off = np.abs(v - V0) > 300.0
    if off.sum() < 5:
        off = np.abs(v - V0) > 200.0
    rms = np.nanstd(data_f[off]) if off.any() else np.nan
    sigma_m0 = rms * np.sqrt((w ** 2).sum()) * dv if np.isfinite(rms) and rms > 0 else np.nan
    snr = m0 / sigma_m0 if np.isfinite(sigma_m0) and sigma_m0 > 0 else np.full_like(m0, np.nan)
    return snr, h


def reproject_to_b3(arr_b7, hdr_b7, hdr_b3_ref):
    wcs_b3 = WCS(hdr_b3_ref).celestial
    wcs_b7 = WCS(hdr_b7).celestial
    ny_b3 = hdr_b3_ref["NAXIS2"]; nx_b3 = hdr_b3_ref["NAXIS1"]
    yi, xi = np.mgrid[0:ny_b3, 0:nx_b3]
    ra_b3, dec_b3 = wcs_b3.wcs_pix2world(xi, yi, 0)
    x_b7, y_b7 = wcs_b7.wcs_world2pix(ra_b3, dec_b3, 0)
    coords = np.vstack([y_b7.ravel(), x_b7.ravel()])
    return map_coordinates(arr_b7, coords, order=1, mode="constant",
                            cval=np.nan).reshape(ny_b3, nx_b3)


def norm_pos(arr, lo=0.0, hi=99.5):
    """Normalize to [0, 1] with positive-only clipping."""
    a = np.clip(arr, 0, None)
    vmax = np.nanpercentile(a, hi)
    if vmax <= 0:
        return np.zeros_like(a)
    return np.clip(a / vmax, 0, 1)


def main():
    fit_so2_shape_locked()

    # Greyscale base: Gaussian model M0 (so it shows the smooth Gaussian
    # component without any bow-shock contamination)
    grey_m0, h_so = cube_mom0(SO_MODEL)

    # Color layers: residual M0 (positive-only — bow-shock structure)
    R_m0, _ = cube_mom0(SO_RESID, positive_only=True)
    G_m0, _ = cube_mom0(CS_RESID, positive_only=True)
    B_m0, _ = cube_mom0(SO2_RESID, positive_only=True)

    # Faint-line bow-shock SNR maps
    snr_sio, h_sio = bow_weighted_m0_snr(SIO_CUBE)
    snr_hcn, h_hcn = bow_weighted_m0_snr(HCN_CUBE)
    snr_cs76_b7, h_cs76 = bow_weighted_m0_snr(CS76_CUBE)
    snr_cs76 = reproject_to_b3(snr_cs76_b7, h_cs76, h_so)

    # Spatial cutout around MUBLO
    wcs_b3 = WCS(h_so).celestial
    x_src, y_src = wcs_b3.wcs_world2pix(RA_MUBLO, DEC_MUBLO, 0)
    ix, iy = int(round(float(x_src))), int(round(float(y_src)))
    cdelt = abs(h_so["CDELT1"])
    half = int(round(1.5 / 3600.0 / cdelt))
    x_lo = max(0, ix - half); x_hi = min(h_so["NAXIS1"], ix + half + 1)
    y_lo = max(0, iy - half); y_hi = min(h_so["NAXIS2"], iy + half + 1)

    def cut(a):
        return a[y_lo:y_hi, x_lo:x_hi]

    grey = norm_pos(cut(grey_m0), hi=99.5)
    R = norm_pos(cut(R_m0), hi=99.0)
    G = norm_pos(cut(G_m0), hi=99.0)
    B = norm_pos(cut(B_m0), hi=99.0)

    # Compose: greyscale base + saturated color from each residual
    grey_weight = 0.55   # keeps central Gaussian visible
    bow_weight  = 1.10   # boost color brightness for visibility
    rgb = np.stack([
        np.clip(grey * grey_weight + R * bow_weight, 0, 1),
        np.clip(grey * grey_weight + G * bow_weight, 0, 1),
        np.clip(grey * grey_weight + B * bow_weight, 0, 1),
    ], axis=-1)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(rgb, origin="lower",
              extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5],
              interpolation="nearest")
    levels = [4.0, 5.0]
    contour_spec = [
        (cut(snr_sio),  "#ff66ff",  "SiO(2-1)"),
        (cut(snr_hcn),  "#ffff66",  "HCN(1-0)"),
        (cut(snr_cs76), "#66ffff",  "CS(7-6)"),
    ]
    handles = []
    for arr, color, name in contour_spec:
        if np.isfinite(arr).sum() < 10:
            continue
        ax.contour(arr, levels=levels, colors=color,
                   linewidths=[0.8, 1.4],
                   extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
        handles.append(plt.Line2D([], [], color=color, lw=1.4,
                                   label=f"{name}  4/5σ"))

    ax.plot(x_src, y_src, "+", color="white", ms=10, mew=1.5, alpha=0.7)
    ax.set_xlim(x_lo - 0.5, x_hi - 0.5)
    ax.set_ylim(y_lo - 0.5, y_hi - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("MUBLO  greyscale = SO 3D-Gaussian model M0 (central component)\n"
                 "RGB = bow-shock residual M0:  R=SO(3-2)  G=CS(2-1)  B=SO₂(2₂,₁-1₁,₁)\n"
                 "contours: faint-line bow-shock-weighted M0 SNR (4σ/5σ; thicker=5σ)",
                 fontsize=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.4,
              facecolor="black", edgecolor="white", labelcolor="white")

    out_png = os.path.join(GAL, "MUBLO_RGB_bowshock_emphasis.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
