"""
Make an RGB composite of the three brightest MUBLO lines
(SO(3-2) red, SO2(2_2,1-1_1,1) green, CS(2-1) blue) with the SO+CS
bow-shock template added as a "whitening" layer.  Overlay 4σ and 5σ
contours of the bow-shock-weighted moment-0 SNR for the faint lines
(SiO(2-1), HCN(1-0), CS(7-6)).

Bow-shock weighting uses the 1D velocity profile from the combined
SO+CS residual template (SOCS_bowshock_profile.fits); it is narrower
than the 3D Gaussian, so it preferentially captures bow-shock emission.

Outputs:
  galleries/MUBLO_RGB_bowshock.png
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
RA_MUBLO = 266.4905821
DEC_MUBLO = -28.9529931

SO32_CUBE  = os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits")
SO2_CUBE   = os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SO2211.fits")
CS21_CUBE  = os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.CS21.fits")
SIO_CUBE   = os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SiO21.fits")
HCN_CUBE   = os.path.join(BASE, "b3.spw27.cube.I.pbcor.mublo.HCN1-0.fits")
CS76_CUBE  = os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw29.cube.I.selfcal.pbcor.mublo.CS76.fits")
BOW_TEMPLATE = os.path.join(BASE, "SOCS_bowshock_template.fits")
BOW_PROFILE  = os.path.join(BASE, "SOCS_bowshock_profile.fits")


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


def load_bow_profile():
    with fits.open(BOW_PROFILE) as hd:
        t = hd[1].data
    v = np.array(t["velocity_kms"]); p = np.array(t["profile"])
    if v[0] > v[-1]:
        v = v[::-1]; p = p[::-1]
    return v, p


def bow_weighted_m0_snr(cube_path):
    """Compute bow-shock-weighted moment 0 and its SNR map for a cube."""
    d, h, v = load_cube(cube_path)
    nv, ny, nx = d.shape
    dv = abs(v[1] - v[0])
    v_ref, p_ref = load_bow_profile()
    w = np.interp(v, v_ref, p_ref, left=0.0, right=0.0).astype(np.float32)
    w = np.where(w > 1e-3, w, 0.0)
    data_f = np.where(np.isfinite(d), d, 0.0)
    m0 = np.einsum("i,ijk->jk", w, data_f) * dv
    # Off-signal RMS for noise
    off_mask = np.abs(v - V0) > 300.0
    if off_mask.sum() < 5:
        off_mask = np.abs(v - V0) > 200.0
    rms = np.nanstd(data_f[off_mask]) if off_mask.any() else np.nan
    sigma_m0 = rms * np.sqrt((w ** 2).sum()) * dv if np.isfinite(rms) else np.nan
    snr = m0 / sigma_m0 if np.isfinite(sigma_m0) and sigma_m0 > 0 else np.full_like(m0, np.nan)
    return m0, snr, sigma_m0, h


def standard_m0(cube_path, v_lo=V0 - 1.2 * FWHM_V, v_hi=V0 + 1.2 * FWHM_V):
    d, h, v = load_cube(cube_path)
    sel = (v >= v_lo) & (v <= v_hi)
    dv = abs(v[1] - v[0])
    return np.nansum(d[sel], axis=0) * dv, h


def bow_shock_layer():
    """2D image of the bow-shock template summed over velocity (its natural
    positive-only shape)."""
    d = fits.getdata(BOW_TEMPLATE).astype(np.float32)
    while d.ndim > 3:
        d = d[0]
    return d.sum(axis=0)


def reproject_to_b3(snr_b7, hdr_b7, hdr_b3_ref):
    """Reproject a 2D map from B7 WCS onto B3 WCS via pixel-coord round-trip
    and linear interpolation."""
    wcs_b3 = WCS(hdr_b3_ref).celestial
    wcs_b7 = WCS(hdr_b7).celestial
    ny_b3 = hdr_b3_ref["NAXIS2"]; nx_b3 = hdr_b3_ref["NAXIS1"]
    yi, xi = np.mgrid[0:ny_b3, 0:nx_b3]
    ra_b3, dec_b3 = wcs_b3.wcs_pix2world(xi, yi, 0)
    x_b7, y_b7 = wcs_b7.wcs_world2pix(ra_b3, dec_b3, 0)
    coords = np.vstack([y_b7.ravel(), x_b7.ravel()])
    out = map_coordinates(snr_b7, coords, order=1, mode="constant",
                          cval=np.nan).reshape(ny_b3, nx_b3)
    return out


def normalize_pos(arr, pct_lo=5, pct_hi=99.5):
    """Normalize to [0,1] using percentile clipping of positive values."""
    vmin = np.nanpercentile(arr, pct_lo)
    vmax = np.nanpercentile(arr, pct_hi)
    out = (arr - vmin) / max(vmax - vmin, 1e-30)
    return np.clip(out, 0, 1)


def main():
    # Brightest lines: mom0 maps (standard window integration)
    m0_so32, h_so32   = standard_m0(SO32_CUBE)
    m0_so2,  h_so2    = standard_m0(SO2_CUBE)
    m0_cs21, h_cs21   = standard_m0(CS21_CUBE)
    bow_2d            = bow_shock_layer()
    b3_hdr = h_so32  # shared B3 grid

    # Faint lines: bow-shock-weighted M0 SNR maps
    _, snr_sio, _, h_sio = bow_weighted_m0_snr(SIO_CUBE)
    _, snr_hcn, _, h_hcn = bow_weighted_m0_snr(HCN_CUBE)
    _, snr_cs76_b7, _, h_cs76 = bow_weighted_m0_snr(CS76_CUBE)
    snr_cs76 = reproject_to_b3(snr_cs76_b7, h_cs76, b3_hdr)

    # Spatial crop around MUBLO (±1.5")
    wcs_b3 = WCS(b3_hdr).celestial
    x_src, y_src = wcs_b3.wcs_world2pix(RA_MUBLO, DEC_MUBLO, 0)
    ix, iy = int(round(float(x_src))), int(round(float(y_src)))
    cdelt1 = abs(b3_hdr["CDELT1"])
    half = int(round(1.5 / 3600.0 / cdelt1))
    x_lo = max(0, ix - half); x_hi = min(b3_hdr["NAXIS1"], ix + half + 1)
    y_lo = max(0, iy - half); y_hi = min(b3_hdr["NAXIS2"], iy + half + 1)

    def cut(a):
        return a[y_lo:y_hi, x_lo:x_hi]

    # Build RGB + whitening
    R = normalize_pos(cut(m0_so32))
    G = normalize_pos(cut(m0_so2))
    B = normalize_pos(cut(m0_cs21))
    W = normalize_pos(cut(bow_2d), pct_hi=99.0)
    # Whitening weight — tuneable; 0.4 gives visible but not dominant white glow
    w_factor = 0.5
    R_w = np.clip(R + w_factor * W, 0, 1)
    G_w = np.clip(G + w_factor * W, 0, 1)
    B_w = np.clip(B + w_factor * W, 0, 1)
    rgb = np.stack([R_w, G_w, B_w], axis=-1)

    # Contour data for faint lines, cropped
    snr_sio_c = cut(snr_sio)
    snr_hcn_c = cut(snr_hcn)
    snr_cs76_c = cut(snr_cs76)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(rgb, origin="lower",
              extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5],
              interpolation="nearest")

    levels = [4.0, 5.0]
    contour_spec = [
        (snr_sio_c,  "#ff4444",  "SiO(2-1)"),
        (snr_hcn_c,  "#44ff44",  "HCN(1-0)"),
        (snr_cs76_c, "#44ffff",  "CS(7-6)"),
    ]
    handles = []
    for arr, color, name in contour_spec:
        finite = np.isfinite(arr)
        if finite.sum() < 10:
            continue
        # Different linewidths for 4σ vs 5σ
        cs = ax.contour(arr, levels=levels, colors=color,
                        linewidths=[0.8, 1.4],
                        extent=[x_lo - 0.5, x_hi - 0.5, y_lo - 0.5, y_hi - 0.5])
        # Legend proxy
        handles.append(plt.Line2D([], [], color=color, lw=1.4,
                                   label=f"{name}  4/5σ"))

    ax.plot(x_src, y_src, "x", color="white", ms=10, mew=2, alpha=0.7)
    ax.set_xlim(x_lo - 0.5, x_hi - 0.5)
    ax.set_ylim(y_lo - 0.5, y_hi - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("MUBLO  RGB = SO(3-2) / SO₂(2₂,₁-1₁,₁) / CS(2-1)   "
                 "+ SO+CS bow-shock (white glow)\n"
                 "Contours (4σ & 5σ, thicker=5σ): bow-shock-weighted M0 SNR of faint lines",
                 fontsize=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.4,
              facecolor="black", edgecolor="white", labelcolor="white")

    out_png = os.path.join(GAL, "MUBLO_RGB_bowshock.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
