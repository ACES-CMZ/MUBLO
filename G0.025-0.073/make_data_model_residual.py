"""
Six-panel figure for the SO(3-2) and CS(2-1) lines:
  | data M0 | 3D-Gaussian model M0 | residual M0 |
arranged with SO on top row and CS on bottom row.

Data and model panels in each row share a common color scale; the
residual uses a diverging RdBu scale to show the bow-shock structure.

Output: galleries/MUBLO_data_model_residual.png
"""
import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(BASE, "galleries")
os.makedirs(GAL, exist_ok=True)

V0 = 38.962
FWHM_V = 187.96
RA_MUBLO = 266.4905821
DEC_MUBLO = -28.9529931

CUBES = [
    {
        "name": "SO(3-2)",
        "data":  os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits"),
        "model": os.path.join(BASE, "SO32_model3D_gaussian.fits"),
        "resid": os.path.join(BASE, "SO32_residual3D_gaussian.fits"),
    },
    {
        "name": "CS(2-1)",
        "data":  os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.CS21.fits"),
        "model": os.path.join(BASE, "CS21_model3D_gaussian_solocked.fits"),
        "resid": os.path.join(BASE, "CS21_residual3D_gaussian_solocked.fits"),
    },
]


def spectral_axis_kms(header, nchan=None):
    if nchan is None:
        nchan = header["NAXIS3"]
    crpix = header["CRPIX3"]; crval = header["CRVAL3"]; cdelt = header["CDELT3"]
    unit = str(header.get("CUNIT3", "")).strip().lower()
    scale = 1e-3 if unit in ("m/s", "m s-1", "m s^-1") else 1.0
    return (crval + cdelt * (np.arange(nchan) + 1 - crpix)) * scale


def mom0(path, v_lo=V0 - 1.2 * FWHM_V, v_hi=V0 + 1.2 * FWHM_V):
    d = fits.getdata(path).astype(np.float32)
    while d.ndim > 3:
        d = d[0]
    h = fits.getheader(path)
    v = spectral_axis_kms(h, d.shape[0])
    sel = (v >= v_lo) & (v <= v_hi)
    dv = abs(v[1] - v[0])
    return np.nansum(d[sel], axis=0) * dv, h


def _safe_hdr(h, key, default=None):
    val = h.get(key, default)
    try:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return default
    except Exception:
        return default
    return val


def main():
    rows_data = []
    for cube in CUBES:
        m0_data,  hdr  = mom0(cube["data"])
        m0_model, _    = mom0(cube["model"])
        m0_resid, _    = mom0(cube["resid"])
        rows_data.append((cube["name"], hdr, m0_data, m0_model, m0_resid))

    # Use the first cube's WCS for the spatial cutout (all share the B3 grid).
    hdr_ref = rows_data[0][1]
    wcs = WCS(hdr_ref).celestial
    x_src, y_src = wcs.wcs_world2pix(RA_MUBLO, DEC_MUBLO, 0)
    ix, iy = int(round(float(x_src))), int(round(float(y_src)))
    cdelt = abs(hdr_ref["CDELT1"])
    half = int(round(1.5 / 3600.0 / cdelt))
    nx = hdr_ref["NAXIS1"]; ny = hdr_ref["NAXIS2"]
    x_lo = max(0, ix - half); x_hi = min(nx, ix + half + 1)
    y_lo = max(0, iy - half); y_hi = min(ny, iy + half + 1)

    def cut(a):
        return a[y_lo:y_hi, x_lo:x_hi]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))

    for irow, (name, hdr, m0_data, m0_model, m0_resid) in enumerate(rows_data):
        d_c = cut(m0_data); m_c = cut(m0_model); r_c = cut(m0_resid)
        # Shared scale for data + model
        vmax_dm = np.nanpercentile(np.concatenate([d_c.ravel(), m_c.ravel()]), 99.5)
        vmin_dm = 0
        # Symmetric scale for residual
        vmax_r = np.nanpercentile(np.abs(r_c), 99.0)

        for icol, (im_arr, title, vmin, vmax, cmap) in enumerate([
            (d_c, f"{name}  data M0", vmin_dm, vmax_dm, "viridis"),
            (m_c, f"{name}  Gaussian model M0", vmin_dm, vmax_dm, "viridis"),
            (r_c, f"{name}  residual M0", -vmax_r, vmax_r, "RdBu_r"),
        ]):
            ax = axes[irow, icol]
            im = ax.imshow(im_arr, origin="lower", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[x_lo - 0.5, x_hi - 0.5,
                                   y_lo - 0.5, y_hi - 0.5],
                           interpolation="nearest")
            ax.plot(x_src, y_src, "+", color="white", ms=10, mew=1.5, alpha=0.7)
            ax.set_xlim(x_lo - 0.5, x_hi - 0.5)
            ax.set_ylim(y_lo - 0.5, y_hi - 0.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)

        # Beam ellipse on the data panel
        bmaj = _safe_hdr(hdr, "BMAJ"); bmin = _safe_hdr(hdr, "BMIN")
        bpa = _safe_hdr(hdr, "BPA", 0.0)
        if bmaj and bmin:
            ex = x_lo + 0.1 * (x_hi - x_lo); ey = y_lo + 0.1 * (y_hi - y_lo)
            beam = Ellipse((ex, ey), width=bmin / cdelt, height=bmaj / cdelt,
                           angle=bpa, fill=False, edgecolor="white", lw=1.2)
            axes[irow, 0].add_patch(beam)

    fig.suptitle(f"Data, 3D-Gaussian model, and residual moment 0 maps "
                 f"(integrated over v_LSR in [{V0-1.2*FWHM_V:.0f}, {V0+1.2*FWHM_V:.0f}] km/s)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(GAL, "MUBLO_data_model_residual.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
