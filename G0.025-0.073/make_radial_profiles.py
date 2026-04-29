"""
Radial profiles of all per-line MUBLO cubes, in four flavors:

  Top-left  : Gaussian-weighted M0 of the raw data        (pre-subtraction)
  Top-right : Gaussian-weighted M0 of the residual cube   (post-subtraction)
  Bot-left  : bow-shock-weighted M0 of the raw data       (pre-subtraction)
  Bot-right : bow-shock-weighted M0 of the residual cube  (post-subtraction)

The "residual" for each line is computed by amplitude+offset-only fitting
the SO 3D-Gaussian shape (positions and widths fixed from the SO fit) to
the cube and subtracting that model.  This avoids running curve-fit on
every cube and yields per-line residuals comparable to the SO and CS
shape-locked fits already in disk.

The Gaussian weight uses the SO 3D-Gaussian's velocity profile (FWHM ~188
km/s); the bow-shock weight uses the 1D profile from the SO+CS combined
residual template (~53 km/s FWHM, centered at v ~66 km/s).

Output: galleries/MUBLO_radial_profiles.png
"""
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
GAL = os.path.join(BASE, "galleries")
os.makedirs(GAL, exist_ok=True)

# SO best-fit shape (used as the locked-shape template for every line)
RA0 = 266.4905821
DEC0 = -28.9529931
SX = 0.3545   # arcsec
SY = 0.3620
V0 = 38.962
SV = 79.813   # km/s
FWHM_V = 2.355 * SV

BOW_PROFILE_PATH = os.path.join(BASE, "SOCS_bowshock_profile.fits")

# Per-line cubes — (label, color, path).  Color groups by chemistry/band.
CUBES = [
    # B3 — strong sulfur-bearing
    ("SO(3-2)",       "#e41a1c", os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits")),
    ("CS(2-1)",       "#377eb8", os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.CS21.fits")),
    ("SO2(2_2,1-1_1,1)", "#4daf4a", os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SO2211.fits")),
    ("SO(2_7,3,5-8_2,6)", "#984ea3", os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.SO2735-826.fits")),
    ("OCS(8-7)",      "#ff7f00", os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.OCS8-7.fits")),
    # B3 — N/H-bearing
    ("HCN(1-0)",      "#a65628", os.path.join(BASE, "b3.spw27.cube.I.pbcor.mublo.HCN1-0.fits")),
    ("H13CN(1-0)",    "#f781bf", os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.H13CN10.fits")),
    ("H13CO+(1-0)",   "#999999", os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.H13CO+10.fits")),
    ("HNCO(4-3)",     "#666666", os.path.join(BASE, "b3.spw27.cube.I.pbcor.mublo.HNCO4-3.fits")),
    ("SiO(2-1)",      "#000000", os.path.join(BASE, "b3.spw25.cube.I.pbcor.mublo.SiO21.fits")),
    # B7
    ("13CO(3-2)",     "#1f77b4", os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw25.cube.I.selfcal.pbcor.mublo.13CO32.fits")),
    ("12CO(3-2)",     "#bcbd22", os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw31.cube.I.selfcal.pbcor.mublo.12CO32.fits")),
    ("CS(7-6)",       "#17becf", os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw29.cube.I.selfcal.pbcor.mublo.CS76.fits")),
    ("SO(8_8-7_7)",   "#8c564b", os.path.join(BASE, "b7", "member.uid___A001_X3833_X67e2._G0.02467-0.0727__sci.spw31.cube.I.selfcal.pbcor.mublo.SO8877.fits")),
]


def spectral_axis_kms(header, nchan=None):
    if nchan is None:
        nchan = header["NAXIS3"]
    crpix = header["CRPIX3"]; crval = header["CRVAL3"]; cdelt = header["CDELT3"]
    unit = str(header.get("CUNIT3", "")).strip().lower()
    scale = 1e-3 if unit in ("m/s", "m s-1", "m s^-1") else 1.0
    return (crval + cdelt * (np.arange(nchan) + 1 - crpix)) * scale


def load_bow_profile():
    with fits.open(BOW_PROFILE_PATH) as hd:
        t = hd[1].data
    v = np.array(t["velocity_kms"]); p = np.array(t["profile"])
    if v[0] > v[-1]:
        v = v[::-1]; p = p[::-1]
    return v, p


def shape_locked_residual(cube_path):
    """Load cube, fit SO-shape Gaussian (amp + offset only), return raw +
    residual data plus header and velocity axis."""
    data = fits.getdata(cube_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    hdr = fits.getheader(cube_path)
    nv, ny, nx = data.shape
    vel = spectral_axis_kms(hdr, nv)
    if "FREQ" in str(hdr.get("CTYPE3", "")).upper():
        return None
    wcs = WCS(hdr).celestial
    yi, xi = np.mgrid[0:ny, 0:nx]
    ra2d, dec2d = wcs.wcs_pix2world(xi, yi, 0)
    cosd = np.cos(np.deg2rad(DEC0))
    dx = (ra2d - RA0) * cosd * 3600.0
    dy = (dec2d - DEC0) * 3600.0
    spatial_arg = 0.5 * (dx ** 2 / SX ** 2 + dy ** 2 / SY ** 2)
    spec_arg = 0.5 * ((vel - V0) / SV) ** 2
    unit_tpl = (np.exp(-spatial_arg)[None, :, :] *
                np.exp(-spec_arg)[:, None, None]).astype(np.float32)
    data_f = np.where(np.isfinite(data), data, 0.0).astype(np.float64)
    tpl_flat = unit_tpl.ravel().astype(np.float64)
    d_flat = data_f.ravel()
    ones = np.ones_like(tpl_flat)
    M = np.array([[np.dot(tpl_flat, tpl_flat), np.dot(tpl_flat, ones)],
                  [np.dot(ones, tpl_flat), np.dot(ones, ones)]])
    y = np.array([np.dot(tpl_flat, d_flat), np.dot(ones, d_flat)])
    A, B = np.linalg.solve(M, y)
    model = (A * unit_tpl + B).astype(np.float32)
    residual = (data.astype(np.float32) - model)
    return data.astype(np.float32), residual, hdr, vel, A


def weighted_m0(cube_data, vel, w_v):
    """Sum_v cube_data * w(v) * dv → 2D map (Jy/beam·km/s)."""
    dv = abs(vel[1] - vel[0])
    data_f = np.where(np.isfinite(cube_data), cube_data, 0.0)
    return np.einsum("i,ijk->jk", w_v.astype(np.float32), data_f) * dv


def beam_pixels(hdr):
    """Number of pixels per beam (correlation length for noise)."""
    bmaj = hdr.get("BMAJ"); bmin = hdr.get("BMIN")
    if not bmaj or not bmin:
        return 1.0
    pix = abs(hdr["CDELT1"])
    beam_area = np.pi * bmaj * bmin / (4 * np.log(2))
    return max(1.0, beam_area / pix ** 2)


def radial_profile(m0, hdr, n_bins=15, r_max_arcsec=1.2,
                    off_r_min_arcsec=1.0, off_r_max_arcsec=1.4):
    """Azimuthal average per radial bin, with beam-aware error on the mean
    derived from an off-source annulus.  Returns centers, profile, sigma_mean."""
    wcs = WCS(hdr).celestial
    ny, nx = m0.shape
    yi, xi = np.mgrid[0:ny, 0:nx]
    ra, dec = wcs.wcs_pix2world(xi, yi, 0)
    cosd = np.cos(np.deg2rad(DEC0))
    dx = (ra - RA0) * cosd * 3600.0
    dy = (dec - DEC0) * 3600.0
    r = np.sqrt(dx ** 2 + dy ** 2)

    # Off-source noise: median absolute deviation, scaled to sigma. Use an
    # annulus that's clearly outside the source but well within the cube.
    off_sel = (r > off_r_min_arcsec) & (r < off_r_max_arcsec) & np.isfinite(m0)
    if off_sel.sum() < 30:
        # Fallback to outermost pixels available
        off_sel = (r > 0.7 * r.max()) & np.isfinite(m0)
    off_vals = m0[off_sel]
    if off_vals.size > 5:
        sigma_pix = 1.4826 * np.median(np.abs(off_vals - np.median(off_vals)))
    else:
        sigma_pix = np.nanstd(m0)

    pix_per_beam = beam_pixels(hdr)

    edges = np.linspace(0, r_max_arcsec, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(n_bins, np.nan)
    sigma_mean = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(m0)
        n_pix = int(sel.sum())
        if n_pix >= 3:
            prof[i] = np.nanmean(m0[sel])
            n_indep = max(1.0, n_pix / pix_per_beam)
            sigma_mean[i] = sigma_pix / np.sqrt(n_indep)
    return centers, prof, sigma_mean


def find_peak_radius(centers, prof, sigma_mean, snr_min=3.0):
    """Locate radius of maximum in the profile, restricted to bins where
    SNR>snr_min.  Refine via parabolic interpolation about argmax."""
    valid = np.isfinite(prof) & np.isfinite(sigma_mean) & (sigma_mean > 0)
    snr = np.where(valid, prof / sigma_mean, -np.inf)
    detected = snr > snr_min
    if not detected.any():
        return np.nan
    idx_candidates = np.where(detected)[0]
    if idx_candidates.size == 0:
        return np.nan
    imax = idx_candidates[np.argmax(prof[idx_candidates])]
    if 0 < imax < len(prof) - 1 and np.isfinite(prof[imax-1]) and np.isfinite(prof[imax+1]):
        y0, y1, y2 = prof[imax - 1], prof[imax], prof[imax + 1]
        denom = (y0 - 2 * y1 + y2)
        if denom != 0:
            offset = 0.5 * (y0 - y2) / denom
            offset = float(np.clip(offset, -1, 1))
            dr = centers[1] - centers[0]
            return float(centers[imax] + offset * dr)
    return float(centers[imax])


def main():
    # Build weight axes once per cube (each cube has its own velocity grid)
    bow_v_ref, bow_p_ref = load_bow_profile()

    # Compute four sets of (radius, profile) per line
    results = []  # each: dict with keys label, color, profiles[gauss_data, gauss_res, bow_data, bow_res]
    for label, color, path in CUBES:
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        out = shape_locked_residual(path)
        if out is None:
            print(f"  SKIP FREQ axis: {label}")
            continue
        data, residual, hdr, vel, amp = out

        # Velocity weights on this cube's grid
        w_gauss = np.exp(-0.5 * ((vel - V0) / SV) ** 2).astype(np.float32)
        w_bow   = np.interp(vel, bow_v_ref, bow_p_ref, left=0.0, right=0.0).astype(np.float32)
        w_bow   = np.where(w_bow > 1e-3, w_bow, 0.0)

        m0_gd  = weighted_m0(data,     vel, w_gauss)
        m0_gr  = weighted_m0(residual, vel, w_gauss)
        m0_bd  = weighted_m0(data,     vel, w_bow)
        m0_br  = weighted_m0(residual, vel, w_bow)

        r1, p_gd, e_gd = radial_profile(m0_gd, hdr)
        r2, p_gr, e_gr = radial_profile(m0_gr, hdr)
        r3, p_bd, e_bd = radial_profile(m0_bd, hdr)
        r4, p_br, e_br = radial_profile(m0_br, hdr)

        # Bow-shock-residual peak radius (the diagnostic for SO vs CS offset)
        r_peak_br = find_peak_radius(r4, p_br, e_br)

        results.append({
            "label": label, "color": color, "amp": amp,
            "r": r1,
            "gd": (p_gd, e_gd), "gr": (p_gr, e_gr),
            "bd": (p_bd, e_bd), "br": (p_br, e_br),
            "r_peak_br": r_peak_br,
        })
        print(f"  {label:22s}: SO-shape amp = {amp:.3e} Jy/beam, "
              f"bow-shock residual peak r = {r_peak_br:.3f}\"" if np.isfinite(r_peak_br)
              else f"  {label:22s}: SO-shape amp = {amp:.3e} Jy/beam, no bow-shock peak (SNR<3)")

    # Plot
    SNR_DETECT = 3.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    titles = [
        ("gd", "Gaussian-weighted M0  —  raw data (pre-subtraction)", True),
        ("gr", "Gaussian-weighted M0  —  residual (post-subtraction)", False),
        ("bd", "bow-shock-weighted M0  —  raw data (pre-subtraction)", False),
        ("br", "bow-shock-weighted M0  —  residual (post-subtraction)", False),
    ]
    pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (key, title, log_y), (i, j) in zip(titles, pos):
        ax = axes[i, j]
        for r in results:
            prof, sigma = r[key]
            color = r["color"]
            label = r["label"]
            r_arr = r["r"]
            valid = np.isfinite(prof) & np.isfinite(sigma) & (sigma > 0)
            snr = np.where(valid, prof / sigma, -np.inf)
            det = valid & (np.abs(snr) >= SNR_DETECT)
            non_det = valid & ~det

            # Connect detected points with a line (only between adjacent detections)
            if det.any():
                ax.errorbar(r_arr[det], prof[det], yerr=sigma[det],
                             color=color, lw=1.4, capsize=2, alpha=0.9,
                             marker="o", ms=4, label=label)
            else:
                # Add a label-only proxy so legend shows non-detected lines
                ax.plot([], [], color=color, lw=1.4, alpha=0.5, label=f"{label} (n.d.)")
            # Upper limits at 3σ for non-detections
            if non_det.any():
                up_lim = SNR_DETECT * sigma[non_det]
                ax.errorbar(r_arr[non_det], up_lim,
                             yerr=[0.3 * up_lim, np.zeros_like(up_lim)],
                             color=color, alpha=0.45, lw=0,
                             elinewidth=1.0, uplims=True, marker="v", ms=3)

        ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(0.42, color="red", lw=0.7, ls=":", alpha=0.6)
        ax.text(0.43, ax.get_ylim()[1] if not log_y else 1, "SO HWHM",
                color="red", fontsize=7, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("M0 (Jy/beam · km/s)")
        if log_y:
            ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(True, alpha=0.2)

        # Annotate bow-shock-residual peak radii on the bottom-right panel
        if key == "br":
            so_r = next((r["r_peak_br"] for r in results if r["label"] == "SO(3-2)"), np.nan)
            cs_r = next((r["r_peak_br"] for r in results if r["label"] == "CS(2-1)"), np.nan)
            ymin, ymax = ax.get_ylim()
            if np.isfinite(so_r):
                ax.axvline(so_r, color="#e41a1c", lw=1.5, ls="-", alpha=0.7)
                ax.text(so_r, ymax * 0.95, f" SO peak @ {so_r:.2f}\"",
                         color="#e41a1c", fontsize=9, va="top")
            if np.isfinite(cs_r):
                ax.axvline(cs_r, color="#377eb8", lw=1.5, ls="-", alpha=0.7)
                ax.text(cs_r, ymax * 0.85, f" CS peak @ {cs_r:.2f}\"",
                         color="#377eb8", fontsize=9, va="top")
            if np.isfinite(so_r) and np.isfinite(cs_r):
                ax.text(0.02, 0.02, f"Δr (CS−SO) = {(cs_r-so_r)*1000:+.0f} mas",
                         transform=ax.transAxes, fontsize=9,
                         color="white",
                         bbox=dict(facecolor="black", alpha=0.6, edgecolor="white"))

    axes[1, 0].set_xlabel("radius from MUBLO center (arcsec)")
    axes[1, 1].set_xlabel("radius from MUBLO center (arcsec)")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.7)

    fig.suptitle(
        "MUBLO radial profiles  —  Gaussian (FWHM=188 km/s) vs bow-shock "
        "(FWHM=53 km/s, centered v≈66 km/s) velocity weighting",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(GAL, "MUBLO_radial_profiles.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
