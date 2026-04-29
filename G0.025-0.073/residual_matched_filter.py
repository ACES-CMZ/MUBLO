"""
Residual-based matched filter.

Starting from the SO(3-2) - 3D-Gaussian-model residual cube, build a template
that isolates the bow-shock/edge-brightened structure:

 1. Compute off-signal sigma from far-from-line channels.
 2. Mask by S/N expansion: seed = |resid| > n_seed*sigma; grow iteratively
    into |resid| > n_expand*sigma via binary_dilation.  Try several
    (n_seed, n_expand) pairs and report coverage statistics.
 3. Apply the chosen mask and smooth (3D Gaussian kernel) to produce a
    high-S/N template.
 4. Interpolate the template onto each target cube's (RA, Dec, v) grid.
 5. Run the matched filter (reuse matched_filter_spectrum from fit3d_gaussian).

Also fit the CS(2-1) cube with the SO32 shape (positions, widths, velocity
centroid, velocity width fixed from SO fit), free only amplitude + offset,
and save the CS model + residual so the bow-shock structure can be
compared between SO and CS.
"""
import os
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

from fit3d_gaussian import (spectral_axis_kms, gauss3d_on_grid,
                              matched_filter_spectrum)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

# SO fit parameters (from fit3d_gaussian.py best-fit)
SO_PARAMS = (2.1432e-03,       # amp (Jy/beam, reused only as starting guess
             266.4905821,      # x0 deg
             -28.9529931,      # y0 deg
             0.3545,           # sx arcsec
             0.3620,           # sy arcsec
             38.962,           # v0 km/s
             79.813,           # sv km/s
             0.0)              # offset
_, X0, Y0, SX, SY, V0, SV, _ = SO_PARAMS
FWHM_V = 2.355 * SV

SO32_RESIDUAL = os.path.join(BASE, "SO32_residual3D_gaussian.fits")
SO32_CUBE = os.path.join(BASE, "b3.spw31.cube.I.pbcor.mublo.SO32.fits")
CS21_CUBE = os.path.join(BASE, "b3.spw29.cube.I.pbcor.mublo.CS21.fits")


def snr_expand_mask(data, sigma, n_seed, n_expand, max_iter=40):
    """Grow a seed mask (>n_seed*sigma) into a permitted region (>n_expand*sigma)
    via iterated binary dilation until convergence."""
    abs_d = np.abs(data)
    allowed = abs_d > n_expand * sigma
    seed = abs_d > n_seed * sigma
    mask = seed.copy()
    for _ in range(max_iter):
        new = binary_dilation(mask) & allowed
        if new.sum() == mask.sum():
            break
        mask = new
    return mask


def _plot_mom0_diagnostic(data, masks_dict, vel_sel, out_png, title="Residual mom0"):
    """Plot moment-0 of residual with masks overlaid for inspection."""
    n = len(masks_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    m0_raw = data[vel_sel].sum(axis=0)
    # Small spatial crop around MUBLO for inspection (SO32 cube pixel scale)
    ny, nx = m0_raw.shape
    cy, cx = ny // 2, nx // 2
    half = 40
    sl = (slice(max(0, cy - half), min(ny, cy + half)),
          slice(max(0, cx - half), min(nx, cx + half)))
    vmax = np.nanpercentile(np.abs(m0_raw[sl]), 99.5)
    ax = axes[0]
    im = ax.imshow(m0_raw[sl], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("raw residual mom0")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i, (key, mask) in enumerate(masks_dict.items(), start=1):
        masked = np.where(mask & (data > 0), data, 0.0)
        m0 = masked[vel_sel].sum(axis=0)
        ax = axes[i]
        im = ax.imshow(m0[sl], origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_title(f"mask {key}\n({int(mask.sum())} vox)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_residual_template(resid_path, out_prefix,
                             candidate_masks=(("3->1", 3.0, 1.0),
                                              ("4->1", 4.0, 1.0),
                                              ("4->1.5", 4.0, 1.5),
                                              ("4->2", 4.0, 2.0),
                                              ("5->2", 5.0, 2.0)),
                             chosen_key="4->2",
                             v_window_kms=250.0,
                             smooth_spatial_pix=2.0, smooth_spec_chan=1.2):
    """Build candidate S/N-expansion masks, restrict to a velocity window
    around MUBLO, smooth the chosen mask, and use its positive residual as
    the template.  Produces a diagnostic PNG of mom0 with each mask overlay
    so the choice can be reviewed.
    """
    hdr = fits.getheader(resid_path)
    data = fits.getdata(resid_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    nv, ny, nx = data.shape
    vel = spectral_axis_kms(hdr, nv)

    off = np.abs(vel - V0) > 200.0
    if off.sum() < 4:
        off = np.abs(vel - V0) > 150.0
    sigma = float(np.nanstd(data[off]))
    # Velocity clip: only voxels within ±v_window of MUBLO are eligible —
    # this avoids building the template out of noise in the far wings.
    v_clip = np.abs(vel - V0) <= v_window_kms
    vmask3d = np.zeros_like(data, dtype=bool)
    vmask3d[v_clip] = True
    print(f"[residual] shape={data.shape}  sigma={sigma:.3e}  "
          f"n_off_chan={int(off.sum())}  v_window=±{v_window_kms:.0f} km/s "
          f"({int(v_clip.sum())} chan)")

    masks = {}
    for key, ns, ne in candidate_masks:
        m_raw = snr_expand_mask(data, sigma, n_seed=ns, n_expand=ne)
        m = m_raw & vmask3d
        pos = m & (data > 0)
        print(f"  mask {key:8s}  (seed {ns:.1f}σ, expand {ne:.1f}σ, |v-v0|<{v_window_kms:.0f}): "
              f"voxels={int(m.sum()):7d}  pos={int(pos.sum()):7d}  "
              f"pos_flux={float(data[pos].sum()):.3e}")
        masks[key] = m

    if chosen_key not in masks:
        raise KeyError(f"chosen_key={chosen_key} not in candidate masks")
    chosen = masks[chosen_key]
    print(f"[residual] using chosen={chosen_key}")

    # Apply mask (positive residual only) and smooth
    masked = np.where(chosen & (data > 0), data, 0.0).astype(np.float32)
    smoothed = gaussian_filter(masked, sigma=(smooth_spec_chan,
                                              smooth_spatial_pix,
                                              smooth_spatial_pix))
    peak = float(np.nanmax(smoothed))
    if peak <= 0:
        raise RuntimeError(f"Smoothed template peak={peak}; mask too tight")
    smoothed /= peak

    # Save cubes
    out_hdr = hdr.copy()
    for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
        if k in out_hdr:
            del out_hdr[k]
    out_hdr["NAXIS"] = 3
    out_hdr["HISTORY"] = (f"S/N-expansion mask ({chosen_key} sigma), "
                          f"|v-v0|<={v_window_kms:.0f} km/s, pos-only, smoothed")
    fits.writeto(f"{out_prefix}_template.fits",
                 smoothed.astype(np.float32), out_hdr, overwrite=True)
    fits.writeto(f"{out_prefix}_mask.fits", chosen.astype(np.uint8),
                 out_hdr, overwrite=True)
    fits.writeto(f"{out_prefix}_masked.fits", masked,
                 out_hdr, overwrite=True)
    print(f"[residual] wrote {out_prefix}_template.fits")

    # Diagnostic PNG
    diag_png = f"{out_prefix}_mask_diagnostic.png"
    os.makedirs(os.path.join(BASE, "galleries"), exist_ok=True)
    gallery_png = os.path.join(BASE, "galleries", os.path.basename(diag_png))
    _plot_mom0_diagnostic(data, masks, vel_sel=v_clip,
                          out_png=gallery_png,
                          title=f"Residual mom0 (|v-v0|<{v_window_kms:.0f} km/s), "
                                f"σ={sigma:.2e}")
    print(f"[residual] wrote diagnostic {gallery_png}")
    return smoothed, hdr, vel


def interpolate_template_to(target_path, template, template_header, template_vel,
                             out_path=None):
    """Interpolate a template cube (on the SO residual grid) onto the target
    cube's (v, y, x) grid, using RA/Dec and velocity matching."""
    thdr = fits.getheader(target_path)
    if "FREQ" in str(thdr.get("CTYPE3", "")).upper():
        return None, None
    tnv = thdr["NAXIS3"]
    tvel = spectral_axis_kms(thdr, tnv)
    tny = thdr["NAXIS2"]
    tnx = thdr["NAXIS1"]

    # Target spatial coords in degrees
    wcs_cel_t = WCS(thdr).celestial
    yi, xi = np.mgrid[0:tny, 0:tnx]
    t_ra, t_dec = wcs_cel_t.wcs_pix2world(xi, yi, 0)

    # Convert target (RA, Dec) → template pixel (y_pix, x_pix) using template WCS
    wcs_cel_s = WCS(template_header).celestial
    s_x, s_y = wcs_cel_s.wcs_world2pix(t_ra, t_dec, 0)

    # Template velocity axis -> template channel index
    s_dv = template_vel[1] - template_vel[0]
    s_cstart = template_vel[0]

    tpl_shape = template.shape  # (nv_s, ny_s, nx_s)

    # Build target cube of template values via map_coordinates (trilinear)
    target_template = np.zeros((tnv, tny, tnx), dtype=np.float32)
    # For each target velocity channel, interpolate the 2D (s_y, s_x) grid from
    # the template at the corresponding fractional template-channel index
    for iv in range(tnv):
        tv = tvel[iv]
        s_chan = (tv - s_cstart) / s_dv
        if s_chan < -0.5 or s_chan > tpl_shape[0] - 0.5:
            continue  # outside template velocity coverage -> template = 0
        # Bilinear in spectral axis between the two adjacent template channels
        c0 = int(np.floor(s_chan))
        c1 = c0 + 1
        frac = s_chan - c0
        if c0 < 0:
            c0 = 0; frac = 0.0; c1 = 0
        if c1 >= tpl_shape[0]:
            c1 = tpl_shape[0] - 1; c0 = c1; frac = 0.0

        # Use map_coordinates on the two spatial slices
        coords = np.vstack([s_y.ravel(), s_x.ravel()])
        sl0 = map_coordinates(template[c0], coords, order=1, mode="constant",
                              cval=0.0).reshape(tny, tnx)
        if c1 == c0:
            target_template[iv] = sl0
        else:
            sl1 = map_coordinates(template[c1], coords, order=1, mode="constant",
                                  cval=0.0).reshape(tny, tnx)
            target_template[iv] = (1 - frac) * sl0 + frac * sl1

    if out_path:
        out_hdr = thdr.copy()
        for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
            if k in out_hdr:
                del out_hdr[k]
        out_hdr["NAXIS"] = 3
        fits.writeto(out_path, target_template, out_hdr, overwrite=True)
    return target_template, tvel


def run_residual_mf(cube_path, template, template_header, template_vel,
                    label=""):
    """Evaluate the residual-based template on the target cube, write outputs."""
    print(f"\n[resid-mf] {os.path.basename(cube_path)}  [{label}]")
    thdr = fits.getheader(cube_path)
    if "FREQ" in str(thdr.get("CTYPE3", "")).upper():
        print("   skipped (FREQ axis)")
        return
    tpl_out = cube_path.replace(".fits", ".residtpl.fits")
    target_tpl, tvel = interpolate_template_to(cube_path, template,
                                                template_header, template_vel,
                                                out_path=tpl_out)
    if target_tpl is None:
        print("   skipped")
        return
    data = fits.getdata(cube_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    if data.shape != target_tpl.shape:
        print(f"   shape mismatch {data.shape} vs {target_tpl.shape}")
        return
    if np.nanmax(target_tpl) <= 0:
        print("   template is zero on this cube (no overlap)")
        return

    off_mask = np.abs(tvel - V0) > 250.0
    if off_mask.sum() < 5:
        off_mask = np.abs(tvel - V0) > 150.0

    spec, sigma, i_center = matched_filter_spectrum(data, target_tpl, off_mask=off_mask)
    # The residual template has extended structure; report SNR at MUBLO v0
    # specifically, in addition to the template-peak and spectrum-max values.
    i_v0 = int(np.argmin(np.abs(tvel - V0)))
    v_peak = tvel[i_center]
    snr_spec = spec / sigma if sigma > 0 else np.full_like(spec, np.nan)
    snr_at_v0 = snr_spec[i_v0]
    snr_at_peak = snr_spec[i_center]
    finite_snr = np.where(np.isfinite(snr_spec), snr_spec, -np.inf)
    imax = int(np.argmax(finite_snr))
    print(f"   template peak channel={i_center} at v={v_peak:.2f} km/s")
    print(f"   flux@v0={spec[i_v0]:.3e}  SNR@v0={snr_at_v0:.2f}  (v0={V0:.1f} km/s)")
    print(f"   flux@tpl_peak={spec[i_center]:.3e}  SNR@tpl_peak={snr_at_peak:.2f}")
    print(f"   sigma={sigma:.3e}  max SNR in spectrum={snr_spec[imax]:.2f} at v={tvel[imax]:.2f}")

    out_path = cube_path.replace(".fits", ".resid_mf_spectrum.fits")
    cols = [
        fits.Column(name="velocity_kms", format="D", array=tvel),
        fits.Column(name="flux", format="D", array=spec),
        fits.Column(name="snr", format="D", array=snr_spec),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)

    def safe(v):
        if not np.isfinite(v):
            return 0.0
        return float(v)

    hdu.header["SIGMA"] = (safe(sigma), "off-signal stdev of flux spectrum")
    hdu.header["VMUBLO"] = (safe(V0), "MUBLO v0, km/s")
    hdu.header["SNRATV0"] = (safe(snr_at_v0), "SNR at MUBLO v0")
    hdu.header["VPEAK"] = (safe(v_peak), "template peak channel velocity, km/s")
    hdu.header["SNRPEAK"] = (safe(snr_at_peak), "SNR at template peak channel")
    hdu.header["SNRMAX"] = (safe(snr_spec[imax]), "peak SNR in spectrum")
    hdu.header["VSNRMAX"] = (safe(tvel[imax]), "velocity of peak SNR, km/s")
    hdu.header["TEMPLATE"] = ("SO32_residual_snrexpand_smoothed",
                              "residual-based template source")
    hdu.writeto(out_path, overwrite=True)
    print(f"   wrote {out_path}")


def build_combined_residual_template(so_resid_path, cs_resid_path, out_prefix,
                                       chosen_key="5->2",
                                       v_window_kms=250.0,
                                       smooth_spatial_pix=2.0,
                                       smooth_spec_chan=1.2):
    """Combine SO(3-2) and CS(2-1) residual cubes into a higher-S/N
    bow-shock template.  Both residuals are on the same spatial grid and
    very similar velocity grids; CS is linearly resampled onto SO's velocity
    axis, each is normalized by its own off-signal sigma, and they are
    summed (equal weight).  Then the usual S/N-expansion, masking, and
    smoothing pipeline runs on the combined cube.

    Also extracts and writes the 1D spatial-sum velocity profile of the
    combined template — useful for narrow-line MF-weighted moment maps.
    """
    so_data = fits.getdata(so_resid_path).astype(np.float32)
    while so_data.ndim > 3:
        so_data = so_data[0]
    so_hdr = fits.getheader(so_resid_path)
    so_vel = spectral_axis_kms(so_hdr, so_data.shape[0])

    cs_data = fits.getdata(cs_resid_path).astype(np.float32)
    while cs_data.ndim > 3:
        cs_data = cs_data[0]
    cs_hdr = fits.getheader(cs_resid_path)
    cs_vel = spectral_axis_kms(cs_hdr, cs_data.shape[0])

    if so_data.shape[1:] != cs_data.shape[1:]:
        raise RuntimeError(f"Spatial shape mismatch: SO {so_data.shape[1:]} vs "
                           f"CS {cs_data.shape[1:]}")

    # Resample CS onto SO's velocity axis (linear interp per-pixel)
    nv_so, ny, nx = so_data.shape
    cs_on_so = np.zeros_like(so_data)
    for yi in range(ny):
        for xi in range(nx):
            cs_on_so[:, yi, xi] = np.interp(so_vel, cs_vel[::-1],
                                              cs_data[::-1, yi, xi],
                                              left=0.0, right=0.0)
            # Note: cs_vel decreases with channel; reverse to get ascending.

    # Off-signal sigma for each
    off = np.abs(so_vel - V0) > 200.0
    if off.sum() < 4:
        off = np.abs(so_vel - V0) > 150.0
    sigma_so = float(np.nanstd(so_data[off]))
    sigma_cs = float(np.nanstd(cs_on_so[off]))
    print(f"[combined] SO σ={sigma_so:.3e}  CS σ={sigma_cs:.3e}")

    # S/N cubes
    snr_so = so_data / sigma_so
    snr_cs = cs_on_so / sigma_cs

    # Combined S/N: inverse-variance weighted sum in S/N units (equivalent
    # to sum of S/N per voxel divided by sqrt(N_lines) since both are in
    # S/N units). We take the unweighted average and then treat it as a
    # "combined S/N" with effective sigma = 1.
    combined_snr = (snr_so + snr_cs) / np.sqrt(2.0)  # in combined-sigma units
    sigma_eff = 1.0
    print(f"[combined] combined S/N cube: peak={np.nanmax(combined_snr):.2f}, "
          f"min={np.nanmin(combined_snr):.2f}")

    # Apply velocity window
    v_clip = np.abs(so_vel - V0) <= v_window_kms
    vmask3d = np.zeros_like(combined_snr, dtype=bool)
    vmask3d[v_clip] = True

    # S/N expansion masks on combined S/N
    candidate = (("3->1", 3.0, 1.0), ("4->1", 4.0, 1.0),
                 ("4->1.5", 4.0, 1.5), ("4->2", 4.0, 2.0),
                 ("5->2", 5.0, 2.0), ("5->3", 5.0, 3.0))
    masks = {}
    for key, ns, ne in candidate:
        m_raw = snr_expand_mask(combined_snr, sigma_eff, n_seed=ns, n_expand=ne)
        m = m_raw & vmask3d
        pos = m & (combined_snr > 0)
        print(f"  mask {key:7s}: voxels={int(m.sum()):7d}  pos={int(pos.sum()):7d}  "
              f"peak_snr_in_mask={float(combined_snr[pos].max() if pos.any() else 0):.2f}")
        masks[key] = m

    if chosen_key not in masks:
        raise KeyError(f"{chosen_key} not in candidate masks")
    chosen = masks[chosen_key]
    print(f"[combined] using chosen={chosen_key}  ({int(chosen.sum())} voxels, "
          f"{100*chosen.sum()/chosen.size:.3f}% of cube)")

    # Positive-only + smooth the combined S/N cube (carries the bow-shock
    # structure at higher S/N than either input alone).
    masked = np.where(chosen & (combined_snr > 0), combined_snr, 0.0).astype(np.float32)
    smoothed = gaussian_filter(masked, sigma=(smooth_spec_chan,
                                              smooth_spatial_pix,
                                              smooth_spatial_pix))
    peak = float(np.nanmax(smoothed))
    if peak <= 0:
        raise RuntimeError("Smoothed combined template peak is <=0")
    smoothed /= peak

    # Save cubes
    out_hdr = so_hdr.copy()
    for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
        if k in out_hdr:
            del out_hdr[k]
    out_hdr["NAXIS"] = 3
    out_hdr["HISTORY"] = (f"Combined SO+CS S/N bow-shock template "
                          f"({chosen_key} sigma, |v-v0|<={v_window_kms:.0f} km/s)")
    fits.writeto(f"{out_prefix}_template.fits",
                 smoothed.astype(np.float32), out_hdr, overwrite=True)
    fits.writeto(f"{out_prefix}_mask.fits", chosen.astype(np.uint8),
                 out_hdr, overwrite=True)
    fits.writeto(f"{out_prefix}_combined_snr.fits",
                 combined_snr.astype(np.float32), out_hdr, overwrite=True)
    print(f"[combined] wrote {out_prefix}_template.fits")
    print(f"[combined] wrote {out_prefix}_mask.fits")
    print(f"[combined] wrote {out_prefix}_combined_snr.fits")

    # 1D velocity profile: spatial sum per channel, normalized to peak=1.
    profile = smoothed.sum(axis=(1, 2))
    if profile.max() > 0:
        profile = profile / profile.max()

    # Report FWHM of the profile
    halfmax = profile > 0.5
    if halfmax.any():
        v_in_half = so_vel[halfmax]
        fwhm_v = float(np.abs(v_in_half.max() - v_in_half.min()))
    else:
        fwhm_v = float("nan")
    v_centroid = float(np.sum(so_vel * profile) / max(profile.sum(), 1e-30))
    print(f"[combined] bow-shock profile: centroid={v_centroid:.2f} km/s  "
          f"FWHM~{fwhm_v:.1f} km/s  (cf. SO-Gaussian FWHM={2.355*SV:.1f})")

    cols = [fits.Column(name="velocity_kms", format="D", array=so_vel),
            fits.Column(name="profile", format="D", array=profile)]
    phdu = fits.BinTableHDU.from_columns(cols)
    phdu.header["VCENT"] = (v_centroid, "profile centroid, km/s")
    phdu.header["FWHM"] = (fwhm_v, "FWHM by half-max bracket, km/s")
    phdu.header["SOURCE"] = ("SO+CS combined residual bow-shock", "origin")
    prof_path = f"{out_prefix}_profile.fits"
    phdu.writeto(prof_path, overwrite=True)
    print(f"[combined] wrote {prof_path}")

    # Diagnostic PNG
    gallery_dir = os.path.join(BASE, "galleries")
    os.makedirs(gallery_dir, exist_ok=True)
    diag_png = os.path.join(gallery_dir,
                            os.path.basename(out_prefix) + "_mask_diagnostic.png")
    _plot_mom0_diagnostic(combined_snr, masks, vel_sel=v_clip,
                          out_png=diag_png,
                          title=f"Combined SO+CS S/N residual mom0 "
                                f"(|v-v0|<{v_window_kms:.0f} km/s)")
    print(f"[combined] wrote diagnostic {diag_png}")

    # Also a tiny PNG of the 1D profile vs the Gaussian used previously
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(so_vel, profile, color="navy", lw=1.2, label="bow-shock profile")
    gauss = np.exp(-0.5 * ((so_vel - V0) / SV) ** 2)
    ax.plot(so_vel, gauss / gauss.max(), color="red", lw=1.0, ls="--",
            label=f"SO 3D Gaussian (FWHM={2.355*SV:.0f})")
    ax.axvline(V0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("v_LSR (km/s)")
    ax.set_ylabel("normalized weight")
    ax.set_title(f"bow-shock velocity profile  (FWHM~{fwhm_v:.0f} km/s, "
                 f"centroid={v_centroid:.1f})", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(V0 - 300, V0 + 300)
    prof_png = os.path.join(gallery_dir,
                            os.path.basename(out_prefix) + "_profile.png")
    fig.savefig(prof_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[combined] wrote {prof_png}")

    return smoothed, out_hdr, so_vel, profile


def _pseudo_velocity_axis(hdr):
    """Return a pseudo-velocity axis (km/s) for a FREQ cube, using the
    cube's median frequency as the reference.  Returns (vel, f_ref_hz)."""
    nv = hdr["NAXIS3"]
    crpix = hdr["CRPIX3"]
    crval = hdr["CRVAL3"]
    cdelt = hdr["CDELT3"]
    freq = crval + cdelt * (np.arange(nv) + 1 - crpix)  # Hz
    f_ref = float(np.median(freq))
    c_kms = 299792.458
    vel = c_kms * (f_ref - freq) / f_ref
    return vel, freq, f_ref


def interpolate_template_to_fullwindow(target_path, template, template_header,
                                        template_vel, out_path=None):
    """Interpolate the residual template onto a full-window FREQ cube.
    Target velocities are computed relative to the cube's median frequency;
    the template's v=V0 feature lands at pseudo-velocity=V0 in the target.
    """
    thdr = fits.getheader(target_path)
    if "FREQ" not in str(thdr.get("CTYPE3", "")).upper():
        return None, None, None
    tnv = thdr["NAXIS3"]
    t_vel, t_freq, f_ref = _pseudo_velocity_axis(thdr)
    tny = thdr["NAXIS2"]
    tnx = thdr["NAXIS1"]
    wcs_cel_t = WCS(thdr).celestial
    yi, xi = np.mgrid[0:tny, 0:tnx]
    t_ra, t_dec = wcs_cel_t.wcs_pix2world(xi, yi, 0)
    wcs_cel_s = WCS(template_header).celestial
    s_x, s_y = wcs_cel_s.wcs_world2pix(t_ra, t_dec, 0)
    s_dv = template_vel[1] - template_vel[0]
    s_cstart = template_vel[0]
    tpl_shape = template.shape
    target_template = np.zeros((tnv, tny, tnx), dtype=np.float32)
    coords = np.vstack([s_y.ravel(), s_x.ravel()])
    for iv in range(tnv):
        tv = t_vel[iv]
        s_chan = (tv - s_cstart) / s_dv
        if s_chan < -0.5 or s_chan > tpl_shape[0] - 0.5:
            continue
        c0 = int(np.floor(s_chan)); c1 = c0 + 1
        frac = s_chan - c0
        if c0 < 0:
            c0 = 0; frac = 0.0; c1 = 0
        if c1 >= tpl_shape[0]:
            c1 = tpl_shape[0] - 1; c0 = c1; frac = 0.0
        sl0 = map_coordinates(template[c0], coords, order=1, mode="constant",
                              cval=0.0).reshape(tny, tnx)
        if c1 == c0:
            target_template[iv] = sl0
        else:
            sl1 = map_coordinates(template[c1], coords, order=1, mode="constant",
                                  cval=0.0).reshape(tny, tnx)
            target_template[iv] = (1 - frac) * sl0 + frac * sl1
    if out_path:
        out_hdr = thdr.copy()
        for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
            if k in out_hdr:
                del out_hdr[k]
        out_hdr["NAXIS"] = 3
        fits.writeto(out_path, target_template, out_hdr, overwrite=True)
    return target_template, t_vel, t_freq


def run_residual_mf_fullwindow(cube_path, template, template_header, template_vel):
    """Residual-template matched filter on a full-window FREQ cube.  Slides
    the interpolated template along the pseudo-velocity (and thus frequency)
    axis and returns the MF SNR spectrum with frequency tagged."""
    print(f"\n[resid-mf fullwin] {os.path.basename(cube_path)}")
    thdr = fits.getheader(cube_path)
    if "FREQ" not in str(thdr.get("CTYPE3", "")).upper():
        print("   skipped (not FREQ axis)")
        return None
    tpl_out = cube_path.replace(".fits", ".residtpl_fullwin.fits")
    target_tpl, t_vel, t_freq = interpolate_template_to_fullwindow(
        cube_path, template, template_header, template_vel, out_path=tpl_out)
    if target_tpl is None:
        return None
    data = fits.getdata(cube_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    if data.shape != target_tpl.shape:
        print(f"   shape mismatch {data.shape} vs {target_tpl.shape}")
        return None
    if np.nanmax(target_tpl) <= 0:
        print("   template is zero on this cube")
        return None
    # Off-channels for noise estimate: outside ±250 km/s of V0 in pseudo-v
    # frame (handles bulk of the window for 10kms cubes).
    off_mask = np.abs(t_vel - V0) > 500.0
    if off_mask.sum() < 20:
        off_mask = np.abs(t_vel - V0) > 300.0
    spec, sigma, i_center = matched_filter_spectrum(data, target_tpl, off_mask=off_mask)
    snr_spec = spec / sigma if sigma > 0 else np.full_like(spec, np.nan)
    imax = int(np.argmax(np.where(np.isfinite(snr_spec), snr_spec, -np.inf)))
    print(f"   dv/chan={t_vel[1]-t_vel[0]:.3f} km/s, f_range={t_freq.min()/1e9:.3f}"
          f"..{t_freq.max()/1e9:.3f} GHz")
    print(f"   sigma={sigma:.3e}, peak SNR={snr_spec[imax]:.2f} at "
          f"v={t_vel[imax]:.1f} km/s (f={t_freq[imax]/1e9:.4f} GHz)")

    out_path = cube_path.replace(".fits", ".resid_fullwin_mf.fits")
    cols = [
        fits.Column(name="frequency_hz", format="D", array=t_freq),
        fits.Column(name="pseudo_vel_kms", format="D", array=t_vel),
        fits.Column(name="flux", format="D", array=spec),
        fits.Column(name="snr", format="D", array=snr_spec),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)

    def safe(v):
        if not np.isfinite(v):
            return 0.0
        return float(v)

    hdu.header["SIGMA"] = (safe(sigma), "off-signal stdev of flux spectrum")
    hdu.header["SNRMAX"] = (safe(snr_spec[imax]), "peak SNR in spectrum")
    hdu.header["VSNRMAX"] = (safe(t_vel[imax]), "velocity at peak SNR, km/s")
    hdu.header["FSNRMAX"] = (safe(t_freq[imax]), "frequency at peak SNR, Hz")
    hdu.header["TEMPLATE"] = ("SO32_residual_snrexpand_smoothed",
                              "residual-based bow-shock template")
    hdu.writeto(out_path, overwrite=True)
    print(f"   wrote {out_path}")
    return {"vel": t_vel, "freq": t_freq, "flux": spec, "snr": snr_spec, "sigma": sigma}


def fit_cs21_shape_locked():
    """Fit CS(2-1) with SO's spatial + velocity shape; amplitude + offset free."""
    print("\n[cs fit] amplitude-only refit of CS(2-1) with SO32 shape")
    data = fits.getdata(CS21_CUBE).astype(np.float64)
    hdr = fits.getheader(CS21_CUBE)
    while data.ndim > 3:
        data = data[0]
    nv, ny, nx = data.shape
    vel = spectral_axis_kms(hdr, nv)
    wcs_cel = WCS(hdr).celestial
    yi, xi = np.mgrid[0:ny, 0:nx]
    ra2d, dec2d = wcs_cel.wcs_pix2world(xi, yi, 0)

    # Evaluate unit-amplitude SO-shape template on CS grid
    unit_tpl = gauss3d_on_grid((1.0, X0, Y0, SX, SY, V0, SV, 0.0),
                                ra2d, dec2d, vel).astype(np.float32)

    off = np.abs(vel - V0) > 200.0
    if off.sum() < 4:
        off = np.abs(vel - V0) > 150.0
    data_finite = np.where(np.isfinite(data), data, 0.0)
    # Least-squares for amplitude A and offset B: data ≈ A*tpl + B
    tpl_flat = unit_tpl.ravel().astype(np.float64)
    d_flat = data_finite.ravel()
    ones = np.ones_like(tpl_flat)
    # Normal equations
    M = np.array([[np.dot(tpl_flat, tpl_flat), np.dot(tpl_flat, ones)],
                  [np.dot(ones, tpl_flat), np.dot(ones, ones)]])
    y = np.array([np.dot(tpl_flat, d_flat), np.dot(ones, d_flat)])
    A, B = np.linalg.solve(M, y)
    print(f"   CS peak amp (SO-shape) = {A:.4e} Jy/beam,  offset={B:.3e}")
    print(f"   SO peak amp (from fit) = {SO_PARAMS[0]:.4e} Jy/beam")
    print(f"   CS/SO peak ratio = {A/SO_PARAMS[0]:.3f}")

    model = (A * unit_tpl + B).astype(np.float32)
    residual = (data.astype(np.float32) - model)

    out_hdr = hdr.copy()
    for k in ("NAXIS4", "CTYPE4", "CRVAL4", "CDELT4", "CRPIX4", "CUNIT4"):
        if k in out_hdr:
            del out_hdr[k]
    out_hdr["NAXIS"] = 3
    out_hdr["HISTORY"] = "CS(2-1) model: SO32 shape, amp+offset only refit"

    model_path = os.path.join(BASE, "CS21_model3D_gaussian_solocked.fits")
    resid_path = os.path.join(BASE, "CS21_residual3D_gaussian_solocked.fits")
    fits.writeto(model_path, model, out_hdr, overwrite=True)
    fits.writeto(resid_path, residual, out_hdr, overwrite=True)
    print(f"   wrote {model_path}")
    print(f"   wrote {resid_path}")

    res_rms = float(np.nanstd(residual[off]))
    res_rms_on = float(np.nanstd(residual[~off]))
    print(f"   CS residual rms (off-line): {res_rms:.3e}")
    print(f"   CS residual rms (near-line): {res_rms_on:.3e}")
    return residual, hdr, vel


if __name__ == "__main__":
    # Build SO-only residual template (kept for comparison)
    template_so, t_hdr_so, t_vel_so = build_residual_template(
        SO32_RESIDUAL, os.path.join(BASE, "SO32_residual"),
    )

    # Fit CS with SO shape locked to produce a CS residual cube
    fit_cs21_shape_locked()

    # Combined SO+CS residual template (higher S/N)
    cs_resid_path = os.path.join(BASE, "CS21_residual3D_gaussian_solocked.fits")
    template, t_hdr, t_vel, bow_profile = build_combined_residual_template(
        SO32_RESIDUAL, cs_resid_path,
        os.path.join(BASE, "SOCS_bowshock"),
    )

    # Run the residual MF on the 8 full spectral windows using the combined
    # template.
    full_window_cubes = (
        sorted(glob.glob(os.path.join(BASE, "b3.spw*.cube.I.pbcor.10kms.fits")))
        + sorted(glob.glob(os.path.join(BASE, "b7", "*.cube.I.selfcal.pbcor.10kms.fits")))
        + sorted(glob.glob(os.path.join(BASE, "b9", "*.cube.I.selfcal.pbcor.10kms.fits")))
    )
    print(f"\n=== Combined residual template MF on {len(full_window_cubes)} full windows ===")
    for cp in full_window_cubes:
        run_residual_mf_fullwindow(cp, template, t_hdr, t_vel)
