#!/usr/bin/env python3
"""
The MUBLO from 2 um to 3 mm: ALMA continuum, JWST NIRCam, and the two
brightest molecular lines, on one page.

Layout (4 columns x 2 rows):

    Band 3   Band 7   Band 9     SO 3(2)-2(1) moment 0
    F212N    F480M    F212N+F480M  CS 2-1 moment 0

The top three panels repeat the continuum row of ``make_morphology_figure.py``
at native joint resolution, so the arc and the interior source are visible.
The second row puts the JWST NIRCam mosaics on the same sky at the same scale:
F212N is a 2.12 um narrow band dominated by starlight and F480M a 4.8 um medium
band, the pair used for the extinction measurement in ``jwst_cmd_onoff.py``.
The third panel of that row is an RGB composite of the two, following the
480-mean-212 convention of the survey's own mosaic previews: red is F480M, blue
is F212N and green their mean, all on the same linear VMIN..VMAX cut.

The two line panels are the highest signal-to-noise SO and CS transitions
available (peak moment-0 S/N 44 and 29, against 9-10 for SO 8(8)-7(7), CS 7-6
and CS 13-12).  They are deliberately NOT masked: the background noise is left
in so the reader can judge the extent of the emission against it rather than
against a threshold chosen here.

The figure carries no title and no per-panel captions: each panel is labelled
in its top-right corner with its wavelength or line alone.  Panels are butted
edge to edge with tick marks only on the outer boundary of the grid, one pair
of axis labels for the whole figure, and thin magenta crosshairs marking the
MUBLO position in every panel.  Colour bars are omitted, since every panel is
scaled to its own range and a bar per panel would break the grid; the beam or
PSF marker and the 0.5" scale bar carry the quantitative information that
remains.

Every panel uses one colour map, the grey-to-heat scale used elsewhere in this
project's figures: the lower half is ``gray_r``, so faint background runs white
through grey to black, and the upper half is ``hot``, so bright structure runs
black through red and orange to white.  Sharing it across panels means a colour
means the same fraction of each panel's own range everywhere.

Scaling: the mm panels run to their actual peak rather than to a percentile, so
the bright interior is not clipped flat.  The JWST panels run to a fixed
VMIN..VMAX cut well below their stellar peaks, which deliberately saturates the
stars so the faint nebulosity between them is visible.

Caveats

*   F212N and F480M are registered independently and sit ~0.05-0.13" apart
    (measured in ``jwst_cmd_onoff.align``).  The offset is not corrected here;
    at the 1.6" field of these panels it is a small fraction of the box but it
    is comparable to the F212N PSF, so do not read a 0.1" positional difference
    between the two bands as real.
*   The JWST pixel scales (0.031" and 0.063") are far finer than the 4.8 um
    PSF, and the ALMA panels are at three different native beams.  Each panel
    carries its own beam or PSF marker.
*   The JWST stars are saturated by choice of stretch, not by the detector.
    Do not read relative brightness off those panels; the catalogue photometry
    in ``jwst_cmd_onoff.py`` is the measurement.

Outputs
    png_figures/MUBLO_multiwavelength_panels.{png,pdf}

Usage:
    python make_multiwavelength_panels.py
    python make_multiwavelength_panels.py --size 2.0 --lines SO32,CS21
    python make_multiwavelength_panels.py --jwst-vmax 300
"""
import argparse
import os
import sys

import numpy as np
import radio_beam
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from reproject import reproject_interp

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import figure_style as fs  # noqa: E402
import make_matched_cubes as mmc  # noqa: E402

fs.apply_style()

COORD = SkyCoord(config.MUBLO_RA_DEG * u.deg, config.MUBLO_DEC_DEG * u.deg,
                 frame="icrs")
FIGDIR = os.path.join(config.ROOT, "png_figures")
MOSAICS = "/orange/adamginsburg/jwst/gc-treasury/mosaics"

#: Half-size of every panel.  The source is ~0.7" across, so +/-1.5" frames it
#: with enough background to judge the noise against.
SIZE = 1.5 * u.arcsec

#: Panel labels are the wavelength alone.  92.7, 337 and 641 GHz are 3.2 mm,
#: 890 um and 468 um; the round numbers are the conventional band names.
CONT = [("3 mm", "B3"), ("850 $\\mu$m", "B7"), ("450 $\\mu$m", "B9")]
JWST = [("2.12 $\\mu$m", "f212n_mosaic.fits", 0.07),
        ("4.8 $\\mu$m", "f480m_mosaic.fits", 0.16)]

#: Highest signal-to-noise SO and CS transitions in the matched cubes.
LINEPANELS = [("SO32", "SO 3(2)-2(1)"), ("CS21", "CS 2-1")]

#: Font sizes.  The panels are small, so the few bits of text on them are set
#: large enough to survive a two-column reduction.
FS_LABEL, FS_TICK, FS_AXIS = 16, 12, 17

#: Galactic centre distance, for the physical scale bar.  Matches the value
#: used in ``kinematics.py``.
DISTANCE = 8.2 * u.kpc
SCALEBAR_AU = 4000.0

#: Crosshair marking the MUBLO position: four diagonal segments with a gap
#: across the centre, so the marker does not cover the source it points at.
#: Rotated 45 degrees off the axes so it is not confused with a grid line.
CROSS_GAP, CROSS_LEN, CROSS_COLOUR = 0.09, 0.24, "magenta"

#: Velocity window for the moment, and the line-free region outside it.
V_SYS, V_WINDOW, V_LINEFREE = 45.0, 220.0, 250.0

#: Fixed linear cut for the JWST panels, in MJy/sr.  Chosen well below the
#: stellar peaks (1392 and 631 MJy/sr in this field) so the stars saturate and
#: the faint emission between them is visible.
JWST_VMIN, JWST_VMAX = -0.5, 100.0

#: asinh softening parameter for the JWST panels, and for the RGB composite,
#: which reproduces the same transfer function by hand.  The astropy default
#: (0.1) is hard enough that the 28 MJy/sr F480M background lands above the
#: grey-to-heat join and the whole panel goes orange; 0.5 lifts the faint
#: emission off the floor while leaving both backgrounds in the grey half.
ASINH_A = 0.5


def grey_heat(n=128):
    """The grey-to-heat map used across this project's figures.

    Lower half ``gray_r``: white background darkening to black.  Upper half
    ``hot``: black through red and orange to white.  The join sits at the
    midpoint of whatever normalisation the panel uses.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "grey_heat",
        np.vstack((plt.cm.gray_r(np.linspace(0.0, 1.0, n)),
                   plt.cm.hot(np.linspace(0.0, 1.0, n)))))


CMAP = grey_heat()


def cutout(path, size, hdu_index=0):
    """A square cutout about the MUBLO, read without loading the full mosaic."""
    with fits.open(path, memmap=True) as hl:
        hdu = hl[hdu_index]
        w = WCS(hdu.header).celestial
        # CASA images carry degenerate Stokes and frequency axes; squeeze is a
        # view on the memmap, so this still does not read the whole mosaic.
        cut = Cutout2D(np.squeeze(hdu.data), COORD, 2 * size, wcs=w, copy=True)
    return cut.data.astype("float64"), cut.wcs


def extent_of(wcs, data):
    """Arcsec offsets from the MUBLO, east to the left."""
    x0, y0 = wcs.world_to_pixel(COORD)
    scale = abs(wcs.proj_plane_pixel_scales()[0].to(u.arcsec)).value
    ny, nx = data.shape[:2]      # an RGB array carries a trailing colour axis
    # east is +x in offset space, and RA decreases with pixel x, so the left
    # edge of the panel is the largest positive offset
    return [(x0 + 0.5) * scale, (x0 + 0.5 - nx) * scale,
            -(y0 + 0.5) * scale, (ny - y0 - 0.5) * scale]


def show(ax, data, wcs, label, size, norm_kw=None, rgb=False):
    """One panel: image, corner label, crosshair.  No title, no colour bar.

    ``rgb`` takes an (ny, nx, 3) array and skips the normalisation.
    """
    ext = extent_of(wcs, data)
    if rgb:
        im = ax.imshow(data, origin="lower", extent=ext)
    else:
        norm = simple_norm(data[np.isfinite(data)], **norm_kw)
        im = ax.imshow(data, origin="lower", cmap=CMAP, norm=norm, extent=ext)
    s = size.to(u.arcsec).value
    ax.set_xlim(s, -s)
    ax.set_ylim(-s, s)
    ax.set_aspect("equal")
    ax.grid(False)          # the project style turns a grid on; not on images
    if label:
        # white with a black outline, so the label reads on either end of the
        # grey-to-heat scale
        ax.text(0.96, 0.94, label, transform=ax.transAxes, ha="right",
                va="top", color="w", fontsize=FS_LABEL, zorder=7,
                path_effects=[withStroke(linewidth=2.5, foreground="k")])
    crosshair(ax)
    return im


def crosshair(ax, colour=CROSS_COLOUR):
    """Four thin diagonal segments pointing at 0,0, with a central gap."""
    d = np.sqrt(0.5)          # 45 degrees: equal steps along both axes
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([sx * d * CROSS_GAP, sx * d * CROSS_LEN],
                    [sy * d * CROSS_GAP, sy * d * CROSS_LEN],
                    color=colour, lw=0.8, zorder=7)


def beam_patch(ax, beam, colour="w"):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[0] + 0.12 * (xlim[1] - xlim[0])
    y = ylim[0] + 0.12 * (ylim[1] - ylim[0])
    ax.add_patch(Ellipse((x, y), beam.minor.to(u.arcsec).value,
                         beam.major.to(u.arcsec).value,
                         angle=beam.pa.to(u.deg).value,
                         facecolor=colour, edgecolor="k", lw=0.6, zorder=6))


def psf_patch(ax, fwhm, colour="w"):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[0] + 0.12 * (xlim[1] - xlim[0])
    y = ylim[0] + 0.12 * (ylim[1] - ylim[0])
    ax.add_patch(Circle((x, y), 0.5 * fwhm, facecolor=colour, edgecolor="k",
                        lw=0.6, zorder=6))


def scalebar(ax, au=SCALEBAR_AU, colour="w"):
    """A physical scale bar.  The arcsecond scale is already on the axes."""
    length = float((au * u.au / DISTANCE).to(u.arcsec,
                                             u.dimensionless_angles()).value)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = xlim[1] + 0.10 * (xlim[0] - xlim[1])
    y = ylim[0] + 0.09 * (ylim[1] - ylim[0])
    stroke = [withStroke(linewidth=2.6, foreground="k")]
    ax.plot([x, x - length * np.sign(xlim[1] - xlim[0])], [y, y], color=colour,
            lw=2.6, solid_capstyle="butt", zorder=7, path_effects=stroke)
    ax.text(x - 0.5 * length * np.sign(xlim[1] - xlim[0]),
            y + 0.03 * (ylim[1] - ylim[0]), "%.0f AU" % au, color=colour,
            ha="center", va="bottom", fontsize=FS_LABEL - 2, zorder=7,
            path_effects=stroke)


def moment0(name):
    """Unmasked moment 0 of a matched cube, with its noise."""
    path = os.path.join(mmc.OUTDIR, "%s_matched.fits" % name)
    hdu = fits.open(path)[0]
    data = np.squeeze(hdu.data).astype("float64")
    hdr = hdu.header
    vel = hdr["CRVAL3"] + (np.arange(data.shape[0]) + 1
                           - hdr["CRPIX3"]) * hdr["CDELT3"]
    linefree = np.abs(vel - V_SYS) > V_LINEFREE
    data = data - np.median(data[linefree], axis=0)[None]
    window = np.abs(vel - V_SYS) < V_WINDOW
    dv = abs(hdr["CDELT3"])
    m0 = data[window].sum(axis=0) * dv
    rms = float(np.median(np.std(data[linefree], axis=0))
                * dv * np.sqrt(window.sum()))
    return m0, WCS(hdr).celestial, rms, radio_beam.Beam.from_fits_header(hdr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--size", type=float, default=SIZE.to(u.arcsec).value,
                    help="half-size of every panel, arcsec")
    ap.add_argument("--lines", default=",".join(n for n, _ in LINEPANELS),
                    help="comma-separated matched-cube names for the two "
                         "line panels")
    ap.add_argument("--jwst-vmin", type=float, default=JWST_VMIN,
                    help="lower cut for the JWST panels, MJy/sr")
    ap.add_argument("--jwst-vmax", type=float, default=JWST_VMAX,
                    help="upper cut for the JWST panels, MJy/sr; low values "
                         "saturate the stars on purpose")
    args = ap.parse_args()
    size = args.size * u.arcsec
    names = [n for n in args.lines.split(",") if n]

    os.makedirs(FIGDIR, exist_ok=True)
    # butted panels: shared axes, no gaps, tick marks only on the outside
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.9), sharex=True,
                             sharey=True,
                             gridspec_kw=dict(wspace=0.0, hspace=0.0))

    # ---- row 1, columns 1-3: ALMA continuum at native joint resolution
    print("=== ALMA continuum")
    for i, (label, band) in enumerate(CONT):
        path = mmc.CONTINUUM[band]["path"]
        data, w = cutout(path, size)
        data = np.squeeze(data) * 1e3        # Jy/beam -> mJy/beam
        hdr = fits.getheader(path)
        beam = radio_beam.Beam.from_fits_header(hdr)
        ax = axes[0, i]
        # max_percent=100 rather than a percentile: clipping the top 0.3%
        # flattens the interior source, which is the structure of interest.
        show(ax, data, w, label, size,
             norm_kw=dict(stretch="asinh", min_percent=1.0, max_percent=100.0))
        beam_patch(ax, beam)
        if i == 0:
            scalebar(ax)
        print("   %-26s peak %8.3f mJy/beam, beam %.3f x %.3f\""
              % (band, np.nanmax(data), beam.major.to(u.arcsec).value,
                 beam.minor.to(u.arcsec).value))

    # ---- row 2, columns 1-3: JWST F212N, F480M, and their RGB composite
    print("\n=== JWST NIRCam")
    stretched = []
    for i, (label, fname, fwhm) in enumerate(JWST):
        data, w = cutout(os.path.join(MOSAICS, fname), size)
        ax = axes[1, i]
        # asinh on the fixed cut: the stars still saturate, but the faint
        # nebulosity between them is lifted well off the floor.
        show(ax, data, w, label, size,
             norm_kw=dict(stretch="asinh", asinh_a=ASINH_A,
                          vmin=args.jwst_vmin, vmax=args.jwst_vmax))
        psf_patch(ax, fwhm, colour="#ffd27f")
        print("   %-26s %d x %d pixels, peak %.1f MJy/sr, PSF %.2f\""
              % (fname, *data.shape, np.nanmax(data), fwhm))
        stretched.append((data, w))

    # RGB composite on the F212N grid, following the 480-mean-212 convention
    # of the survey mosaic previews.  Both bands take the same cut, so a red
    # source is genuinely brighter at 4.8 um than at 2.12 um relative to that
    # cut rather than to its own stretch.
    (d212, w212), (d480, w480) = stretched
    hdr212 = w212.to_header()
    hdr212["NAXIS"] = 2
    hdr212["NAXIS1"], hdr212["NAXIS2"] = d212.shape[1], d212.shape[0]
    d480r, _ = reproject_interp((d480, w480), hdr212, shape_out=d212.shape)

    def cut(a):
        # same asinh as the single-band panels, so the three panels of the
        # bottom row share one transfer function
        x = np.clip((a - args.jwst_vmin)
                    / (args.jwst_vmax - args.jwst_vmin), 0.0, 1.0)
        return np.arcsinh(x / ASINH_A) / np.arcsinh(1.0 / ASINH_A)

    red, blue = cut(d480r), cut(d212)
    green = 0.5 * (red + blue)
    rgb = np.nan_to_num(np.dstack([red, green, blue]))
    ax_rgb = axes[1, 3 - 1]
    show(ax_rgb, rgb, w212, "", size, rgb=True)
    # name the two channels in their own colours, so the composite reads
    # without the caption: red is the 4.8 um band, blue the 2.12 um one
    for k, (text, colour) in enumerate([("4.8 $\\mu$m", "#ff8a3d"),
                                        ("2.12 $\\mu$m", "#6fb7ff")]):
        ax_rgb.text(0.96, 0.94 - 0.105 * k, text, transform=ax_rgb.transAxes,
                    ha="right", va="top", color=colour, fontsize=FS_LABEL,
                    zorder=7,
                    path_effects=[withStroke(linewidth=2.5, foreground="k")])
    nfin = np.isfinite(d480r).sum()
    print("   RGB on the F212N grid, %d of %d reprojected pixels finite; "
          "cut %.1f to %.0f MJy/sr" % (nfin, d480r.size, args.jwst_vmin,
                                       args.jwst_vmax))

    # ---- column 4: the two brightest lines, unmasked
    print("\n=== lines (unmasked: the background noise is left in)")
    for i, name in enumerate(names[:2]):
        m0, w, rms, beam = moment0(name)
        ax = axes[i, 3]
        show(ax, m0, w, dict(LINEPANELS).get(name, name), size,
             norm_kw=dict(stretch="asinh", min_percent=0.2,
                          max_percent=100.0))
        beam_patch(ax, beam)
        print("   %-10s peak %7.1f K km/s, rms %5.1f, peak S/N %.1f"
              % (name, np.nanmax(m0), rms, np.nanmax(m0) / rms))

    # ---- one set of axis labels, ticks on the outer boundary only
    ticks = [t for t in (-1.0, 0.0, 1.0) if abs(t) < args.size]
    for (r, c), ax in np.ndenumerate(axes):
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(direction="out", labelsize=FS_TICK, length=4,
                       top=False, right=False,
                       bottom=(r == axes.shape[0] - 1), left=(c == 0),
                       labelbottom=(r == axes.shape[0] - 1),
                       labelleft=(c == 0))
    fig.supxlabel(r"$\Delta\alpha$ ['']", fontsize=FS_AXIS, y=0.018)
    fig.supylabel(r"$\Delta\delta$ ['']", fontsize=FS_AXIS, x=0.012)
    fig.subplots_adjust(left=0.062, right=0.996, bottom=0.105, top=0.996,
                        wspace=0.0, hspace=0.0)
    out = os.path.join(FIGDIR, "MUBLO_multiwavelength_panels")
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    print("\n=== wrote %s.{png,pdf}" % out)


if __name__ == "__main__":
    main()
