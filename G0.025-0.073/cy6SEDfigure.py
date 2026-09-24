"""
Proposal Cycle 6 figure
"""
import os
import sys

from astropy import units as u
import astroquery
import astroquery.herschel.higal
import pylab as pl
pl.rcParams['figure.dpi'] = 200
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.colors as mcolors
import numpy as np
import dust_emissivity
from astropy.visualization import simple_norm
from astropy.table import Table
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

if 'ulimtbl' not in locals():
    ulimtbl = Table.read('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED.ecsv')


from astropy.table import Table
centrfittbl = Table.read('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/centroid_fits.ecsv')
centrfittbl.add_index(['Image Data', 'Image Type'])

import glob
from spectral_cube import SpectralCube

path = '/orange/adamginsburg/ACES/data/2021.1.00172.L/science_goal.uid___A001_X1590_X30a8/group.uid___A001_X1590_X30a9/member.uid___A001_X15a0_X13c/calibrated/working/'
imgs = sorted(glob.glob(f"{path}/*.cube.*.image"))
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    cubes = [SpectralCube.read(fn) for fn in imgs[:1]]
try:
    b3beam = cubes[0].beams.common_beam()
except AttributeError:
    b3beam = cubes[0].beam

# JWST Proposal Figure

# dust_beta_map.py
#    B3     92.7 GHz       2.05 +/-  0.12 mJy
#   B7    337.0 GHz      81.57 +/-  8.18 mJy
#   B9    641.0 GHz     302.42 +/- 60.55 mJy
b3flx = (2.05*u.mJy).to(u.Jy)
b7flx = (81.57*u.mJy).to(u.Jy)
b9flx = (302*u.mJy).to(u.Jy)
b3wl = (92.7*u.GHz).to(u.um, u.spectral())
b7wl = (337*u.GHz).to(u.um, u.spectral())
b9wl = (641*u.GHz).to(u.um, u.spectral())

pl.figure(figsize=(9,4.5))
ax = pl.subplot()
# inset axes....
x1, x2, y1, y2 = 400, 1000, 0.04, 0.6  # subregion of the original image


ulwl = ulimtbl['Wavelength']
ax.plot(ulwl[ulwl < 800*u.um], ulimtbl['Flux'][ulwl < 800*u.um], 'v', markerfacecolor='none', markeredgecolor='k')
ax.plot(ulwl[ulwl > 1*u.cm], ulimtbl['Flux'][ulwl > 1*u.cm], 'v', markerfacecolor='none', markeredgecolor='k')
ax.plot(b3wl, b3flx, 's', markeredgecolor='k', markerfacecolor='b')
ax.plot(b7wl, b7flx, 's', markeredgecolor='k', markerfacecolor='b')
ax.plot(b9wl, b9flx, 's', markeredgecolor='k', markerfacecolor='b')

axins = ax.inset_axes(
    bounds=[0.65, 0.45, 0.3, 0.25],
    xlim=(x1, x2), ylim=(y1, y2),
    #xticklabels=[], yticklabels=[]
)

temperatures = [10, 15, 20, 25]

nurange = np.geomspace(10*u.GHz, 100*u.THz, 1000)
beta = 1.13 # see https://data.rc.ufl.edu/secure/adamginsburg/ACES/broadline_sources/G0.025-0.073/png_figures/MUBLO_dust_alpha_beta_maps.png
for tem, ls in zip(temperatures, ('-', '--', '-.', ':')):
    bm = b3beam.sr
    column = dust_emissivity.dust.colofsnu(nu=b3wl.to(u.GHz, u.spectral()),
                                           snu_per_beam=b3flx/bm,
                                           beta=beta,
                                           nu0=271.1*u.GHz, 
                                           kappa0=0.0114*u.cm**2*u.g**-1,
                                           temperature=tem*u.K)
    mbb = dust_emissivity.blackbody.modified_blackbody(nurange,
                                                       temperature=tem*u.K,
                                                       beta=beta,
                                                       nu0=271.1*u.GHz, 
                                                       kappa0=0.0114*u.cm**2*u.g**-1*100,
                                                       column=column,
                                                      ).to(u.Jy/u.sr)*bm
    L, = ax.plot(nurange.to(u.um, u.spectral()), mbb.to(u.Jy), linestyle=ls, zorder=-5, label=f'T={tem} K')#\nN={column.value:0.1e} cm$^{{-2}}$')
    axins.plot(nurange.to(u.um, u.spectral()), mbb.to(u.Jy), linestyle=ls, zorder=-5, 
               color=L.get_color(),
               label=f'T={tem} K')#\nN={column.value:0.1e} cm$^{{-2}}$')


# The infrared background of the general region, per JWST pixel.  ISO SWS01
# saw the diffuse Galactic-centre emission in an aperture of 14x20 to 20x33
# arcsec, so dividing by that aperture gives a surface brightness; multiplying
# by the solid angle of one JWST pixel gives what a spectrum of uniform
# background would contribute to a single spaxel, which is what has to be
# subtracted from, and sets the photon noise on, any point-source measurement.
# The curve is drawn over the MIRI MRS range alone, 4.93-28.2 um, using the
# spaxel of whichever MRS channel covers each wavelength.  Continuing it into
# the NIRSpec IFU (0.1" spaxel) or onto the MIRI imager (0.11" pixel) puts a
# factor 3.8 step at 5.3 um and a factor 6.2 step at 28 um into the curve,
# which read as spectral features and are only the pixel area changing.  The
# SWS grating scan is spiky at the 10% level from order joins and imperfect
# dark subtraction; a running median takes those out without moving the
# continuum.
sys.path.insert(0, os.path.join(
    '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073', 'ir_background'))
import ir_background_spectrum as irb
from scipy.ndimage import median_filter

#: (upper wavelength [um], spaxel size ["]) for the four MIRI MRS channels;
#: Ch1 and Ch2 share 0.196".
JWST_PIXEL = [(11.70, 0.196), (17.98, 0.245), (1e4, 0.273)]
#: MRS wavelength coverage, the range the background curve is drawn over
MRS_RANGE = (4.93, 28.2)
ISO_MEDIAN = 101     # samples; the SWS grid is ~50000 points over 2.4-45 um


def jwst_pixel_arcsec(wave_um):
    """Spaxel size, arcsec, of whichever MRS channel covers each wavelength."""
    out = np.full(np.shape(wave_um), JWST_PIXEL[-1][1])
    for hi, pix in JWST_PIXEL[::-1]:
        out[np.asarray(wave_um) < hi] = pix
    return out


sws = irb.load_sws()
if sws is None:
    print("ISO SWS spectrum not found; skipping the background curve")
else:
    iso_wave, iso_sb = sws                       # um, MJy/sr
    iso_sb = median_filter(iso_sb, size=ISO_MEDIAN, mode="nearest")
    keep = (iso_wave >= MRS_RANGE[0]) & (iso_wave <= MRS_RANGE[1])
    iso_wave, iso_sb = iso_wave[keep], iso_sb[keep]
    omega_pix = ((jwst_pixel_arcsec(iso_wave) * u.arcsec) ** 2).to(u.sr).value
    iso_jy = iso_sb * 1e6 * omega_pix            # MJy/sr -> Jy per pixel
    ax.plot(iso_wave, iso_jy, color='red', alpha=0.5, linewidth=1.5,
            zorder=-3)
    # labelled in place rather than in the legend, which is already four
    # modified blackbodies long
    itarg = np.argmin(np.abs(iso_wave - 5.6))
    ax.annotate('ISO background', xy=(iso_wave[itarg]-0.1, iso_jy[itarg]),
                xytext=(1.35, 1.5e-3), color='red', ha='center', va='center',
                fontsize=12, weight='bold',
                arrowprops=dict(arrowstyle='->', edgecolor='red',
                                facecolor='red', shrinkB=4,
                                connectionstyle='arc3,rad=-0.2'))
    print("ISO background per MRS spaxel: %.3g Jy at 6 um, %.3g at 10 um, "
          "%.3g at 20 um" % tuple(iso_jy[np.argmin(np.abs(iso_wave - w))]
                                  for w in (6., 10., 20.)))

#ax.indicate_inset_zoom(axins, edgecolor="black", )
def mark_inset_behind(parent, inset, loc1, loc2, **kwargs):
    """mark_inset, with the connectors drawn behind the inset.

    mark_inset adds the two connector patches to the INSET axes with clipping
    turned off, so wherever a connector crosses the inset it paints over it --
    an axes draws its own background before any of its children, so no zorder
    on the connector can put it underneath.  Reparenting the connectors to the
    parent axes lets the inset, which is drawn later and at a higher zorder,
    cover the parts that fall inside it.  Their transform is display
    coordinates, so the reparenting leaves the geometry untouched.
    """
    pp, p1, p2 = mark_inset(parent, inset, loc1=loc1, loc2=loc2, **kwargs)
    for p in (p1, p2):
        p.remove()
        p.set_clip_on(False)
        parent.add_patch(p)
    inset.set_zorder(max(inset.get_zorder(), parent.get_zorder() + 1))
    return pp, p1, p2


blah = mark_inset_behind(ax, axins, loc1=3, loc2=2)
axins.plot(b7wl, b7flx, 's', markeredgecolor='k', markerfacecolor='b')
axins.plot(b9wl, b9flx, 's', markeredgecolor='k', markerfacecolor='b')

# The ALMA label sits in the inset, where the Band 7 and Band 9 points are
# resolved from each other, rather than on the main panel where it lands under
# the legend.  One arrow per point, both from the same text.
alma_label_xy = (950., 0.52)      # top right corner of the inset
alma_arrow_from1 = (800., 0.50) # from the left
alma_arrow_from2 = (950., 0.38) # just below it, so the arrows clear the text
print("doing the loops...")
for wl, flx, rad, alma_arrow_from in ((b9wl, b9flx, 0.2, alma_arrow_from1), (b7wl, b7flx, -0.3, alma_arrow_from2)):
    axins.annotate('', xy=(wl.value, flx.value), xytext=alma_arrow_from,
                   arrowprops=dict(arrowstyle='->', edgecolor='b',
                                   facecolor='b', shrinkA=2, shrinkB=5,
                                   connectionstyle='arc3,rad=%g' % rad))
axins.text(alma_label_xy[0], alma_label_xy[1], 'ALMA', color='b',
           ha='right', va='top', weight='bold')


ax.loglog();
ax.axis([1,1e5,5e-4,300]);
pl.legend(loc='upper right',);
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel("Flux Density $S_\\nu$ [Jy]")
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits.pdf', bbox_inches='tight')
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits.png', bbox_inches='tight')
ax.axis([1,3e5,1e-5,300]);
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits_VLA.pdf', bbox_inches='tight')
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits_VLA.png', bbox_inches='tight')

# SMA?
ax.plot((230*u.GHz).to(u.um, u.spectral()), (26.0*u.mJy).to(u.Jy), 's', markeredgecolor='k', markerfacecolor='r')
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits_VLA_SMA.pdf', bbox_inches='tight')
# pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_upperlimits_VLA_SMA.png', bbox_inches='tight')

axins.loglog()
#axins.semilogx()

# logscale: hide ticks
# holy fucking hackery.  This took absolutely forever to come up with.
axins.yaxis.set_major_formatter(lambda x, y: f"{x:0.2f}")
axins.yaxis.set_minor_formatter(lambda x, y: "")
axins.set_ylim(y1, y2)
axins.yaxis.set_ticks([0.05, 0.1, 0.2, 0.5, ]);

axins.xaxis.set_minor_formatter(lambda x, y: "")
axins.xaxis.set_major_formatter(lambda x, y: f"{int(x):3d}")
axins.xaxis.set_ticks([500,700,1000]);



pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_observed_B9_wide.pdf', bbox_inches='tight')
pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_observed_B9_wide.png', bbox_inches='tight')


# JWST sensitivities, all pulled from JIST (https://jist.stsci.edu/jist) with
# fetch_jist_sensitivity.py.  JIST reports S/N per spectral element for a flat
# f_nu point source at a per-mode default input flux and exposure time, both
# recorded in each JSON, so the n sigma sensitivity is
#     f(n sigma, t) = f_in * n / SN * sqrt(t_default / t),
# which holds while the measurement is background- or detector-noise limited.
# The MRS and NIRSpec IFU grids assume two integrations (a two-point nod), and
# a background of 120% of the minimum zodiacal level.  Every curve here is
# POINT SOURCE sensitivity, matching the point-source photometry in this SED.
import json

JWST_TEXP = 3600.  # seconds on source
jist_dir = '/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/'


def jist_nsigma(fn, config, texp=JWST_TEXP, nsigma=5, trim=1):
    """Wavelength [um] and n sigma point-source sensitivity [Jy] for one config.

    `trim` drops that many points from each end of the grid.  The first and
    last sample of every MRS sub-band and every grating sits on the edge of the
    wavelength solution, where the S/N JIST reports drops abruptly; plotted,
    those points appear as a narrow spike at each band boundary rather than as
    a feature of the sensitivity.
    """
    data = json.load(open(jist_dir + fn))
    cfg = data['configs'][config]
    wave = np.array(cfg['wave_um'])
    sn = np.array(cfg['sn'])
    ok = sn > 0  # JIST reports saturated points as S/N = 0
    wave, sn = wave[ok], sn[ok]
    if trim and wave.size > 2 * trim + 2:
        wave, sn = wave[trim:-trim], sn[trim:-trim]
    flux = (data['input_flux_mjy'] * nsigma / sn
            * np.sqrt(data['texp_s'] / texp))
    return wave, (flux * u.mJy).to(u.Jy)


# MIRI imaging, the three filters that bracket the SED peak
miri_pts = [jist_nsigma('jist_MIRI_Imaging.json', f, trim=0)
            for f in ('F1800W IMAGER', 'F2100W IMAGER', 'F2550W IMAGER')]
miri_wls = np.array([w[0] for w, f in miri_pts]) * u.um
miri_flx = u.Quantity([f[0] for w, f in miri_pts])
dots, = pl.plot(miri_wls, miri_flx, 'o')

mirit = pl.text(63, 7e-7, 'JWST MIRI\n(proposed)', color=dots.get_color(),
                weight='bold', ha='center', fontsize=13, va='center',)
mirit.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='none'))

pl.annotate('', xy=(miri_wls[-1].value, miri_flx[-1].value), xytext=(63, 5.2e-6),
            arrowprops=dict(facecolor=dots.get_color(),
                            connectionstyle="arc3,rad=0.2",
                            arrowstyle='->',
                            shrinkB=8,
                            edgecolor=dots.get_color(),),
            ha='left', va='bottom')


pl.text(300, 50, 'Herschel', color='k', ha='center')
pl.text(25, 2, 'Spitzer', color='k', ha='center')

pl.text(1.5e3, 0.05, 'SMA', color='r', ha='center')

# MIRI MRS: one curve per sub-band, drawn in a single color
mrs_color = 'teal'
for band in ('CH1 SHORT', 'CH1 MEDIUM', 'CH1 LONG',
             'CH2 SHORT', 'CH2 MEDIUM', 'CH2 LONG',
             'CH3 SHORT', 'CH3 MEDIUM', 'CH3 LONG',
             'CH4 SHORT', 'CH4 MEDIUM', 'CH4 LONG'):
    wave, flux = jist_nsigma('jist_MIRI_Medium-Resolution_Spectroscopy.json', band)
    mrs, = pl.plot(wave, flux, linewidth=1, color=mrs_color)

pl.annotate('JWST MRS\n(proposed)', xy=(10, 1), xytext=(5, 10),
            color=mrs_color, weight='bold', ha='center', fontsize=14, va='center')

pl.annotate('', xy=(25, 1e-2), xytext=(10, 2,),
            arrowprops=dict(facecolor=mrs_color,
                            connectionstyle="arc3,rad=0.2",
                            arrowstyle='->',
                            edgecolor=mrs_color,),
            ha='left', va='bottom')

# NIRSpec IFU: the high-resolution gratings (R~2700), which are what this
# programme would use; they cover 0.95-5.3 um across the four filter pairings.
nirspec_color = 'darkviolet'
for config in ('F070LP IFU G140H', 'F100LP IFU G140H',
               'F170LP IFU G235H', 'F290LP IFU G395H'):
    wave, flux = jist_nsigma('jist_NIRSpec_IFU.json', config)
    pl.plot(wave, flux, linestyle='-', linewidth=1, color=nirspec_color,
            zorder=-4)

pl.annotate('JWST NIRSpec IFU\n(proposed)', xy=(1.4, 3e-7), xytext=(0.62, 4e-7),
            color=nirspec_color, weight='bold', ha='left', fontsize=14, va='center')
pl.annotate('', xy=(1.4, 1e-5), xytext=(1.4, 1.2e-6),
            arrowprops=dict(facecolor=nirspec_color,
                            connectionstyle="arc3,rad=0.2",
                            arrowstyle='->',
                            edgecolor=nirspec_color,),
            ha='left', va='bottom')

pl.axis([0.5,5000,1e-7,250]);

from matplotlib.transforms import Bbox
# AnchoredPositionLocator does not exist in matplotlib (InsetPosition, the
# class that did this job, was removed in 3.8).  Setting the inset's position
# directly does the same thing: drop the locator that ax.inset_axes installed,
# then place the axes at new_bounds, expressed in `ax` axes coordinates and
# converted to the figure coordinates set_position expects.
new_bounds = [0.66, 0.07, 0.32, 0.33]
for artist in blah:      # the connectors drawn for the inset's old position
    artist.remove()
axins.set_axes_locator(None)
axins.set_position(Bbox.from_bounds(*new_bounds)
                   .transformed(ax.transAxes)
                   .transformed(ax.figure.transFigure.inverted()))
blah = mark_inset_behind(ax, axins, loc1=3, loc2=4, zorder=-5)
axins.set_facecolor('white')
axins.set_zorder(10)




pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_observed_B9_and_JWST_wide.pdf', bbox_inches='tight')
pl.savefig('/orange/adamginsburg/ACES/broadline_sources/G0.025-0.073/SED_with_observed_B9_and_JWST_wide.png', bbox_inches='tight')
