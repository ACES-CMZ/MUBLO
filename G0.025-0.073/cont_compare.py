import numpy as np
import radio_beam
from spectral_cube import SpectralCube
import regions

# airplane / no wifi hack
from astropy.utils.iers import conf
conf.auto_max_age = None

cube_b7 = SpectralCube.read('b7.spw25_27_29_31.cont.I.pbcor.fits')
cube_b7_selfcal = SpectralCube.read('b7.spw25_27_29_31.cont.I.selfcal.pbcor.fits')
cube_b3 = SpectralCube.read('b3.spw25_27_29_31.cont.I.tt0.pbcor.fits')

sm_b7 = cube_b7.convolve_to(cube_b3.beam)
sm_b7_selfcal = cube_b7_selfcal.convolve_to(cube_b3.beam)

sm_b7.write('b7_cont_smooth_to_b3.fits', overwrite=True)
sm_b7_selfcal.write('b7_selfcal_cont_smooth_to_b3.fits', overwrite=True)

tgt_header = cube_b3.wcs.celestial.to_header()
tgt_header['NAXIS'] = 2
tgt_header['NAXIS1'] = cube_b3.shape[2]
tgt_header['NAXIS2'] = cube_b3.shape[1]
sm_rg_b7 = sm_b7[0].reproject(tgt_header)
ratio = sm_rg_b7 / cube_b3

ratio.write('b7_to_b3_ratio_map.fits', overwrite=True)
alpha = np.log(ratio) / np.log(cube_b7.spectral_axis[0] / cube_b3.spectral_axis[0])
alpha.write('b7_to_b3_alpha_map.fits', overwrite=True)

alpha_as_cube = SpectralCube(alpha, wcs=cube_b3.wcs)
reg = regions.Regions.read('mublo_cutout_circle_tight.reg')
subalpha = alpha_as_cube.subcube_from_regions(reg)

import pylab as pl
from astropy import units as u
pl.close('all')
pl.figure()
ax = pl.subplot(projection=subalpha[0].wcs)
pl.imshow(subalpha[0].value, vmin=2, vmax=4, origin='lower')
cb = pl.colorbar()
cb.set_label("Spectral Index")
ax.coords[0].set_axislabel('Right Ascension (J2000)')
ax.coords[1].set_axislabel('Declination (J2000)')
ax.coords[0].set_major_formatter('hh:mm:ss.ss')
ax.coords[1].set_major_formatter('dd:mm:ss.ss')
ax.coords[0].set_separator((r'$^{\mathrm{h}}$', r'$^{\mathrm{m}}$', r'$^{\mathrm{s}}$'))
ax.coords[1].set_separator((r'$^{\circ}$', r'$^{\prime}$', r'$^{\prime\prime}$'))
ax.coords[0].set_ticks(spacing=0.25*u.arcsec)
ax.coords[1].set_ticks(spacing=0.25*u.arcsec)
ax.coords[0].set_ticklabel(rotation=25, pad=30)
pl.tight_layout()
pl.savefig("alpha_b3_b7.png", bbox_inches='tight')

pl.figure()
pl.hist(subalpha[0].ravel(), bins=np.linspace(1, 5))

import dust_emissivity
from astropy import units as u
b7_scube = cube_b7.subcube_from_regions(reg)
column = dust_emissivity.dust.colofsnu(b7_scube.spectral_axis[0], b7_scube[0].to(u.Jy/u.sr), temperature=50*u.K)
pixel_area = (b7_scube.wcs.proj_plane_pixel_area() * (8.1*u.kpc)**2).to(u.cm**2, u.dimensionless_angles())
total_mass = np.nansum(column.quantity * 2*u.Da * pixel_area).to(u.M_sun)
print(f'Total mass: {total_mass}')

pl.figure()
ax = pl.subplot(projection=subalpha[0].wcs)
pl.imshow(np.log10(column.value), vmin=23, vmax=24)
cb = pl.colorbar()
cb.set_label("N(H$_2$) [log cm$^{-2}$]")
ax.coords[0].set_axislabel('Right Ascension (J2000)')
ax.coords[1].set_axislabel('Declination (J2000)')
ax.coords[0].set_major_formatter('hh:mm:ss.ss')
ax.coords[1].set_major_formatter('dd:mm:ss.ss')
ax.coords[0].set_separator((r'$^{\mathrm{h}}$', r'$^{\mathrm{m}}$', r'$^{\mathrm{s}}$'))
ax.coords[1].set_separator((r'$^{\circ}$', r'$^{\prime}$', r'$^{\prime\prime}$'))
ax.coords[0].set_ticks(spacing=0.5*u.arcsec)
ax.coords[1].set_ticks(spacing=0.5*u.arcsec)
ax.coords[0].set_ticklabel(rotation=25, pad=30)
pl.tight_layout()
ax.set_title(f'Total mass (assuming GDR=100): {total_mass:0.1f}')

pl.savefig("b7_column_density_T=50K.png", bbox_inches='tight')