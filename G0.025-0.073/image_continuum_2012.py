mses = ["uid___A002_X651f57_Xadd.ms.split.cal",
"uid___A002_X680f8a_Xadc.ms.split.cal",
"uid___A002_X6a533e_Xd66.ms.split.cal",
"uid___A002_X7fb89e_X10f5.ms.split.cal",
"uid___A002_X7fc9da_X138b.ms.split.cal",
"uid___A002_X7fc9da_X4064.ms.split.cal",
"uid___A002_X7fffd6_X221.ms.split.cal",
"uid___A002_X95e355_X24b0.ms.split.cal",
"uid___A002_X95e355_X2aa6.ms.split.cal",
"uid___A002_X9a3e71_X306e.ms.split.cal",
]
good_mses = [
"uid___A002_X651f57_Xadd.ms.split.cal",
"uid___A002_X680f8a_Xadc.ms.split.cal",
"uid___A002_X6a533e_Xd66.ms.split.cal",
"uid___A002_X7fb89e_X10f5.ms.split.cal",
"uid___A002_X7fc9da_X138b.ms.split.cal",
"uid___A002_X7fc9da_X4064.ms.split.cal",
]

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u
coord = SkyCoord("17:45:57.7530532310 -28:57:10.7694483833", unit=(u.h, u.deg), frame='icrs')

fields_for_mses = []
fields_for_goodmses = []
for msname in mses:
      ms.open(msname)
      msmd.open(msname)
      fields = msmd.fieldsforname('GC50MC')
      fields_to_keep = []
      for field in fields:
            direction = ms.getfielddirmeas(fieldid=field)
            dir_coord = SkyCoord(direction['m0']['value']*u.rad, direction['m1']['value']*u.rad, frame='fk5')
            if dir_coord.separation(coord) < 35*u.arcsec:
                  fields_to_keep.append(field)
      print(f"Keeping fields {fields_to_keep} for {msname}")
      fields_for_mses.append(",".join(map(str, fields_to_keep)))
      if msname in good_mses:
            fields_for_goodmses.append(fields_for_mses[-1])
      msmd.close()
      ms.close()


if False:
      print("Beginning cleaning per-ms")
      for msname, fields in zip(mses, fields_for_mses):
            name = msname[11:msname.find('.')]
            print(name)
            tclean(vis=msname, imagename = f'Tsuboi2012_compactsource_continuum_r-2_{name}',
                  phasecenter = 'J2000 17h45m57.753 -28d57m10.769', 
                  specmode = 'mfs',
                  nterms=2,
                  field=fields,
                  outframe = 'LSRK', veltype = 'radio', niter = 100000,
                  # these will all be too shallow to model the source, most likely
                  threshold = '1.0mJy',
                  mask='source_mask.crtf',
                  deconvolver='hogbom',
                  gridder = 'mosaic', interactive = False, imsize = [1280, 1280], cell = ['0.15arcsec'],
                  weighting = 'briggs', robust = -2)
            impbcor(f'Tsuboi2012_compactsource_continuum_r-2_{name}.image', f'Tsuboi2012_compactsource_continuum_r-2_{name}.pb', f'Tsuboi2012_compactsource_continuum_r-2_{name}.image.pbcor',)
            for suffix in ('image', 'image.pbcor'):
                  exportfits(f'Tsuboi2012_compactsource_continuum_r-2_{name}.'+suffix,
                        f'Tsuboi2012_compactsource_continuum_r-2_{name}.'+suffix+".fits")

if True:
      print("Beginning cleaning")
      tclean(vis=good_mses, imagename = 'Tsuboi2012_compactsource_continuum_r-2_downselectedMSes',
            phasecenter = 'J2000 17h45m57.753 -28d57m10.769', 
            specmode = 'mfs',
            nterms=2,
            field=fields_for_goodmses,
            outframe = 'LSRK', veltype = 'radio', niter = 100000,
            threshold = '0.6mJy',
            mask='source_mask.crtf',
            deconvolver='hogbom',
            gridder = 'mosaic', interactive = False, imsize = [1280, 1280], cell = ['0.15arcsec'],
            weighting = 'briggs', robust = -2)
      impbcor('Tsuboi2012_compactsource_continuum_r-2_downselectedMSes.image',
              'Tsuboi2012_compactsource_continuum_r-2_downselectedMSes.pb',
              'Tsuboi2012_compactsource_continuum_r-2_downselectedMSes.image.pbcor',)
      for suffix in ('image', 'image.pbcor'):
            exportfits('Tsuboi2012_compactsource_continuum_r-2_downselectedMSes.'+suffix,
                  'Tsuboi2012_compactsource_continuum_r-2_downselectedMSes.'+suffix+".fits")
      print("Done cleaning")

# tclean(vis=mses, imagename = 'GC50MC_continuum_r-2',      #mask = '../calibrated/calibrated.spw0.CS2-1.taper.mask',
#       field = 'GC50MC', phasecenter = 'J2000 17h45m51.4457 -28d59m20.665', 
#       specmode = 'mfs',
#       nterms=2,
#       outframe = 'LSRK', veltype = 'radio', niter = 100000,
#       threshold = '3.5mJy',
#       deconvolver='hogbom',
#       gridder = 'mosaic', interactive = False, imsize = [2048, 2048], cell = ['0.25arcsec'],
#       weighting = 'briggs', robust = -2)