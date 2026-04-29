import numpy as np
from astropy import units as u, constants
from astroquery.linelists.cdms import CDMS

# SH+
SHplus = CDMS.query_lines(38*u.GHz, 1000*u.GHz, molecule='033505')
#
# HCS+
HCSplus = CDMS.query_lines(38*u.GHz, 1000*u.GHz, molecule='045506')
molecules = {'SH+': '033505',
             'HCS+': '045506',
             'HOC+': '029504',
             'H2S': '034502',
             'C3H+': '037505',
             'H2CS': '046509',
            }


observing_centers = [345.05, 343.15, 331.2, 333.1,
                     259, 261.5, 245.7, 243.8,
                     86, 88, 98.1, 99.9,
                     135.8, 137.79, 149.6, 147.8,
                     217.2, 219.15, 230.5, 232.3,
                    ]*u.GHz
# 35.8 - 50
def check_in_band(freq, bw=1.875*u.GHz):
    for cc in observing_centers:
        if (freq > cc - bw / 2) and (freq < cc + bw / 2):
            return True
        #if (35.8*u.GHz < freq) and (50*u.GHz > freq):
        #    return True
    return False


gijs_list = ['CO', 'O2', 'NO', 'N2', 'HOC+', 'NO+', 'H3O+', 'OH', 'HCO+', 'H2O', 'HCNH+', 'HCOO', 'CH3CNH', 'CH',  'SO+', 'CH3', 'CH3CCH', 'CN', 'HCN', 'HNO', 'H2NO+', 'NH2', 'N+', 'SO', 'SiO', 'CH3OH', 'C2NH+', 'H3CS+', 'H2CO', 'C2N+', 'C2', 'HCOOH', 'HCS+', 'H2CO+', 'C2H', 'NH3+', 'NH', 'HS', 'HCO', 'H2SiO', 'C2H2+', 'CS', 'HNC', 'NO2', 'C2H2', 'H2CS', 'H3CO', 'HS2+', 'OCS', 'NS+', 'C2N', 'CH5+', 'OCS+', 'HNO+', 'HCL', 'S2']
#  'C3H5+','CH4', 'HC(O)NH2', NH4+,  'NH3', (nh3 has bad QNs for CDMS right now), 'O2+', 'O2H',  'SiOH+', 'HSO+', 'CH3OH2+', 'H3CO+', 'HCO2+', 'CH3+', 

for molname in gijs_list:
    if '+' in molname:
        molname = molname.replace("+", "\\+")
    qq = CDMS.query_lines(30*u.GHz, 1000*u.GHz, molecule=molname, parse_name_locally=True)
    sel = np.array([check_in_band(frq) for frq in qq['FREQ'].quantity])
    print(molname)
    print(qq[sel])

for molname, molid in molecules.items():
    qq = CDMS.query_lines(30*u.GHz, 1000*u.GHz, molecule=molid)
    sel = np.array([check_in_band(frq) for frq in qq['FREQ'].quantity])
    print(molname)
    print(qq[sel])




"""
SH+
   FREQ    ERR   LGINT   DR  ELO   GUP MOLWT TAG QNFMT  Ju  Ku  vu F1u F2u F3u  Jl  Kl  vl F1l F2l F3l name Lab
   MHz     MHz  nm2 MHz     1 / cm       u
---------- ---- ------- --- ------ --- ----- --- ----- --- --- --- --- --- --- --- --- --- --- --- --- ---- ----
345858.271 0.05 -2.5753   2 0.0029   2    33 505   113   1   0   1  --  --  --   0   1   1  --  --  --  SH+ True
345944.379 0.05 -2.2739   2    0.0   2    33 505   113   1   0   1  --  --  --   0   1   2  --  --  --  SH+ True
HCS+
   FREQ     ERR    LGINT   DR  ELO   GUP MOLWT TAG QNFMT  Ju  Ku  vu F1u F2u F3u  Jl  Kl  vl F1l F2l F3l name  Lab
   MHz      MHz   nm2 MHz     1 / cm       u
---------- ------ ------- --- ------ --- ----- --- ----- --- --- --- --- --- --- --- --- --- --- --- --- ---- -----
42674.1954 0.0013 -3.8017   2    0.0   3    45 506   101   1  --  --  --  --  --   0  --  --  --  --  -- HCS+ False
  85347.89   0.03 -2.9031   2 1.4235   5    45 506   101   2  --  --  --  --  --   1  --  --  --  --  -- HCS+  True
HOC+
FREQ ERR  LGINT   DR  ELO   GUP MOLWT TAG QNFMT  Ju  Ku  vu F1u F2u F3u  Jl  Kl  vl F1l F2l F3l name Lab
MHz  MHz nm2 MHz     1 / cm       u
---- --- ------- --- ------ --- ----- --- ----- --- --- --- --- --- --- --- --- --- --- --- --- ---- ---
H2S
    FREQ     ERR    LGINT   DR   ELO    GUP MOLWT TAG QNFMT  Ju  Ku  vu F1u F2u F3u  Jl  Kl  vl F1l F2l F3l name  Lab
    MHz      MHz   nm2 MHz      1 / cm        u
----------- ------ ------- --- -------- --- ----- --- ----- --- --- --- --- --- --- --- --- --- --- --- --- ---- -----
 39701.8293 0.0415 -9.1661   3 415.5219  13    34 502   303   6   6   0  --  --  --   7   3   5  --  --  --  H2S False
 149662.617   0.05   -6.91   3 737.8607  63    34 502   303  10   3   8  --  --  --   9   4   5  --  --  --  H2S  True
216710.4365 0.0015 -3.0171   3  51.1402   5    34 502   303   2   2   0  --  --  --   2   1   1  --  --  --  H2S  True
C3H+
   FREQ     ERR    LGINT   DR   ELO   GUP MOLWT TAG QNFMT  Ju  Ku  vu F1u F2u F3u  Jl  Kl  vl F1l F2l F3l name Lab
   MHz      MHz   nm2 MHz      1 / cm       u
---------- ------ ------- --- ------- --- ----- --- ----- --- --- --- --- --- --- --- --- --- --- --- --- ---- ----
44979.5486 0.0102  -3.364   2  0.7502   5    37 505   101   2  --  --  --  --  --   1  --  --  --  --  -- C3H+ True
134932.733  0.017 -1.9577   2 11.2525  13    37 505   101   6  --  --  --  --  --   5  --  --  --  --  -- C3H+ True
"""
