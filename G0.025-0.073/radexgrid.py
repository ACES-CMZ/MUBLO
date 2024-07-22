import pylab as pl
import numpy as np
import pyradex
import importlib as imp
from astropy import units as u
from tqdm.auto import tqdm
from astropy.table import Table
import warnings

R = pyradex.Radex(temperature=14*u.K, collider_densities={'h2': 1e3*u.cm**-3}, abundance=1e-8, species='SO.Fine', deltav=70)

# 13 = 86.09395	19.314	2_2	1_1
# 2 =  99.29987 9.226   2_3 1_2
# the rest are B7 transitions
linesel = np.array([2, 13, 5, 15, 21, 65, 70, 74])

so32tbl = Table(names=['Density', 'Temperature', 'Column', 'Tex', 'tau', 'T_B', 'brightness'])
so21tbl = Table(names=['Density', 'Temperature', 'Column', 'Tex', 'tau', 'T_B', 'brightness'])

sob7tbls = {}
so_b7_trans = ('21_10', '32_12', '33_23', '89_78', '87_76', '88_77')
for trans in so_b7_trans:
    sob7tbls[trans] = Table(names=['Density', 'Temperature', 'Column', 'Tex', 'tau', 'T_B', 'brightness'])

nden, ncol, ntem = 13, 7, 12
mindens, maxdens = 1e2, 1e8
mincol, maxcol = 1e14, 1e18
mintem, maxtem = 5, 50
densities = np.geomspace(mindens, maxdens, nden)
columns = np.geomspace(mincol, maxcol, ncol)
temperatures = np.linspace(mintem, maxtem, ntem)

for density in densities:
    print(f"Density={density}")
    for column in tqdm(columns):
        for temperature in temperatures:
            tbl = R(density={'h2': density*u.cm**-3}, column=column, temperature=temperature)[linesel]
            row32 = tbl[0]
            so32tbl.add_row([density, temperature, column,
                             row32['Tex'],
                             row32['tau'],
                             row32['T_B'],
                             row32['brightness']])

            row21 = tbl[1]
            so21tbl.add_row([density, temperature, column,
                             row21['Tex'],
                             row21['tau'],
                             row21['T_B'],
                             row21['brightness']])

            # investigate B7 possibilities
            for ii, trans in enumerate(so_b7_trans):
                row = tbl[ii + 2]
                sob7tbls[trans].add_row([density, temperature, column,
                                         row['Tex'], row['tau'], row['T_B'],
                                         row['brightness']])


so32tbl.write('RADEX_Model_SO32.ecsv', overwrite=True)
so21tbl.write('RADEX_Model_SO21.ecsv', overwrite=True)

for trans in so_b7_trans:
    sob7tbls[trans].write(f'RADEX_Model_SO{trans}.ecsv', overwrite=True)

so32tbcube = np.empty([nden, ncol, ntem])
for ind in range(so32tbcube.size):
    ii, jj, kk = np.unravel_index(ind, so32tbcube.shape)
    so32tbcube[ii, jj, kk] = so32tbl[ind]['T_B']

so21tbcube = np.empty([nden, ncol, ntem])
for ind in range(so21tbcube.size):
    ii, jj, kk = np.unravel_index(ind, so21tbcube.shape)
    so21tbcube[ii, jj, kk] = so21tbl[ind]['T_B']

ratiocube = so32tbcube/so21tbcube

# use different sizes for each to make sure we get the index ordering right
nden_fine, ncol_fine, ntem_fine = 100, 101, 102
finegrid_dens = np.geomspace(mindens, maxdens, nden_fine)
finegrid_col = np.geomspace(mincol, maxcol, ncol_fine)
finegrid_tem = np.linspace(mintem, maxtem, ntem_fine)

from scipy import ndimage

in_crds = np.meshgrid(densities, columns, temperatures, indexing='ij')
out_crds = np.meshgrid(finegrid_dens, finegrid_col, finegrid_tem, indexing='ij')

mg = np.meshgrid(np.linspace(0, nden-1, nden_fine), np.linspace(0, ncol-1, ncol_fine), np.linspace(0, ntem-1, ntem_fine), indexing='ij')
old_mg = np.mgrid[0:nden:nden/100, 0:ncol:ncol/101, 0:ntem:ntem/102]
ratiocube_fine = ndimage.map_coordinates(ratiocube, mg, order=1)
so32tbcube_fine = ndimage.map_coordinates(so32tbcube, mg, order=1)
so21tbcube_fine = ndimage.map_coordinates(so21tbcube, mg, order=1)


for ii, tem in enumerate((10, 15, 25, 50)):
    temind = np.argmin(np.abs(finegrid_tem - tem))
    print(tem, finegrid_tem[temind])
    pl.figure(ii+1).clf()
    pl.imshow(ratiocube_fine[:,:,temind], extent=np.log10([mincol, maxcol, mindens, maxdens, ]), cmap='gray', origin='lower')
    pl.colorbar()
    #pl.contour(np.log10(in_crds[1][:,:,0]), np.log10(in_crds[0][:,:,0]), ratiocube[:,:,0], levels=[4.26,4.87])
    pl.contour(np.log10(out_crds[1][:,:,temind]), np.log10(out_crds[0][:,:,temind]), ratiocube_fine[:,:,temind], levels=[4.55,5.21], colors=['r']*2)
    pl.contour(np.log10(out_crds[1][:,:,temind]), np.log10(out_crds[0][:,:,temind]), so32tbcube_fine[:,:,temind], levels=[1.83,1.90], colors=['b']*2)
    pl.contour(np.log10(out_crds[1][:,:,temind]), np.log10(out_crds[0][:,:,temind]), so21tbcube_fine[:,:,temind], levels=[0.355, 0.405], colors=['c']*2)
    #pl.contour(ratiocube_fine[:,:,0], levels=[4.26,4.87])
    pl.gca().set_aspect( 1/((np.log10(maxdens)-np.log10(mindens)) / (np.log10(maxcol)-np.log10(mincol)) ))
    pl.ylabel("H$_2$ Density [cm$^{-3}$]")
    pl.xlabel("SO Column [cm$^{-2}$]")
    pl.title(f"{tem:0.1f} K")
    pl.savefig(f"RADEX_SOratio_model_T={tem:0.1f}.png", bbox_inches='tight')


# check against LTE solution
pl.figure(5).clf()
from pyspeckit.spectrum.models import lte_molecule
nurest_so32 = 99.29987e9*u.Hz
nurest_so21 = 86.09395e9*u.Hz
freqs_SO, aij_SO, deg_SO, EU_SO, partfunc_SO = lte_molecule.get_molecular_parameters(molecule_name=None, molecule_tag=48501,
                                                                      catalog='CDMS', parse_name_locally=False)
temperatures_lin = np.linspace(5, 50)*u.K
for col_this in [5e15, 6e15, 8e15, 1e16, 2e16, 4e16]:
    so32ofT = [lte_molecule.generate_model(nurest_so32, 0*u.km/u.s, 71*u.km/u.s, tex=T,
                                column=col_this, freqs=freqs_SO, aij=aij_SO, deg=deg_SO, EU=EU_SO, partfunc=partfunc_SO)
               for T in temperatures_lin]
    so21ofT = [lte_molecule.generate_model(nurest_so21, 0*u.km/u.s, 71*u.km/u.s, tex=T,
                                column=col_this, freqs=freqs_SO, aij=aij_SO, deg=deg_SO, EU=EU_SO, partfunc=partfunc_SO)
               for T in temperatures_lin]
    so32ofT = np.array(so32ofT)
    #pl.plot(temperatures_lin, so21ofT)
    #pl.plot(temperatures_lin, so32ofT)
    #L, = pl.plot(temperatures_lin, np.array(so32ofT)/np.array(so21ofT), linestyle='--')
    pl.plot(temperatures_lin[so32ofT > 1.75], (np.array(so32ofT)/np.array(so21ofT))[so32ofT > 1.75], linestyle='-',
            label=f'{col_this:0.0e} cm$^{{-2}}$')#, color=L.get_color())

# try to reproduce LTE...
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rslts = [R(density=1e8, column=1.5e15, temperature=T) for T in temperatures_lin]
rats = [rs[2]['T_B']/rs[13]['T_B'] for rs in rslts]
pl.plot(temperatures_lin, rats, linestyle='--')
colind = np.argmin(np.abs(finegrid_col-6e15))
pl.plot(finegrid_tem, ratiocube_fine[-1, colind, :], linestyle='--')
#pl.plot(finegrid_tem, ratiocube_fine[-1, -1, :], linestyle='--')
#pl.plot(finegrid_tem, ratiocube_fine[-1, 0, :], linestyle='--')

pl.axhline(4.88, linestyle='--', color='k')
pl.legend()
pl.xlabel('Temperature [K]')
pl.xlim(0, 52);
pl.ylabel("Ratio SO 3(2)-2(1) / 2(2)-1(1)");

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

sel = np.array([check_in_band(frq) for frq in u.Quantity(rslts[10]['frequency'].quantity, u.GHz)])

so_rslt = R(density=10**5*u.cm**-3, temperature=temperature*u.K, abundance=1e-8, species='SO.fine')
so_nl = np.array([int(x.split("_")[0]) for x in so_rslt['lowerlevel']] )
so_jl = np.array([int(x.split("_")[1]) for x in so_rslt['lowerlevel']] )
so_nu = np.array([int(x.split("_")[0]) for x in so_rslt['upperlevel']] )
so_ju = np.array([int(x.split("_")[1]) for x in so_rslt['upperlevel']] )
f1l = so_nl == so_jl - 1
f2l = so_nl == so_jl
f3l = so_nl == so_jl + 1
f1u = so_nu == so_ju - 1
f2u = so_nu == so_ju
f3u = so_nu == so_ju + 1

pl.figure(6).clf()
for density, temperature, X in [(3, 14, 1e-6), (7, 14, 5e-8), (5, 50, 5e-8)]:
    so_rslt = R(density=10**density*u.cm**-3, temperature=temperature*u.K, abundance=X, species='SO.fine')
    keep = so_rslt['T_B'] > 0.1

    so_kp = so_rslt[keep&f1l]
    so_kp.sort('frequency')

    pl.plot(so_kp['frequency'], so_kp['T_B'], '-', label=f'T={temperature} K $n=10^{{{density}}}$ X={X:0.1e}')
    pl.plot(so_rslt['frequency'][keep&sel&f1l], so_rslt['T_B'][keep&sel&f1l], 's', color='black',)
    pl.plot(so_rslt['frequency'][2], so_rslt['T_B'][2], 'r+',)
    pl.plot(so_rslt['frequency'][13], so_rslt['T_B'][13], 'rx',)

pl.xlim(0, 375)
pl.xlabel("Frequency [GHz]")
pl.ylabel("T$_B$ [K]")

pl.legend(loc='best')


cs_mod = pyradex.Radex(temperature=14*u.K, collider_densities={'h2': 1e8*u.cm**-3}, abundance=1e-9, species='cs@lique', deltav=70)

cs_rslt = cs_mod()

sel_cs = np.array([check_in_band(frq) for frq in u.Quantity(cs_rslt['frequency'].quantity, u.GHz)])

pl.figure(7).clf()
#for density, temperature, X in [(5, 14, 1.5e-9), (6, 14, 1e-10), (4.5, 50, 5e-9)]:
for density, temperature, X in [(5, 14, 1e-7), (6, 14, 5e-9), (5, 50, 2e-8), (7, 14, 1e-9), ]:
    cs_rslt = cs_mod(density=10**density*u.cm**-3, temperature=temperature*u.K, abundance=X)
    pl.plot(cs_rslt['frequency'][:20], cs_rslt['T_B'][:20], 'o-', label=f'T={temperature} K $n=10^{{{density}}}$ X={X:0.1e}')
    pl.plot(cs_rslt['frequency'][:20][sel_cs[:20]], cs_rslt['T_B'][:20][sel_cs[:20]], 's', color='black',)
    pl.plot(cs_rslt['frequency'][1], cs_rslt['T_B'][1], 'rs',)

pl.xlim(0, 375)
pl.xlabel("Frequency [GHz]")
pl.ylabel("T$_B$ [K]")
pl.legend(loc='best')
pl.savefig("CS_RADEX_models_withobs.png")
pl.savefig("CS_RADEX_models_withobs.pdf")

# DEBUG
if False:
    pl.figure(3).clf()
    pl.imshow(ratiocube_fine[:,:,0], cmap='gray', origin='lower')
    #pl.contour(ratiocube[:,:,0], levels=[4.26,4.87])
    pl.contour(ratiocube_fine[:,:,0], levels=[4.26,4.87])

    pl.figure(4).clf()
    pl.imshow(ratiocube[:,:,0], cmap='gray', origin='lower')
    pl.contour(ratiocube[:,:,0], levels=[4.26,4.87])

    pl.figure(2).clf()
    pl.imshow(ratiocube[:,:,0], extent=np.log10([mincol, maxcol, mindens, maxdens,]), cmap='gray', origin='lower')
    pl.colorbar()
    pl.contour(np.log10(in_crds[1][:,:,0]), np.log10(in_crds[0][:,:,0]), ratiocube[:,:,0], levels=[4.26,4.87], colors=['r']*2)
    pl.contour(np.log10(in_crds[1][:,:,0]), np.log10(in_crds[0][:,:,0]), so32tbcube[:,:,0], levels=[1.7,1.8], colors=['b']*2)
    pl.gca().set_aspect( 1/((np.log10(maxdens)-np.log10(mindens)) / (np.log10(maxcol)-np.log10(mincol)) ))
    pl.ylabel("Density")
    pl.xlabel("Column")

# SO B7 exploration
pl.figure(8).clf()

col = columns[2]
for temperature, linestyle in zip(temperatures[2::9], ('-', '--')):
    for trans in sob7tbls:
        tb = sob7tbls[trans]
        sel = (tb['Temperature'] == temperature) & (tb['Column'] == col)
        pl.plot(tb['Density'][sel], tb['T_B'][sel], label=f'{trans} T={temperature:0.1f}', linestyle=linestyle)
pl.xscale('log')
pl.legend(loc='best')
