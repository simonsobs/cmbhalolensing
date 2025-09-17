from pixell import enmap,utils as u,lensing as plensing,bench,bunch,wcsutils,curvedsky as cs
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi,foregrounds as ofg
import numpy as np
import symlens
import sys, shutil, os, warnings
import healpy as hp

broot = "/data5/act/dr6_public/beams/main_beams/nominal/"
dbeams = { 'f090': f"{broot}coadd_pa5_f090_night_beam_tform_jitter_cmb.txt",
          'f150': f"{broot}coadd_pa5_f150_night_beam_tform_jitter_cmb.txt",
          'f220': f"{broot}coadd_pa4_f220_night_beam_tform_jitter_cmb.txt"}

# sky fraction
mask = enmap.read_map('mask.fits')
fsky = maps.wfactor(2,mask)


theory = cosmology.default_theory()

freqs = ['f090','f150']
flmax = 5000
flmin = 200

alms = []
for freq in freqs:

    alm = hp.read_alm(f"inpainted_subtracted_alm_{freq}.fits")
    alms.append(alm.copy())
    

cl_dict = {}
for i in range(len(freqs)):
    for j in range(i, len(freqs)):   # j >= i to cover autos + unique crosses
        cl = cs.alm2cl(alms[i],alms[j]) / fsky
        ells = np.arange(cl.size)
        fells = ells.copy()
        sel = np.logical_and(ells>flmin,ells<flmax)
        ells = ells[sel]
        cl_dict[i,j] = cl[sel].copy()

beams = []
for freq in freqs:
    l,bl = np.loadtxt(dbeams[freq],unpack=True,usecols=[0,1])
    sbl = maps.sanitize_beam(fells,maps.interp(l,bl)(fells))
    if not(np.all(np.isfinite(sbl))):
        print(l[~np.isfinite(sbl)])
        print(sbl[~np.isfinite(sbl)])
        raise ValueError
    print(freq,sbl)
    beams.append(maps.interp(fells,sbl))
        
freqs = np.array([90, 150])
noise_rms = np.array([12., 12.])
lknees = np.array([1000,3000])
alphas = np.array([-3.,-3.])



best, chi2, dof, model_dict = ofg.quick_fit(ells, cl_dict,
                                            freqs, noise_rms, beams, lknees, alphas, fsky,eval_ells=fells,verbose=True,plot=True)

out_fwhm = 1.48

alm_ilc = maps.harmonic_coaddition(
    alms,
    [b(fells) for b in beams],
    model_dict['total'],
    maps.gauss_beam(out_fwhm,fells),
    resp_factors=None,
    return_weights=False
)

hp.write_alm('alm_ilc.fits',alm_ilc,overwrite=True)

cl_ilc = cs.alm2cl(alm_ilc,alm_ilc)
theory = cosmology.default_theory()
cltt = theory.lCl('TT',fells)
pl = io.Plotter('Cl')
pl.add(fells,cltt,color='k')
pl.add(fells,cl_ilc/fsky)
pl.done('cl_ilc.png')

omap = cs.alm2map(alm_ilc,enmap.empty(mask.shape,mask.wcs,dtype=np.float32))
enmap.write_map('omap_ilc.fits',omap)
io.hplot(omap,f'omap_ilc',downgrade=2)

