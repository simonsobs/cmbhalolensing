from pixell import enmap,utils as u,lensing as plensing,bench,bunch,wcsutils,curvedsky as cs
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi,foregrounds as ofg
import numpy as np
import symlens
import sys, shutil, os, warnings
import healpy as hp

alm_ilc = hp.read_alm('alm_ilc.fits')

in_fwhm = 1.57
out_fwhm = 1.48



cl_ilc = cs.alm2cl(alm_ilc,alm_ilc)

fells = np.arange(cl_ilc.size)

trans = maps.gauss_beam(fells,out_fwhm)/maps.gauss_beam(fells,in_fwhm)
oalm_ilc = cs.almxfl(alm_ilc,trans)
cl_ilc = cs.alm2cl(alm_ilc,alm_ilc)


theory = cosmology.default_theory()
cltt = theory.lCl('TT',fells)
pl = io.Plotter('Cl')
pl.add(fells,cltt,color='k')
pl.add(fells,cl_ilc)
pl.done('cl_ilc.png')

omap = cs.alm2map(alm_ilc,enmap.empty(mask.shape,mask.wcs,dtype=np.float32))
enmap.write_map('omap_ilc.fits',omap)
io.hplot(omap,f'omap_ilc',downgrade=2)
