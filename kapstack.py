import os,sys
import numpy as np
import healpy as hp
from orphics import maps,io,interfaces as ints,lensing,stats
from pixell import utils,bench,enmap

rstamp = 30.0 * utils.arcmin
res = 0.25 * utils.arcmin

with bench.show("load kappa"):
    kappa = hp.read_map("/data5/sims/agora_sims/cmbkappa/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")

z_min = 0.5
z_max = 0.54
mmin = 3e14
mmax = 3.4e14

oras,odecs,ozs,oms = ints.get_agora_halos(z_min = z_min, z_max = z_max,
                    mass_min = mmin, mass_max = mmax,
                    massdef='m500') # Get agora halos

print(f"Starting stack on {len(oras)}")
out = 0.
for i,(ora,odec) in enumerate(zip(oras,odecs)):
    othumb = maps.thumbnail_healpix(kappa,np.asarray((odec,ora))*utils.degree,r=rstamp,res=res) # reproject directly from healpix
    if i%100==0: print(f"Done {i+1} / {len(oras)}")
    out = out + othumb

out = out / len(oras)

lmin = 200
lmax = 3500
rmin = 0.
rmax = 10*utils.arcmin
rwidth = 1.0 * utils.arcmin

cents,kap_agora = lensing.filter_bin_kappa2d(out,lmin=lmin,lmax=lmax,rmin=0.,rmax=rmax,rwidth=rwidth)


z = ozs.mean()
ez = ozs.std()
m500c = oms.mean()
em500c = oms.std()

thetas,kappa_1h,kappa_2h,tot_kappa,cents_t,b1d1h,b1d2h,b1d_t = lensing.kappa_nfw_profiley(mass=m500c,conc=None,
                                                                                          z=z,z_s=1100.,background='critical',delta=500,apply_filter=True,
                                                                                          lmin=lmin,lmax=lmax,res=res,
                                                                                          rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
pl = io.Plotter(xyscale='linlin',xlabel='$\\theta$ (arcmin)',ylabel='$\\kappa$')
pl.add(cents/utils.arcmin,kap_agora,ls='-',color='r',label=f'Agora stack; M={m500c/1e14:.1f} $\\pm$ {em500c/1e14:.1f} ;  z={z:.1f} $\\pm$ {ez:.1f} ; ')
pl.add(cents_t/utils.arcmin,b1d_t,ls='--',color='k',label='profiley theory')
pl.hline(y=0)
pl.done('profile.png')
enmap.write_map("stack.fits",out)
