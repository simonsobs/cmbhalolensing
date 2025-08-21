import os,sys
import numpy as np
import healpy as hp
from orphics import maps,io,interfaces as ints,lensing,stats
from pixell import utils,bench,enmap,reproject

rstamp = 30.1 * utils.arcmin
res = 0.1 * utils.arcmin
Nmax = None
#Nmax = 3

with bench.show("load kappa"):
    kappa = hp.read_map("/data5/sims/agora_sims/cmbkappa/raytrace16384_ip20_cmbkappa.zs1.kg1g2_highzadded_lowLcorrected.fits")
    npix = kappa.size
    nside = hp.npix2nside(npix)
    print("nside: ", nside)

# with bench.show("load car kappa"):
#     k = enmap.read_map("kappa_car_highres.fits")

z_min = 0.5
z_max = 0.51
mmin = 2e14
mmax = 2.2e14

oras,odecs,ozs,oms = ints.get_agora_halos(z_min = z_min, z_max = z_max,
                    mass_min = mmin, mass_max = mmax,
                    massdef='m500') # Get agora halos

oras = oras[:Nmax]
odecs = odecs[:Nmax]

print(f"Starting stack on {len(oras)}")
out = 0.
cout = 0.
for i,(ora,odec) in enumerate(zip(oras,odecs)):
    coord = np.asarray((odec,ora))*utils.degree
    othumb = maps.thumbnail_healpix(kappa,coord,r=rstamp,res=res) # reproject directly from healpix

    # cthumb = reproject.thumbnails(k, coord, r=rstamp, res=res, proj='tan', apod=2*utils.arcmin,
    #     	                  method="mixed", order=3, oversample=4, pol=None, oshape=None, owcs=None,
    #     	                  extensive=False, verbose=False, filter=None,pixwin=False,pixwin_order=0)

    
    if (i+1)%100==0: print(f"Done {i+1} / {len(oras)}")
    out = out + othumb
    # cout = cout + cthumb

N = len(oras)
out = out / N
# cout = cout / N

lmin = 200
lmax = 24000
rmin = 0.
rmax = 10*utils.arcmin
rwidth = 0.5 * utils.arcmin

cents,kap_agora = lensing.filter_bin_kappa2d(out,lmin=lmin,lmax=lmax,rmin=0.,rmax=rmax,rwidth=rwidth)
# cents,ckap_agora = lensing.filter_bin_kappa2d(cout,lmin=lmin,lmax=lmax,rmin=0.,rmax=rmax,rwidth=rwidth)


z = ozs.mean()
ez = ozs.std()
m500c = oms.mean()
em500c = oms.std()

thetas,kappa_1h,kappa_2h,tot_kappa,cents_t,b1d1h,b1d2h,b1d_t = lensing.kappa_nfw_profiley(mass=m500c,conc=None,
                                                                                          z=z,z_s=1100.,background='critical',delta=500, R_off_Mpc = 0.05,
                                                                                          apply_filter=True,
                                                                                          lmin=lmin,lmax=lmax,res=res,fls=hp.pixwin(nside), # note the healpix pixwin here
                                                                                          rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
title =  f"$M_{{500}}$={m500c/1e14:.2f} $\\pm$ {em500c/1e14:.2f} $10^{{14}} M_{{\\odot}}$  ;  z={z:.2f} $\\pm$ {ez:.2f} ; N = {N}"
pl = io.Plotter(xyscale='linlin',xlabel='$\\theta$ (arcmin)',ylabel='$\\kappa$',title=title)
pl.add(cents/utils.arcmin,kap_agora,ls='-',color='r',label=f'Agora stack')
# pl.add(cents/utils.arcmin,ckap_agora,ls='-',color='b',label=f'Agora stack (CAR)')
pl.add(cents_t/utils.arcmin,b1d_t,ls='--',color='k',label='profiley theory')
pl.hline(y=0)
pl.done('profile.png')

pl = io.Plotter('LCL',xlabel='$\\theta$ (arcmin)',ylabel='$\\theta\\kappa$',title=title)
pl.add(cents/utils.arcmin,kap_agora,ls='-',color='r',label=f'Agora stack')
# pl.add(cents/utils.arcmin,ckap_agora,ls='-',color='b',label=f'Agora stack (CAR)')
pl.add(cents_t/utils.arcmin,b1d_t,ls='--',color='k',label='profiley theory')
pl.hline(y=0)
pl.done('profilet.png')

pl = io.Plotter(xyscale='linlin',xlabel='$\\theta$ (arcmin)',ylabel='$\\kappa / \\kappa$',title=title)
pl.add(cents/utils.arcmin,kap_agora/b1d_t,ls='-',color='r',label=f'Agora stack / theory')
# pl.add(cents/utils.arcmin,ckap_agora/b1d_t,ls='-',color='b',label=f'Agora stack / theory')
pl.hline(y=1.)
pl._ax.set_ylim(0.3,1.7)
pl.done('profile_rat.png')


enmap.write_map("stack.fits",out)