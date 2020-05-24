from __future__ import print_function
from orphics import maps,io,cosmology,stats,lensing
from pixell import enmap,utils
import numpy as np
import os,sys
from szar import counts

ombh2 = 0.0223
om = 0.24
h = 0.73
ns = 0.958
omb = ombh2 / h**2
omc = om - omb
omch2 = omc * h**2.
As = cosmology.As_from_s8(sigma8 = 0.76,bounds=[1.9e-9,2.5e-9],rtol=1e-4,omegab = omb, omegac = omc, ns = ns, h = h)
print(As)
params = {}
params['As'] = As
params['H0'] = h * 100.
params['omch2'] = omch2
params['ombh2'] = ombh2
params['ns'] = ns

conc = 3.2
cc = counts.ClusterCosmology(params,skipCls=True,skipPower=True,skip_growth=True)
z = 0.7
mass = 2e14

thetas = np.geomspace(0.1,10,1000)
kappa = lensing.nfw_kappa(mass,thetas*utils.arcmin,cc,zL=z,concentration=conc,overdensity=180,critical=False,atClusterZ=False)
hthetas,hkappa = np.loadtxt("hdv_unfiltered.csv",unpack=True,delimiter=',')

pl = io.Plotter(xyscale='loglog', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl.add(thetas,kappa)
pl.add(hthetas,hkappa,ls='--')
pl.done('test_uhdv.png')

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl.add(hthetas,hkappa/maps.interp(thetas,kappa)(hthetas),ls='--')
pl.hline(y=1)
pl.done('test_uhdv_ratio.png')


shape,wcs = enmap.geometry(pos=(0,0),shape=(512,512),res=0.2 * utils.arcmin,proj='plain')
kmask = maps.mask_kspace(shape,wcs,lmax=8095)

bin_edges_arcmin= np.arange(0,15,0.4)
cents,k1d = lensing.binned_nfw(mass,z,conc,cc,shape,wcs,bin_edges_arcmin,overdensity=180.,critical=False,at_cluster_z=False,kmask=kmask)

hcents,hk1d = np.loadtxt("hdv_filtered_kappa.csv",unpack=True,delimiter=',')

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl.add(cents*180.*60/np.pi,k1d)
pl.add(hcents,hk1d,ls='--')
pl.done('test_hdv.png')

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl.add(hcents,hk1d/maps.interp(cents*180.*60/np.pi,k1d)(hcents),ls='--')
pl.hline(y=1)
pl.done('test_hdv_ratio.png')
