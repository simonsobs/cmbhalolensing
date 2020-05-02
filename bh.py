from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi,lensing
from pixell import enmap,lensing as plensing
import numpy as np
import os,sys
from symlens import qe
from szar import counts

px = 0.5
width = 120./60.
shape,wcs = maps.rect_geometry(width_deg=width,px_res_arcmin=px,proj='plain')
theory = cosmology.default_theory()
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
cltt2d = theory.uCl('TT',modlmap)
lcltt2d = theory.lCl('TT',modlmap)

# Define our NFW cluster
mass = 2e14
c = 3.2
z = 0.7
cc = counts.ClusterCosmology(skipCls=True,skipPower=True)
massOverh = mass / cc.h

# Generate its kappa
kappa = lensing.nfw_kappa(massOverh,modrmap,cc,zL=z,concentration=c,overdensity=500.,critical=True,atClusterZ=True)
#io.plot_img(kappa)

# Convert kappa to deflection
alpha = lensing.alpha_from_kappa(kappa)

# Change number of sims here
nsims = 600

comm,rank,my_tasks = mpi.distribute(nsims)

# Final beam to apply to kappa map
ffwhm = None #2.0

# Simulation settings
afwhm = 1. # ACT beam
anlevel = 1.0 # ACT noise
aellmin = 200
aellmax = 6000
alcut = 20
kellmin = 40
kellmax = 6000

# Apply beam and add noise
def get_sim(cmb,task,expid):
    if expid=='act':
        nid = 2
        fwhm = afwhm
        nlevel = anlevel
    seed = (nid,task)
    npower = (nlevel * np.pi/180./60.)**2.
    nmap = enmap.rand_map((1,)+shape, wcs, np.ones(shape)[None,None]*npower,seed=seed)
    return maps.filter_map(cmb,maps.gauss_beam(modlmap,fwhm)) + nmap


# Make masks and feed_dict
xmask = maps.mask_kspace(shape,wcs,lmin=aellmin,lmax=2000)
ymask = maps.mask_kspace(shape,wcs,lmin=aellmin,lmax=aellmax,lxcut=alcut,lycut=alcut)
kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
fkmask = maps.gauss_beam(modlmap,ffwhm) if ffwhm is not None else 1
fwhm = pfwhm
abeam = maps.gauss_beam(modlmap,fwhm)
fwhm = afwhm
pbeam = maps.gauss_beam(modlmap,fwhm)
feed_dict = {}
feed_dict['uC_T_T'] = lcltt2d
nlevel = anlevel
npower = (nlevel * np.pi/180./60.)**2.
feed_dict['tC_T_T'] = lcltt2d + npower/abeam**2.

# Displace the unlensed map with deflection angle
def lens_map(imap):
    lens_order = 5
    return plensing.displace_map(imap, alpha, order=lens_order)

s = stats.Stats(comm)

for task in my_tasks:
    print(rank,task)
    # Make a CMB sim
    cmb = enmap.rand_map((1,)+shape, wcs, cltt2d[None,None],seed=(0,task))
    cmb = lens_map(cmb[0])[None] # do the lensing

    # Make ACT observed aps
    aobs = get_sim(cmb,task,'act')[0]

    """
    EXERCISE: add a Gaussian SZ blob to the Planck map and the ACT
    map (with appropriate beams). Do you get a bias?
    What if you only add it to the ACT map?

    EXERCISE (bit harder): add the SZ blob to the ACT map
    and then use the tools in orphics.stats.fit_linear_model
    to fit a template and subtract it.
    """

    # Deconvolve beam
    akmap = enmap.fft(aobs,normalize='phys') / abeam

    assert np.all(np.isfinite(akmap))

    feed_dict['X'] = akmap
    feed_dict['Y'] = akmap
    
    
    rkmap = qe.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY="TT", xmask=xmask, ymask=ymask, xname='X_l1', yname='Y_l2', kmask=kmask, physical_units=True)*fkmask

    assert np.all(np.isfinite(rkmap))
    
    kmap = enmap.ifft(rkmap,normalize='phys').real
    s.add_to_stack('kmap',kmap)

s.get_stacks()

if rank==0:
    kmap = s.stacks['kmap']
    io.plot_img(kmap,'kmap.png')


