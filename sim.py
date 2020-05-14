from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi,lensing
from pixell import enmap,lensing as plensing
import numpy as np
import os,sys
from symlens import qe
#import HMFunc as hf
#from szar import counts
from HMFunc.cosmology import Cosmology

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
#cc = counts.ClusterCosmology(skipCls=True,skipPower=True)
cc = Cosmology()
massOverh = mass / cc.h
print(cc.h)


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
pfwhm = 7.0 # Planck beam
anlevel = 1.0 # ACT noise
pnlevel = 40.0 # Planck noise
pellmin = 100 
pellmax = 2000
aellmin = 500
aellmax = 6000
alcut = 20
kellmin = 40
kellmax = 5000

# Apply beam and add noise
def get_sim(cmb,task,expid):
    if expid=='planck':
        nid = 1
        fwhm = pfwhm
        nlevel = pnlevel
    elif expid=='act':
        nid = 2
        fwhm = afwhm
        nlevel = anlevel
    seed = (nid,task)
    npower = (nlevel * np.pi/180./60.)**2.
    nmap = enmap.rand_map((1,)+shape, wcs, np.ones(shape)[None,None]*npower,seed=seed)
    return maps.filter_map(cmb,maps.gauss_beam(modlmap,fwhm)) + nmap


# Make masks and feed_dict
xmask = maps.mask_kspace(shape,wcs,lmin=pellmin,lmax=pellmax)
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
feed_dict['tC_A_T_A_T'] = lcltt2d + npower/abeam**2.
nlevel = pnlevel
npower = (nlevel * np.pi/180./60.)**2.
feed_dict['tC_P_T_P_T'] = lcltt2d + npower/pbeam**2.
feed_dict['tC_A_T_P_T'] = lcltt2d

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


    """
    EXERCISE: do everything up to here at a higher resolution with
    geometry ushape,uwcs, which is defined for the same patch width
    but at resolution say 0.1 arcmin pixel width instead of 0.5 arcmin.

    px = 0.1
    width = 120./60.
    oshape,owcs = maps.rect_geometry(width_deg=width,px_res_arcmin=px,proj='plain')

    Then downgrade to the original resolution defined on oshape,owcs
    (with 0.5 arcmin pixel width) using
    px = 0.5
    width = 120./60.
    oshape,owcs = maps.rect_geometry(width_deg=width,px_res_arcmin=px,proj='plain')
    dcmb = enmap.resample(cmb, oshape)
    owcs = dcmb.wcs # Override the wcs with the more correct that resample provides

    Note that things like modrmap and modlmap are defined for their corresponding
    geometries (oshape,owcs) or (dshape,dwcs), so you'll have to keep track of that
    separately and carefully.

    The idea behind the above is to do just the lensing simulation operation at higher resolution
    for better accuracy, and do the rest of the analysis at the usual pixelization
    we have for ACT.
    """

    # Make Planck and ACT observed aps
    pobs = get_sim(cmb,task,'planck')[0]
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
    pkmap = enmap.fft(pobs,normalize='phys') / pbeam
    akmap = enmap.fft(aobs,normalize='phys') / abeam

    assert np.all(np.isfinite(pkmap))
    assert np.all(np.isfinite(akmap))

    feed_dict['X'] = pkmap
    feed_dict['Y'] = akmap
    
    
    rkmap = qe.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY="TT", xmask=xmask, ymask=ymask, field_names=['P','A'], xname='X_l1', yname='Y_l2', kmask=kmask, physical_units=True)*fkmask

    # pr2d = (rkmap*np.conj(rkmap)).real
    # io.plot_img(np.log(np.fft.fftshift(pr2d)),'pr2d.png')

    assert np.all(np.isfinite(rkmap))
    
    kmap = enmap.ifft(rkmap,normalize='phys').real
    s.add_to_stack('kmap',kmap)

s.get_stacks()

if rank==0:
    kmap = s.stacks['kmap']
    io.plot_img(kmap,'kmap.png')


    """
    EXERCISE: plot the profile of the stacked kappa
    EXERCISE: compare the profile to that of the input kappa
    """
