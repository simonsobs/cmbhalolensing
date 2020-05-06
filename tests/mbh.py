from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi,lensing
from pixell import enmap,lensing as plensing
import numpy as np
import os,sys
from symlens import qe
from szar import counts
import ptfit

savedir = "/scratch/r/rbond/msyriac/data/depot/cmbh"
debug = False

px = 0.5
ipx = 0.1
width = 100./60.
dshape,dwcs = maps.rect_geometry(width_deg=width,px_res_arcmin=px,proj='plain')
ishape,iwcs = maps.rect_geometry(width_deg=width,px_res_arcmin=ipx,proj='plain')
theory = cosmology.default_theory()
imodrmap = enmap.modrmap(ishape,iwcs)
crop = 20
mask = True

# Define our NFW cluster
mass = 2e14
c = 3.2
z = 0.7
cc = counts.ClusterCosmology(skipCls=True,skipPower=True)
massOverh = mass / cc.h

# Generate its kappa
ikappa = lensing.nfw_kappa(massOverh,imodrmap,cc,zL=z,concentration=c,overdensity=500.,critical=True,atClusterZ=True)

# Convert kappa to deflection
alpha = lensing.alpha_from_kappa(ikappa)

# Change number of sims here
nsims = 2000

comm,rank,my_tasks = mpi.distribute(nsims)

# Final beam to apply to kappa map
ffwhm = None #2.0

# Simulation settings
afwhm = 1.5 # ACT beam
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
    nmap = maps.white_noise((1,)+shape,wcs,nlevel,seed=seed)
    return maps.filter_map(cmb,maps.gauss_beam(modlmap,fwhm)) + nmap



# Displace the unlensed map with deflection angle
def lens_map(imap):
    lens_order = 5
    return enmap.resample(plensing.displace_map(imap, alpha, order=lens_order),dshape)

s = stats.Stats(comm)

imodlmap = enmap.modlmap(ishape,iwcs)
cltt2d = theory.uCl('TT',imodlmap)

def fit_and_subtract(imap):
    stamp = maps.crop_center(imap,crop)
    sel = maps.crop_center(imap,crop,sel=True)
    pflux,cov,fit,chisquare = pfit.fit(imap=stamp,dec=0,ra=0,rbeam=sz_sigma)
    imap[sel] = stamp-fit
    return imap

for i,task in enumerate(my_tasks):
    print(rank,task)
    # Make a CMB sim
    cmb = enmap.rand_map((1,)+ishape, iwcs, cltt2d[None,None],seed=(0,task))
    cmb = lens_map(cmb[0])[None] # do the lensing

    if i==0:
        shape,wcs = cmb.shape[-2:],cmb.wcs
        modlmap = enmap.modlmap(shape,wcs)
        modrmap = enmap.modrmap(shape,wcs)
        lcltt2d = theory.lCl('TT',modlmap)
        kappa = lensing.nfw_kappa(massOverh,modrmap,cc,zL=z,concentration=c,overdensity=500.,critical=True,atClusterZ=True)
        # Make masks and feed_dict
        xmask = maps.mask_kspace(shape,wcs,lmin=aellmin,lmax=2000)
        ymask = maps.mask_kspace(shape,wcs,lmin=aellmin,lmax=aellmax,lxcut=alcut,lycut=alcut)
        kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
        fkmask = maps.gauss_beam(modlmap,ffwhm) if ffwhm is not None else 1
        abeam = maps.gauss_beam(modlmap,afwhm)
        feed_dict = {}
        feed_dict['uC_T_T'] = lcltt2d
        nlevel = anlevel
        npower = (nlevel * np.pi/180./60.)**2.
        feed_dict['tC_T_T'] = lcltt2d + npower/abeam**2.
        feed_dict['pc_T_T'] = 1.
        h = qe.HardenedTT(shape,wcs,feed_dict,xmask=xmask,ymask=ymask,kmask=kmask,Al=None,estimator='hdv',hardening='mask')
        q = qe.QE(shape,wcs,feed_dict,estimator="hdv",XY="TT",xmask=xmask,ymask=ymask,
                  field_names=None,groups=None,kmask=kmask)

        inp = maps.filter_map(kappa,kmask)
        bin_edges = np.arange(0.,10.,1.5)
        binner = stats.bin2D(inp.modrmap()*(180*60/np.pi),bin_edges)

        sz_fwhm_arcmin = 1.5
        sz_sigma = maps.sigma_from_fwhm(sz_fwhm_arcmin*np.pi/180./60.)
        sz_amp = -100
        szmap = sz_amp * np.exp(-(modrmap)**2./2./sz_sigma**2.)

        ells = np.arange(modlmap.max())
        cltt = theory.uCl('TT',ells)

        # stamp = maps.crop_center(szmap,crop)
        # div = maps.ivar(stamp.shape,stamp.wcs,anlevel)
        # pfit = ptfit.Pfit(stamp.shape,stamp.wcs,dec=None,ra=None,rbeam=sz_sigma,div=div,beam=afwhm,iau=True,
        #                  ps=cltt,n2d=None,totp2d=None,invert=False)

        taper,_ = maps.get_taper(shape,wcs)
      
    

    # Make ACT observed aps
    aobs = get_sim(cmb,task,'act')[0] * taper

    if debug and i==0:
        io.plot_img(aobs,f'{savedir}/obs.png',arc_width=width*60.,lim=300)
        io.plot_img(aobs_sz,f'{savedir}/obs_sz.png',arc_width=width*60.,lim=300)
        io.plot_img(szmap,f'{savedir}/obs_szonly.png',arc_width=width*60.,lim=300)


    """
    EXERCISE: add a Gaussian SZ blob to the Planck map and the ACT
    map (with appropriate beams). Do you get a bias?
    What if you only add it to the ACT map?

    EXERCISE (bit harder): add the SZ blob to the ACT map
    and then use the tools in orphics.stats.fit_linear_model
    to fit a template and subtract it.
    """

    # Deconvolve beam
    akmap1 = enmap.fft(aobs,normalize='phys') / abeam

    assert np.all(np.isfinite(akmap1))

    feed_dict['X'] = akmap1
    feed_dict['Y'] = akmap1
    
    
    rkmap = q.reconstruct(feed_dict,xname='X_l1', yname='Y_l2', physical_units=True)
    brkmap = h.reconstruct(feed_dict,xname='X_l1',yname='Y_l2',groups=None,physical_units=True)

    assert np.all(np.isfinite(rkmap))
    assert np.all(np.isfinite(brkmap))
    
    kmap = enmap.ifft(rkmap,normalize='phys').real
    s.add_to_stack('kmap',kmap)
    bkmap = enmap.ifft(brkmap,normalize='phys').real
    s.add_to_stack('bkmap',bkmap)

    cents,r1d = binner.bin(kmap)
    cents,b1d = binner.bin(bkmap)

    s.add_to_stats('r1d',r1d)
    s.add_to_stats('b1d',b1d)


s.get_stacks()
s.get_stats()

if rank==0:
    kmap = s.stacks['kmap']
    io.plot_img(kmap,f'{savedir}/kmap.png',arc_width=width*60.)
    bkmap = s.stacks['bkmap']
    io.plot_img(bkmap,f'{savedir}/bkmap.png',arc_width=width*60.)




    cents,i1d = binner.bin(inp)

    r1d = s.stats['r1d']['mean']
    er1d = s.stats['r1d']['errmean']
    b1d = s.stats['b1d']['mean']
    eb1d = s.stats['b1d']['errmean']


    pl = io.Plotter(xyscale='linlin',xlabel='R',ylabel='k')
    pl.add(cents,i1d,label='inp')
    pl.add_err(cents,r1d,yerr=er1d,label='sqe')
    pl.add_err(cents,b1d,yerr=eb1d,label='bh')
    pl.hline(y=0)
    pl.done(f'{savedir}/prof.png')

    pl = io.Plotter(xyscale='linlin',xlabel='R',ylabel='k')
    pl.add(cents,eb1d/er1d)
    pl.hline(y=1)
    pl.done(f'{savedir}/eratprof.png')
