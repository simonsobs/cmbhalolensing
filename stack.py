import numpy as np
import matplotlib.pyplot as plt
import utils as cutils
from pixell import enmap, reproject, enplot, utils, wcsutils
from orphics import maps, mpi, io, stats,cosmology
from scipy.optimize import curve_fit
from numpy import save
import symlens
import healpy as hp
import os, sys
import time as t
from enlib import bench
import warnings


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Stacked CMB lensing.')
parser.add_argument("version", type=str,help='Version label.')
parser.add_argument("cat_type", type=str,help='Catalog path relative to data directory.')
parser.add_argument("-N", "--nmax",     type=int,  default=None,help="Limit number of objects used e.g. for debugging or quick tests, or for sim injections.")
parser.add_argument("--plmin",     type=int,  default=200,help="Minimum multipole for Planck.")
parser.add_argument("--plmax",     type=int,  default=2000,help="Maximum multipole for Planck.")
parser.add_argument("--almin",     type=int,  default=None,help="Minimum multipole for ACT.")
parser.add_argument("--almax",     type=int,  default=6000,help="Maximum multipole for ACT.")
parser.add_argument("--klmin",     type=int,  default=40,help="Minimum multipole for recon.")
parser.add_argument("--klmax",     type=int,  default=5000,help="Maximum multipole for recon.")
parser.add_argument("--lxcut",     type=int,  default=None,help="Lxcut for ACT.")
parser.add_argument("--lycut",     type=int,  default=None,help="Lycut for ACT.")
parser.add_argument("--arcmax",     type=float,  default=15,help="Maximum arcmin distance for binning.")
parser.add_argument("--arcstep",     type=float,  default=1.0,help="Step arcmin for binning.")
parser.add_argument("--stamp-width-arcmin",     type=float,  default=128.0,help="Stamp width arcmin.")
parser.add_argument("--pix-width-arcmin",     type=float,  default=0.5,help="Stamp width arcmin.")
parser.add_argument("--no-fit-noise", action='store_true',help='If True, do not fit empirical noise, but use RMS values specified in plc-rms, act-150-rms and act-90-rms.')
parser.add_argument("--plc-rms",     type=float,  default=35.0,help="Planck RMS noise to assume either in sims or forced noise powers or both.")
parser.add_argument("--act-150-rms",     type=float,  default=15.0,help="ACT 150 RMS noise to assume either in sims or forced noise powers or both.")
parser.add_argument("--act-90-rms",     type=float,  default=20.0,help="ACT 90 RMS noise to assume either in sims or forced noise powers or both.")
parser.add_argument("--tap-per",     type=float,  default=12.0,help="Taper percentage.")
parser.add_argument("--pad-per",     type=float,  default=3.0,help="Pad percentage.")
parser.add_argument("--debug-fit",     type=str,  default=None,help="Which fit to debug.")
parser.add_argument("--debug-anomalies",     type=str,  default=None,help="Whether to save plots of excluded anomalous stamps.")
parser.add_argument("--debug-powers",     type=str,  default=None,help="Whether to plot various power spectra from each stamp.")
parser.add_argument("--no-90", action='store_true',help='Do not use the 90 GHz map.')
parser.add_argument("--no-sz-sub", action='store_true',help='Use the high-res maps without SZ subtraction.')
parser.add_argument("--inject-sim", action='store_true',help='Instead of using data, simulate a lensing cluster and Planck+ACT (or unlensed for mean-field).')
parser.add_argument("--lensed-sim-version",     type=str,  default="lensed_maps_2e14_m180m0_c3.2_z0.7_v1_swidth_128.00_pwidth_0.50_bfact_2_pfact_5_lorder_5",help="Default lensed sims to inject.")
parser.add_argument("-o","--overwrite", action='store_true',help='Overwrite existing version.')
parser.add_argument("--is-meanfield", action='store_true',help='This is a mean-field run.')
parser.add_argument("--night-only", action='store_true',help='Use night-only maps.')
parser.add_argument("--act-only-in-hres", action='store_true',help='Use ACT only maps in high-res instead of ACT+Planck.')
args = parser.parse_args()

if args.act_only_in_hres:
    almin = args.almin if args.almin is not None else 500
else:
    almin = args.almin if args.almin is not None else 200

if args.act_only_in_hres:
    lycut = args.lycut if args.lycut is not None else 90
else:
    lycut = args.lycut if args.lycut is not None else 2

if args.act_only_in_hres:
    lxcut = args.lxcut if args.lxcut is not None else 50
else:
    lxcut = args.lxcut if args.lxcut is not None else 2


start = t.time() 
p = cutils.p # directory paths dictionary

"""
We will save results to a directory in paths.yml:scratch.
To decide on the name and to ensure that any meanfields we make
have identical noise properties, we build some strings:
"""

dstr = "night" if args.night_only else "daynight"
apstr = "act" if args.act_only_in_hres else "act_planck"
mstr = "_meanfield" if args.is_meanfield else ""
n90str = "_no90" if args.no_90 else ""

# The directory name string
vstr = f"{args.version}_{args.cat_type}_plmin_{args.plmin}_plmax_{args.plmax}_almin_{almin}_almax_{args.almax}_klmin_{args.klmin}_klmax_{args.klmax}_lxcut_{lxcut}_lycut_{lycut}_swidth_{args.stamp_width_arcmin:.2f}_tapper_{args.tap_per:.2f}_padper_{args.pad_per:.2f}_{dstr}_{apstr}{n90str}{mstr}"


# Load a fiducial CMB theory object
theory = cosmology.default_theory()

if not(args.inject_sim):
    # Load the catalog
    ras,decs = cutils.catalog_interface(args.cat_type,args.is_meanfield,args.nmax)
    nsims = len(ras)
else:
    csim = cutils.Simulator(args.is_meanfield,args.stamp_width_arcmin,args.pix_width_arcmin,args.lensed_sim_version,
                            plc_rms=args.plc_rms,act_150_rms=args.act_150_rms,act_90_rms=args.act_90_rms)
    nsims = args.nmax
    assert nsims is not None
    





if not(args.inject_sim):
    with bench.show("load maps"):
        fplc_map = p['data']+"planck_smica_nosz_reproj.fits"
        try:
            pmap = enmap.read_map(fplc_map,delayed=False)
        except:
            plc_map = p['planck_data']+"COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
            # reproject the Planck map (healpix -> CAR) 
            fshape, fwcs = enmap.fullsky_geometry(res=2.*utils.arcmin, proj='car')
            pmap = reproject.enmap_from_healpix(plc_map, fshape, fwcs, ncomp=1, unit=1, lmax=6000, rot="gal,equ")
            enmap.write_map(fplc_map,pmap)


        # ACT coadd map
        if args.no_sz_sub:
            act_map = p['coadd_data'] + f'{apstr}_s08_s18_cmb_f150_{dstr}_srcfree_map.fits'
            amap_150 = enmap.read_map(act_map,delayed=False,sel=np.s_[0,...])
        else:
            act_map = p['data'] + f'modelSubtracted150_{apstr}_{dstr}.fits'
            amap_150 = enmap.read_map(act_map,delayed=False)

        # ACT coadd map
        if args.no_sz_sub:
            act_map = p['coadd_data'] + f'{apstr}_s08_s18_cmb_f090_{dstr}_srcfree_map.fits'
            amap_90 = enmap.read_map(act_map,delayed=False,sel=np.s_[0,...])
        else:
            act_map = p['data'] + f'modelSubtracted90_{apstr}_{dstr}.fits'
            amap_90 = enmap.read_map(act_map,delayed=False)

        ivar_map = p['coadd_data'] + f'{apstr}_s08_s18_cmb_f090_{dstr}_ivar.fits'
        imap_90 = enmap.read_map(ivar_map,delayed=False,sel=np.s_[0,...])


# stamp size and resolution 
stamp_width_deg = args.stamp_width_arcmin/60.
pixel = args.pix_width_arcmin

maxr = stamp_width_deg*utils.degree/2.

if not(args.inject_sim):
    # Remove objects that lie in unobserved regions
    with bench.show("cull"):
        coords = np.stack([decs, ras])*utils.degree
        ipixs = imap_90.sky2pix(coords).astype(np.int)
        Ny,Nx = imap_90.shape
        pixs = []
        sel = np.logical_and(np.logical_and(np.logical_and(ipixs[0]>0,ipixs[0]<Ny),ipixs[1]>0),ipixs[1]<Nx)
        ras = ras[sel]
        pixs.append( ipixs[0][sel] )
        decs = decs[sel]
        pixs.append( ipixs[1][sel])
        pixs = np.stack(pixs)
        ras = ras[np.argwhere(imap_90[pixs[0,:],pixs[1,:]]>0)][:,0]
        decs = decs[np.argwhere(imap_90[pixs[0,:],pixs[1,:]]>0)][:,0]
        nsims = len(ras)
    del pixs,ipixs


# MPI paralellization
comm, rank, my_tasks = mpi.distribute(nsims)


savedir = p['scratch'] + f"/{vstr}/"
debugdir = p['scratch'] + f"/{vstr}/debug/"

overwrite = args.overwrite
if not(overwrite):
    assert not(os.path.exists(savedir)), \
   "This version already exists on disk. Please use a different version identifier or use the overwrite argument."
if rank==0:
    try: 
        os.makedirs(savedir)
    except:
        if overwrite: pass
        else: raise
    try: 
        os.makedirs(debugdir)
    except:
        if overwrite: pass
        else: raise
comm.Barrier()

s = stats.Stats(comm)



# function for fitting 1D power spectrum of given stamp 
def fit_p1d(l_edges,cents, p1d, which,xout,bfunc1,bfunc2,rms=None,lmin=None,lmax=None):
    b1 = bfunc1 if bfunc1 is not None else lambda x: 1
    b2 = bfunc2 if bfunc2 is not None else lambda x: 1
    if args.inject_sim:
        tfunc = lambda x: theory.uCl('TT',x) * b1(x) * b2(x)
    else:
        tfunc = lambda x: theory.lCl('TT',x) * b1(x) * b2(x)

    if args.no_fit_noise:

        x = xout
        ret = tfunc(x) + (rms*np.pi/180./60.)**2.

        if which==args.debug_fit:
            pl = io.Plotter('Cell')
            pl.add(cents,p1d,ls="none",marker='o')
            pl.add(xout,ret,ls="none",marker='o')
            pl._ax.set_ylim(1e-7,1)
            pl._ax.set_xlim(0,6000)
            pl.done(f'{debugdir}fcl.png')
            sys.exit()

    else:
        sel = np.logical_and(cents>lmin,cents<lmax)
        delta_ells = np.diff(l_edges)[sel]
        ells = cents[sel]
        cls = p1d[sel]
        cltt = tfunc(ells)
        if (which=='act' or which=='act_cross') and (args.act_only_in_hres):
            if which=='act':
                w0 = 20.
                sigma2 = stats.get_sigma2(ells,cltt,w0,delta_ells,fsky,ell0=3000,alpha=-4)
            elif which=='act_cross':
                w0 = 20
                w0p = 20
                ell0 = 3000
                ell0p = 2000
                sigma2 = stats.get_sigma2(ells,cltt,w0,delta_ells,fsky,ell0=ell0,alpha=-4,w0p=w0p,ell0p=ell0p,alphap=-4,clxx=cltt,clyy=cltt)
            func = stats.fit_cltt_power(ells,cls,tfunc,w0,sigma2,ell0=3000,alpha=-4,fix_knee=False)
        elif (which=='plc') or  ((which=='act' or which=='act_cross') and not(args.act_only_in_hres)):
            w0 = 40
            sigma2 = stats.get_sigma2(ells,cltt,w0,delta_ells,fsky,ell0=0,alpha=1)
            func = stats.fit_cltt_power(ells,cls,tfunc,w0,sigma2,ell0=0,alpha=1,fix_knee=True)
        elif (which=='apcross'):
            w0 = 40
            w0p = 20
            ell0 = 0
            ell0p = 3000 if (args.act_only_in_hres) else 0
            sigma2 = stats.get_sigma2(ells,cltt,w0,delta_ells,fsky,ell0=ell0,alpha=0,w0p=w0p,ell0p=ell0p,
                                      alphap=-4 if (args.act_only_in_hres) else 1,clxx=cltt,clyy=cltt)
            func = stats.fit_cltt_power(ells,cls,tfunc,w0,sigma2,ell0=ell0p,alpha=-4 if (args.act_only_in_hres) else 1,
                                        fix_knee=True if not(args.act_only_in_hres) else False)

        ret = func(xout)

        if which==args.debug_fit:
            pl = io.Plotter('Dell')
            ls = np.arange(10000)
            pl.add(ls,func(ls))
            pl.add(ls,tfunc(ls),ls='--')
            pl.add_err(cents[sel],p1d[sel],yerr=np.sqrt(sigma2),ls="none",marker='o')
            pl._ax.set_ylim(1e-1,1e5)
            pl.done(f'{debugdir}fcl.png')
            sys.exit()



    ret[xout<2] = 0
    assert np.all(np.isfinite(ret))
    return ret




# beam and FWHM 
plc_beam_fwhm = cutils.plc_beam_fwhm

# Planck mask
xlmin = args.plmin
xlmax = args.plmax

# ACT mask
almax = args.almax
ilcmin = 200
ilcmax = 8000
ylmin = almin
ylmax = almax

# kappa mask
klmin = args.klmin
klmax = args.klmax


# for binned kappa profile 
bin_edges = np.arange(0, args.arcmax, args.arcstep)
centers = (bin_edges[1:] + bin_edges[:-1])/2.

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap,bin_edges)
    cents,ret = binner.bin(data)
    return ret


def ilc(modlmap,m1,m2,p11,p22,p12,b1,b2):
    # A simple two array ILC solution
    # Returns beam deconvolved
    sel = np.logical_and(modlmap>=ilcmin,modlmap<=ilcmax)
    nells = modlmap[sel].size
    cov = np.zeros((nells,2,2))
    cov[:,0,0] = p11[sel]
    cov[:,1,1] = p22[sel]
    cov[:,0,1] = p12[sel]
    cov[:,1,0] = p12[sel]
    ms = np.stack([m1[sel],m2[sel]]).swapaxes(0,1)
    rs = np.stack([b1[sel],b2[sel]]).swapaxes(0,1)
    num = np.linalg.solve(cov,ms)
    den = np.linalg.solve(cov,rs)
    tcov = (1./np.einsum('ij,ij->i',rs,den))
    ksolve = np.einsum('ij,ij->i',rs,num)*tcov
    assert np.all(np.isfinite(ksolve))
    ret = m1*0
    ret[sel] = ksolve
    tret = p11*0
    tret[sel] = tcov
    return ret,tret
    

    

j = 0 # local counter for this MPI task
for task in my_tasks:
    i = task # global counter for all objects
    if rank==0: print(f'Rank {rank} performing task {task} as index {j}')

    if not(args.inject_sim):
        coords = np.array([decs[i], ras[i]])*utils.degree
        ivar_90 = reproject.thumbnails_ivar(imap_90, coords, r=maxr, res=pixel*utils.arcmin, extensive=True, proj="plain")
        if ivar_90 is None: 
            print(f'{task} has no ivar 90 stamp')
            continue
        if np.all(ivar_90<=1e-10):
            print(f'{task} has empty ivar 90 stamp')
            continue
        if np.any(ivar_90 <= 1e-10): 
            print(f'{task} has high 90 noise')
            if args.debug_anomalies: io.plot_img(ivar_90,f'{debugdir}act_90_err_ivar_large_var_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            continue


        # cut out a stamp from the ACT map (CAR -> plain) 
        astamp_150 = reproject.thumbnails(amap_150, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", oversample=2)
        astamp_90 = reproject.thumbnails(amap_90, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", oversample=2)


        ##### temporary 1: avoid weird noisy ACT stamps
        if np.any(astamp_150 >= 1e3) or np.any(astamp_90 >= 1e3): 
            print(f'{task} has anomalously high ACT 150 or 90')
            if args.debug_anomalies: io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_large_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            if args.debug_anomalies: io.plot_img(astamp_90,f'{debugdir}act_90_err_stamp_large_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            continue
        try:
            if not(np.all(np.isfinite(astamp_90))): raise ValueError
            if not(np.all(np.isfinite(ivar_90))): raise ValueError
        except:
            print(f'{task} has anomalous stamp')
            if args.debug_anomalies: io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_an_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            if args.debug_anomalies: io.plot_img(astamp_90,f'{debugdir}act_90_err_stamp_an_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            continue

        # cut out a stamp from the Planck map (CAR -> plain) 
        pstamp = reproject.thumbnails(pmap, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", oversample=2)

        assert wcsutils.equal(astamp_150.wcs,astamp_90.wcs)
        assert wcsutils.equal(astamp_150.wcs,pstamp.wcs)
        assert wcsutils.equal(astamp_150.wcs,ivar_90.wcs)

        # unit: K -> uK 
        if pstamp is None: 
            print(f'{task} has no planck stamp')
            continue
        if not(np.all(np.isfinite(pstamp))): 
            print(f'{task} has anomalous planck stamp; not finite')
            if args.debug_anomalies: io.plot_img(pstamp,f'{debugdir}planck_err_stamp_notfinite_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
            continue

        pstamp = pstamp[0]*1e6

    else:
        pstamp,astamp_150,astamp_90 = csim.get_obs(task)


    if j==0:
        # get an edge taper map and apodize
        taper = maps.get_taper(astamp_150.shape, astamp_150.wcs, taper_percent=args.tap_per, pad_percent=args.pad_per, weight=None)
        taper = taper[0]

    # applying this to the stamp makes it have a nice zeroed edge!    
    act_stamp_150 = astamp_150*taper
    act_stamp_90 = astamp_90*taper
    plc_stamp = pstamp*taper 

    k150 = enmap.fft(act_stamp_150,normalize='phys')
    if not(args.no_90):
        k90 = enmap.fft(act_stamp_90,normalize='phys')
    kp = enmap.fft(plc_stamp,normalize='phys')


    if j==0:

        shape = astamp_150.shape  
        wcs = astamp_150.wcs
        modlmap = enmap.modlmap(shape, wcs)

        bfunc150 = cutils.load_beam('f150')
        bfunc90 = cutils.load_beam('f090')

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid 
        act_150_kbeam2d = bfunc150(modlmap)
        act_90_kbeam2d = bfunc90(modlmap)
        plc_kbeam2d = maps.gauss_beam(modlmap,plc_beam_fwhm)

        ## lensing noise curves require CMB power spectra
        ## this could be from theory (CAMB) or actual map

        # get theory spectrum - this should be the lensed spectrum!
        ells = np.arange(8000)
        cltt = theory.lCl('TT',ells)

        ## interpolate ells and cltt 1D power spectrum specification 
        ## isotropically on to the Fourier 2D space grid

        # build interpolated 2D Fourier CMB from theory and maps 
        ucltt = maps.interp(ells, cltt)(modlmap)

        # bin size and range for 1D binned power spectrum 
        minell = 2*maps.minimum_ell(shape,wcs)
        l_edges = np.arange(minell/2,8001,minell)
        lbinner = stats.bin2D(modlmap,l_edges)
        w2 = np.mean(taper**2)
        fsky = enmap.area(shape,wcs) * w2  / 4./np.pi

        # build a Fourier space mask    
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=lxcut, lycut=lycut)
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        modrmap = enmap.modrmap(shape,wcs)

    pow = lambda x,y: (x*y.conj()).real
    # measure the binned power spectrum from given stamp 
    act_cents, act_p1d_150 = lbinner.bin(pow(k150,k150)/w2) 
    if not(args.no_90):
        act_cents, act_p1d_90 = lbinner.bin(pow(k90,k90)/w2) 
        act_cents, act_p1d_150_90 = lbinner.bin(pow(k150,k90)/w2) 
    plc_cents, plc_p1d = lbinner.bin(pow(kp,kp)/w2)


    # fit 1D power spectrum 
    tclaa_150 = fit_p1d(l_edges,act_cents, act_p1d_150, 'act',modlmap,bfunc150,bfunc150,rms=args.act_150_rms,lmin=500,lmax=8000)
    if not(args.no_90):
        tclaa_90 = fit_p1d(l_edges,act_cents, act_p1d_90, 'act',modlmap,bfunc90,bfunc90,rms=args.act_90_rms,lmin=500,lmax=8000)
        tclaa_150_90 = fit_p1d(l_edges,act_cents, act_p1d_150_90, 'act_cross',modlmap,bfunc150,bfunc90,rms=0,lmin=500,lmax=8000)
    tclpp = fit_p1d(l_edges,plc_cents, plc_p1d, 'plc',modlmap,lambda x: maps.gauss_beam(x,cutils.plc_beam_fwhm),lambda x: maps.gauss_beam(x,cutils.plc_beam_fwhm),rms=args.plc_rms,lmin=200,lmax=3000) 


    # Do ILC coadd
    if not(args.no_90):
        act_kmap,tclaa = ilc(modlmap,k150,k90,tclaa_150,tclaa_90,tclaa_150_90,act_150_kbeam2d,act_90_kbeam2d)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            act_kmap = k150 / act_150_kbeam2d
            tclaa = tclaa_150 / act_150_kbeam2d**2.
        act_kmap[~np.isfinite(act_kmap)] = 0
        tclaa[~np.isfinite(tclaa)] = 0


    ## total TT spectrum includes beam-deconvolved noise
    ## so create a total beam-deconvolved spectrum using a Gaussian beam func.

    tclpp = tclpp/(plc_kbeam2d**2.)
    tclaa[~np.isfinite(tclaa)] = 0
    tclpp[~np.isfinite(tclpp)] = 0

    ## the noise was specified for a beam deconvolved map 
    ## so we deconvolve the beam from our map

    # get a beam deconvolved Fourier map
    plc_kmap = kp/plc_kbeam2d
    act_kmap[~np.isfinite(act_kmap)] = 0
    plc_kmap[~np.isfinite(plc_kmap)] = 0

    cents,c_ap = lbinner.bin(pow(act_kmap,plc_kmap)/w2)
    tclap = fit_p1d(l_edges,cents,c_ap, 'apcross',modlmap,None,None,rms=0,lmin=500,lmax=3000) 


    # build symlens dictionary 
    feed_dict = {
        'uC_T_T' : ucltt,     # goes in the lensing response func = lensed theory 
        'tC_A_T_A_T' : tclaa, # the fit ACT power spectrum with ACT beam deconvolved
        'tC_P_T_P_T' : tclpp, # approximate Planck power spectrum with Planck beam deconvolved 
        'tC_A_T_P_T' : tclap, # same lensed theory as above, no instrumental noise  
        'tC_P_T_A_T' : tclap, # same lensed theory as above, no instrumental noise  
        'X' : plc_kmap,       # Planck map
        'Y' : act_kmap        # ACT map
    }

    for key in feed_dict.keys():
        assert np.all(np.isfinite(feed_dict[key]))

    # ask for reconstruction in Fourier space
    cqe = symlens.QE(shape,wcs,feed_dict,estimator='hdv',XY='TT',
                     xmask=xmask,ymask=ymask, field_names=['P','A'],
                     groups=None,kmask=kmask)
    krecon = cqe.reconstruct(feed_dict,xname='X_l1',yname='Y_l2',physical_units=True)

    # transform to real space for unweighted stack
    kappa = enmap.ifft(krecon, normalize='phys').real

    ##### temporary 3: to get rid of stamps with tSZ cluster in random locations
    if np.any(np.abs(kappa) > 15): 
        print(f'{task} has large kappa')
        if args.debug_anomalies: io.plot_img(kappa,f'{debugdir}kappa_err_stamp_large_kappa_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        if args.debug_anomalies: io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_large_kappa_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue


    if args.debug_powers:
        s.add_to_stats('n150',lbinner.bin(tclaa_150)[1])
        if not(args.no_90):
            s.add_to_stats('n90',lbinner.bin(tclaa_90)[1])
            s.add_to_stats('nc',lbinner.bin(tclaa_150_90)[1])
        s.add_to_stats('np',lbinner.bin(tclpp)[1])

    s.add_to_stack('ustack',kappa)

    # Noise curve for lensing
    Al = cqe.Al

    # Nl = symlens.N_l(shape,wcs,feed_dict,estimator='hdv',XY='TT',
    #     xmask=xmask,ymask=ymask,
    #     Al=Al,field_names=['P','A'],kmask=kmask,power_name="t")

    Nl = symlens.N_l_from_A_l_optimal(shape,wcs,Al)
    cents,bnl = lbinner.bin(Nl)
    nmean = bnl[np.logical_and(cents>3000,cents<5000)].mean()
    if args.debug_powers:
        s.add_to_stats('nl',bnl)

    # if np.any(lbinner.bin(Nl)[1]<0):
    #     ls = lbinner.bin(Nl)[0]
    #     print(task,ls[lbinner.bin(Nl)[1]<0])
    #     raise ValueError

    # print(Nl.shape)
    # pl = io.Plotter('CL')
    # pl.add(ells,theory.gCl('kk',ells),color='k')
    # pl.add(*lbinner.bin(Nl),ls='--')
    # pl.done(f'{debugdir}nlkk.png')
    # io.plot_img(np.fft.fftshift(np.log10(Nl)),f'{debugdir}nl2d.png',arc_width=args.stamp_width_arcmin)
    # sys.exit()



    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iNl = ( 1./ Nl) * kmask
    iNl[~np.isfinite(iNl)] = 0
    wkrecon = krecon*iNl
    s.add_to_stack('wk_real',wkrecon.real)
    s.add_to_stack('wk_imag',wkrecon.imag)
    s.add_to_stack('wk_iwt',iNl)


    # inverse variance noise weighting
    ivmean = 1./nmean

    weight = ivmean
    stack = kappa*weight
    s.add_to_stack('kmap', stack)  

    wbinned = bin(stack, modrmap*(180*60/np.pi), bin_edges)
    binned = bin(kappa, modrmap*(180*60/np.pi), bin_edges)
    s.add_to_stats('wk1d', wbinned)
    s.add_to_stats('k1d', binned)
    s.add_to_stats('kw', (weight,))
    s.add_to_stats('kw2', (weight**2,))


    # check how many stamps are cut out of given map
    s.add_to_stats('ct_selected',(1,))
    j = j + 1



# collect from all MPI cores and calculate stacks
s.get_stacks()
s.get_stats()

if rank==0:

    with bench.show("dump"):
        s.dump(savedir)
        enmap.write_map_geometry(f'{savedir}/map_geometry.fits',shape,wcs)
        enmap.write_map(f'{savedir}/kmask.fits',kmask)
        enmap.write_map(f'{savedir}/modrmap.fits',modrmap)
        np.savetxt(f'{savedir}/bin_edges.txt',bin_edges)

    for ctkey in ['selected']:#,'large_kappa','no_stamp','high_noise','high_stamp','anomalous']:
        try:
            N = s.vectors[f'ct_{ctkey}'].sum()
            print(f'Number {ctkey} : {N}')
        except:
            pass
    N_stamp = s.vectors[f'ct_{ctkey}'].sum()
    assert N_stamp==s.stack_count['kmap']
    assert N_stamp==s.vectors['kw'].shape[0]


    if args.debug_powers:
        pl = io.Plotter('CL')
        pl.add(ells,theory.gCl('kk',ells),color='k')
        for Nl in s.vectors['nl']:
            pl.add(lbinner.centers,Nl,ls='--',alpha=0.2)
        pl.done(f'{savedir}nlkk.png')


    if args.debug_powers:
        pl = io.Plotter('Cell')
        for i in range(s.vectors['n150'].shape[0]):
            pl.add(act_cents,s.vectors['n150'][i])
        pl.done(f'{savedir}n150.png')

        if not(args.no_90):
            pl = io.Plotter('Cell')
            for i in range(s.vectors['n90'].shape[0]):
                pl.add(act_cents,s.vectors['n90'][i])
            pl.done(f'{savedir}n90.png')

            pl = io.Plotter('Cell')
            for i in range(s.vectors['nc'].shape[0]):
                pl.add(act_cents,s.vectors['nc'][i])
            pl.done(f'{savedir}nc.png')


        pl = io.Plotter('Cell')
        for i in range(s.vectors['np'].shape[0]):
            pl.add(plc_cents,s.vectors['np'][i])
        pl.done(f'{savedir}np.png')


    elapsed = t.time() - start
    print("\r ::: entire run took %.1f seconds" %elapsed)

