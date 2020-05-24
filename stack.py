import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
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
parser.add_argument("cat_path", type=str,help='Catalog path relative to data directory.')
parser.add_argument("-N", "--nmax",     type=int,  default=None,help="Limit number of objects used e.g. for debugging or quick tests.")
parser.add_argument("--plmin",     type=int,  default=200,help="Minimum multipole for Planck.")
parser.add_argument("--plmax",     type=int,  default=2000,help="Maximum multipole for Planck.")
parser.add_argument("--almin",     type=int,  default=500,help="Minimum multipole for ACT.")
parser.add_argument("--almax",     type=int,  default=6000,help="Maximum multipole for ACT.")
parser.add_argument("--klmin",     type=int,  default=40,help="Minimum multipole for recon.")
parser.add_argument("--klmax",     type=int,  default=5000,help="Maximum multipole for recon.")
parser.add_argument("--lxcut",     type=int,  default=90,help="Lxcut for ACT.")
parser.add_argument("--lycut",     type=int,  default=50,help="Lycut for ACT.")
parser.add_argument("--arcmax",     type=float,  default=15,help="Maximum arcmin distance for binning.")
parser.add_argument("--arcstep",     type=float,  default=1.0,help="Step arcmin for binning.")
parser.add_argument("--stamp-width-arcmin",     type=float,  default=128.0,help="Stamp width arcmin.")
parser.add_argument("--tap-per",     type=float,  default=12.0,help="Taper percentage.")
parser.add_argument("--pad-per",     type=float,  default=3.0,help="Pad percentage.")
parser.add_argument("--debug-fit",     type=str,  default=None,help="Which fit to debug.")
parser.add_argument("--no-sz-sub", action='store_true',help='Use the high-res maps without SZ subtraction.')
parser.add_argument("--inject-sim", action='store_true',help='Instead of using data, simulate a lensing cluster and Planck+ACT (or unlensed for mean-field).')
parser.add_argument("-o","--overwrite", action='store_true',help='Overwrite existing version.')
parser.add_argument("--is-meanfield", action='store_true',help='This is a mean-field run.')
parser.add_argument("--night-only", action='store_true',help='Use night-only maps.')
parser.add_argument("--planck-in-hres", action='store_true',help='Use Planck+ACT maps in high-res.')
args = parser.parse_args()


dstr = "night" if args.night_only else "daynight"
apstr = "act_planck" if args.planck_in_hres else "act"
mstr = "_meanfield" if args.is_meanfield else ""

vstr = f"{args.version}_plmin_{args.plmin}_plmax_{args.plmax}_almin_{args.almin}_almax_{args.almax}_klmin_{args.klmin}_klmax_{args.klmax}_lxcut_{args.lxcut}_lycut_{args.lycut}_swidth_{args.stamp_width_arcmin:.2f}_tapper_{args.tap_per:.2f}_padper_{args.pad_per:.2f}_{dstr}_{apstr}{mstr}"

# if args.meanfield_version:
#     if args.is_meanfield: raise ValueError
#     mvstr = f"{args.meanfield_version}_plmin_{args.plmin}_plmax_{args.plmax}_almin_{args.almin}_almax_{args.almax}_klmin_{args.klmin}_klmax_{args.klmax}_lxcut_{args.lxcut}_lycut_{args.lycut}_swidth_{args.stamp_width_arcmin:.2f}_tapper_{args.tap_per:.2f}_padper_{args.pad_per:.2f}_{dstr}_{apstr}_meanfield"
#     s_mf,shape_mf,wcs_mf = cutils.load_meanfields(mvstr)
    

p = cutils.p # directory paths dictionary
start = t.time() 

theory = cosmology.default_theory()

# ACT catalogue :D
catalogue_name = p['data']+ args.cat_path
#"AdvACT_S18Clusters_v1.0-beta.fits" #[4024] 
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg'][:args.nmax]
decs = hdu[1].data['DECDeg'][:args.nmax]


N_cluster = len(ras)
nsims = N_cluster


# MPI paralellization! 
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


# Planck tSZ deprojected map
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



# function for fitting 1D power spectrum of given stamp 
def fit_p1d(cents, p1d, which,xout,fwhm1,fwhm2):

    ells = cents
    cltt = p1d

    logy = cltt*ells**2.
       
    def ffunc(x, a, b,c):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b1 = maps.gauss_beam(x,fwhm1) if fwhm1 is not None else 1
            b2 = maps.gauss_beam(x,fwhm2) if fwhm2 is not None else 1
            return (maps.rednoise(x,a,lknee=b,alpha=c) + theory.lCl('TT',x) * b1* b2)*x**2

    if which=='act' or which=='act_cross': 
        popt, pcov = curve_fit(lambda x,a,b: ffunc(x,a,b,-4), ells, logy,p0=[20,3000],bounds=([4,400],[100,4000]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = ffunc(xout, *popt,-4)/xout**2
    elif which=='plc': 
        popt, pcov = curve_fit(lambda x,a: ffunc(x,a,0,1), ells, logy,p0=[30])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = ffunc(xout, *popt,0,1)/xout**2
    elif which=='apcross': 
        popt, pcov = curve_fit(lambda x,a: ffunc(x,a,0,1), ells, logy,p0=[2],bounds=(0,100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = ffunc(xout, *popt,0,1)/xout**2

    if which==args.debug_fit:
        pl = io.Plotter('Cell')
        ls = np.arange(6000)
        if which=='plc' or which=='apcross': pl.add(ls,ffunc(ls,*popt,0,1)/ls**2)
        else: pl.add(ls,ffunc(ls,*popt,-4)/ls**2)
        pl.add(cents,p1d,ls="none",marker='o')
        pl._ax.set_ylim(1e-7,1)
        pl.done(f'{debugdir}fcl.png')
        sys.exit()

    ret[xout<2] = 0
    assert np.all(np.isfinite(ret))
    return ret



# stamp size and resolution 
stamp_width_deg = args.stamp_width_arcmin/60.
pixel = 0.5

# beam and FWHM 
plc_beam_fwhm = 5.

# Planck mask
xlmin = args.plmin
xlmax = args.plmax

# ACT mask
almin = args.almin
almax = args.almax
ilcmin = 200
ilcmax = 8000
ylmin = almin
ylmax = almax
lxcut = args.lxcut
lycut = args.lycut

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

def load_beam(freq,ells):
    if freq=='f150': fname = p['data']+'s16_pa2_f150_nohwp_night_beam_tform_jitter.txt'
    elif freq=='f090': fname = p['data']+'s16_pa3_f090_nohwp_night_beam_tform_jitter.txt'
    ls,bls = np.loadtxt(fname,usecols=[0,1],unpack=True)
    assert ls[0]==0
    bls = bls / bls[0]
    return maps.interp(ls,bls)(ells)

def ilc(modlmap,m1,m2,p11,p22,p12,b1,b2):
    # A simple two array ILC solution
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
    

    
maxr = stamp_width_deg*utils.degree/2.


for j,task in enumerate(my_tasks):
    
    i = task
    print(f'Rank {rank} performing task {task} as index {j}')
    ## extract a postage stamp from a larger map
    ## by reprojecting to a coordinate system centered on the given position 

    coords = np.array([decs[i], ras[i]])*utils.degree

    ivar_90 = reproject.thumbnails_ivar(imap_90, coords, r=maxr, res=pixel*utils.arcmin, extensive=True, proj="plain")
    if ivar_90 is None: 
        print(f'{task} has no ivar 90 stamp')
        continue
    if np.any(ivar_90 <= 1e-10): 
        print(f'{task} has high 90 noise')
        if not(args.is_meanfield): io.plot_img(ivar_90,f'{debugdir}act_90_err_ivar_large_var_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue


    # cut out a stamp from the ACT map (CAR -> plain) 
    astamp_150 = reproject.thumbnails(amap_150, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", apod=0,oversample=2)
    astamp_90 = reproject.thumbnails(amap_90, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", apod=0,oversample=2)


    ##### temporary 1: avoid weird noisy ACT stamps
    if np.any(astamp_150 >= 1e3) or np.any(astamp_90 >= 1e3): 
        print(f'{task} has anomalously high ACT 150 or 90')
        if not(args.is_meanfield): io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_large_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        if not(args.is_meanfield): io.plot_img(astamp_90,f'{debugdir}act_90_err_stamp_large_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue
    try:
        if not(np.all(np.isfinite(astamp_90))): raise ValueError
        if not(np.all(np.isfinite(ivar_90))): raise ValueError
    except:
        print(f'{task} has anomalous stamp')
        if not(args.is_meanfield): io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_an_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        if not(args.is_meanfield): io.plot_img(astamp_90,f'{debugdir}act_90_err_stamp_an_stamp_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue

    # cut out a stamp from the Planck map (CAR -> plain) 
    pstamp = reproject.thumbnails(pmap, coords, r=maxr, res=pixel*utils.arcmin, proj="plain", apod=0,oversample=2)

    assert wcsutils.equal(astamp_150.wcs,astamp_90.wcs)
    assert wcsutils.equal(astamp_150.wcs,pstamp.wcs)
    assert wcsutils.equal(astamp_150.wcs,ivar_90.wcs)

    # unit: K -> uK 
    if pstamp is None: 
        print(f'{task} has no planck stamp')
        continue
    if not(np.all(np.isfinite(pstamp))): 
        print(f'{task} has anomalous planck stamp; not finite')
        if not(args.is_meanfield): io.plot_img(pstamp,f'{debugdir}planck_err_stamp_notfinite_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue

    pstamp = pstamp[0]*1e6

    ## if we want to do any sort of harmonic analysis 
    ## we require periodic boundary conditions
    ## we can prepare an edge taper map on the same footprint as our map of interest

    if j==0:
        # get an edge taper map and apodize
        taper = maps.get_taper(astamp_150.shape, astamp_150.wcs, taper_percent=args.tap_per, pad_percent=args.pad_per, weight=None)
        taper = taper[0]

    # applying this to the stamp makes it have a nice zeroed edge!    
    act_stamp_150 = astamp_150*taper
    act_stamp_90 = astamp_90*taper
    plc_stamp = pstamp*taper 

    k150 = enmap.fft(act_stamp_150,normalize='phys')
    k90 = enmap.fft(act_stamp_90,normalize='phys')
    kp = enmap.fft(plc_stamp,normalize='phys')

    ## all outputs are 2D arrays in Fourier space
    ## so you will need some way to bin it in annuli
    ## a map of the absolute wavenumbers is useful for this : enmap.modlmap


    if j==0:

        shape = astamp_150.shape  
        wcs = astamp_150.wcs
        modlmap = enmap.modlmap(shape, wcs)

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid 
        act_150_kbeam2d = load_beam('f150',modlmap)
        act_90_kbeam2d = load_beam('f090',modlmap)
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

        # build a Fourier space mask    
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=lxcut, lycut=lycut)
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        modrmap = enmap.modrmap(shape,wcs)

    pow = lambda x,y: (x*y.conj()).real
    # measure the binned power spectrum from given stamp 
    act_cents, act_p1d_150 = lbinner.bin(pow(k150,k150)/w2) 
    act_cents, act_p1d_90 = lbinner.bin(pow(k90,k90)/w2) 
    act_cents, act_p1d_150_90 = lbinner.bin(pow(k150,k90)/w2) 
    plc_cents, plc_p1d = lbinner.bin(pow(kp,kp)/w2)


    # fit 1D power spectrum 
    tclaa_150 = fit_p1d(act_cents, act_p1d_150, 'act',modlmap,1.4,1.4)
    tclaa_90 = fit_p1d(act_cents, act_p1d_90, 'act',modlmap,2.2,2.2)
    tclaa_150_90 = fit_p1d(act_cents, act_p1d_150_90, 'act_cross',modlmap,1.4,2.2)
    tclpp = fit_p1d(plc_cents, plc_p1d, 'plc',modlmap,5.0,5.0) 


    act_kmap,tclaa = ilc(modlmap,k150,k90,tclaa_150,tclaa_90,tclaa_150_90,act_150_kbeam2d,act_90_kbeam2d)


    # cents,c11 = lbinner.bin(pow(k150,k150)/w2/act_150_kbeam2d**2)
    # cents,c22 = lbinner.bin(pow(k90,k90)/w2/act_90_kbeam2d**2)
    # cents,cilc = lbinner.bin(pow(act_kmap,act_kmap)/w2)
    # cents,cpow = lbinner.bin(tclaa)
    # pl = io.Plotter('Cell')
    # pl.add(cents,c11,label='150')
    # pl.add(cents,c22,label='90')
    # pl.add(cents,cilc,label='ilc',ls='--')
    # pl.add(cents,cpow,label='cpow',ls=':')
    # pl.done('pilc.png')


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
    tclap = fit_p1d(cents,c_ap, 'apcross',modlmap,None,None) 


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

    ## need to have a Fourier space mask in hand 
    ## that enforces what multipoles in the CMB map are included 


    # ask for reconstruction in Fourier space
    cqe = symlens.QE(shape,wcs,feed_dict,estimator='hdv',XY='TT',
                     xmask=xmask,ymask=ymask, field_names=['P','A'],
                     groups=None,kmask=kmask)
    krecon = cqe.reconstruct(feed_dict,xname='X_l1',yname='Y_l2',physical_units=True)

    # transform to real space
    kappa = enmap.ifft(krecon, normalize='phys').real
    ##### temporary 3: to get rid of stamps with tSZ cluster in random locations
    if np.any(np.abs(kappa) > 15): 
        print(f'{task} has large kappa')
        if not(args.is_meanfield): io.plot_img(kappa,f'{debugdir}kappa_err_stamp_large_kappa_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        if not(args.is_meanfield): io.plot_img(astamp_150,f'{debugdir}act_150_err_stamp_large_kappa_{task}.png',arc_width=args.stamp_width_arcmin,xlabel='$\\theta_x$ (arcmin)',ylabel='$\\theta_y$ (arcmin)')
        continue


    if not(args.is_meanfield):
        s.add_to_stats('n150',lbinner.bin(tclaa_150)[1])
        s.add_to_stats('n90',lbinner.bin(tclaa_90)[1])
        s.add_to_stats('nc',lbinner.bin(tclaa_150_90)[1])
        s.add_to_stats('np',lbinner.bin(tclpp)[1])

    s.add_to_stack('ustack',kappa)
    Al = cqe.Al


    # Nl = symlens.N_l(shape,wcs,feed_dict,estimator='hdv',XY='TT',
    #     xmask=xmask,ymask=ymask,
    #     Al=Al,field_names=['P','A'],kmask=kmask,power_name="t")

    Nl = symlens.N_l_from_A_l_optimal(shape,wcs,Al)
    cents,bnl = lbinner.bin(Nl)
    nmean = bnl[np.logical_and(cents>3000,cents<5000)].mean()
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


    # check actually how many stamps are cut out of given map
    s.add_to_stats('ct_selected',(1,))



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


    if not(args.is_meanfield):
        pl = io.Plotter('CL')
        pl.add(ells,theory.gCl('kk',ells),color='k')
        for Nl in s.vectors['nl']:
            pl.add(lbinner.centers,Nl,ls='--',alpha=0.2)
        pl.done(f'{savedir}nlkk.png')


    if not(args.is_meanfield):
        pl = io.Plotter('Cell')
        for i in range(s.vectors['n150'].shape[0]):
            pl.add(act_cents,s.vectors['n150'][i])
        pl.done(f'{savedir}n150.png')

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

