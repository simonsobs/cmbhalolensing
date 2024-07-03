# inpainting test
# do inpainting on 90 and 150 GHz separately first and then ILC

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils as cutils
from pixell import enmap, reproject, enplot, utils, wcsutils
from orphics import maps, mpi, io, stats, cosmology
from scipy.optimize import curve_fit
from numpy import save
import time
import symlens
import healpy as hp
import os, sys
from enlib import bench
import warnings

""" 
!! Run
python stack.py -h 
!! to see options
"""
start_time,paths,defaults,args,tags,rank,data_choice = cutils.initialize_pipeline_config()
if rank==0:
    print("Paths: ",paths)
    print("Tags: ",tags)
    print("Defaults: ",defaults)
    print("Arguments: ",args)
    print("Data: ",data_choice)

# Load a fiducial CMB theory object
theory = cosmology.default_theory()

if not (args.inject_sim):
    # Load the catalog
    ras, decs, zs, ws, cdata = cutils.catalog_interface(
	    args.cat_type, 
	    args.is_meanfield, 
	    args.nmax, 
	    args.zmin, 
	    args.zmax, 
	    bcg=args.bcg, 
	    snmin=args.snmin, 
	    snmax=args.snmax, 
	    y0min=args.y0min, 
	    y0max=args.y0max, 
	    decmin=args.decmin
    )
else:
    # or if injecting sims, load the sim generator
    csim = cutils.Simulator(
        args.is_meanfield,
        args.swidth,
        args.pwidth,
        args.lensed_sim_version,
        plc_rms=defaults.gradient_fiducial_rms,
        act_150_rms=defaults.highres_fiducial_rms,
        act_90_rms=defaults.highres_fiducial_rms,
    )
    nsims = args.nmax
    assert nsims is not None


""" 
!! MAP LOADING
"""

if not (args.inject_sim):
    with bench.show("load maps"):
        # Planck SMICA NO SZ map 
	    # if reprojected Planck map already exists, use it 
        fplc_map = paths.data + "planck_smica_nosz_reproj.fits"
        if not(args.full_sim_index is None):
            pmap = enmap.read_map(f'{paths.fullsim_path}/planck_sim_{args.full_sim_index:06d}.fits') / 1e6
        else:
            try:
                if not (args.ilc_maps): 
                    pmap = enmap.read_map(fplc_map, delayed=False)
                else:
                    fplc_map = (paths.act_data + data_choice.grad)
                    pmap = enmap.read_map(fplc_map, delayed=False, sel=np.s_[0, ...])
                    print(np.shape(pmap))

            except:
                plc_map = paths.data + "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
                
                # reproject the Planck map (healpix -> CAR)
                fshape, fwcs = enmap.fullsky_geometry(res=2.0 * utils.arcmin, proj="car")

                # this doesn't work with the latest pixell version (v0.20.3)
                # pmap = reproject.enmap_from_healpix(
                #     plc_map, fshape, fwcs, ncomp=1, unit=1, lmax=6000, rot="gal,equ"
                # )

                # reading the input map
                p_map = np.atleast_2d(hp.read_map(plc_map, field=tuple(range(0,1)))).astype(np.float64)

                # perform the actual transform
                pmap = reproject.healpix2map(
                    p_map, fshape, fwcs, lmax=6000, rot="gal,equ"
                )
                enmap.write_map(fplc_map, pmap)


        # ACT 150 GHz coadd map
        if not(args.full_sim_index is None):
            amap_150 = enmap.read_map(f'{paths.fullsim_path}/af150_sim_{args.full_sim_index:06d}.fits')
        else:
            if not (args.ilc_maps): 
                act_map = (paths.act_data + data_choice.hres_150)
                famap_150 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])
            else:
                act_map = (paths.act_data + data_choice.hres)
                famap_150 = enmap.read_map(act_map, delayed=False)
                print(np.shape(famap_150))

            # SZ cluster model image subtraction for 150 GHz
            if args.hres_grad:            
                if not (args.grad_noszsub): 
                    gamap_150 = famap_150 - enmap.read_map(paths.act_data + data_choice.hres_model_150)
                else: 
                    gamap_150 = famap_150
            
            if not(args.no_sz_sub):
                amap_150 = famap_150 - enmap.read_map(paths.act_data + data_choice.hres_model_150)   
            else:
                amap_150 = famap_150

        
        # ACT 90 GHz coadd map
        if not(args.full_sim_index is None):
            amap_90 = enmap.read_map(f'{paths.fullsim_path}/af090_sim_{args.full_sim_index:06d}.fits')
        else:
            act_map = (paths.act_data + data_choice.hres_090)
            famap_90 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])

            # SZ cluster model image subtraction for 90 GHz    
            if args.hres_grad:            
                if not (args.grad_noszsub): 
                    gamap_090 = famap_90 - enmap.read_map(paths.act_data + data_choice.hres_model_090)
                else: 
                    gamap_90 = famap_90

            if not(args.no_sz_sub):
                amap_90 = famap_90 - enmap.read_map(paths.act_data + data_choice.hres_model_090) 
            else:
                amap_90 = famap_90
      
        if args.day_null:
            assert args.full_sim_index is None
            assert not(args.night_only)
            assert not(args.no_90)
            assert not(args.no_150)
            assert not(args.rand_rot)
            act_map = (
                paths.coadd_data + f"{tags.apstr}_s08_{tags.s19str}_cmb_f150_night_srcfree_map.fits"
            )
            namap_150 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])
            act_map = (
                paths.coadd_data + f"{tags.apstr}_s08_{tags.s19str}_cmb_f090_night_srcfree_map.fits"
            )
            namap_90 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])

            null_map_150 = famap_150 - namap_150
            null_map_90 = famap_90 - namap_90

        # Inv var map for 90 GHz
        ivar_map = (paths.act_data + data_choice.hres_ivar)

        # if data_choice.hres_map == 'dr6': imap_90 = enmap.read_map(ivar_map, delayed=False)
        # else: imap_90 = enmap.read_map(ivar_map, delayed=False, sel=np.s_[0, ...]) ##### fix here 

        try:
            imap_90 = enmap.read_map(ivar_map, delayed=False, sel=np.s_[0, ...]) 
        except:
            imap_90 = enmap.read_map(ivar_map, delayed=False)

        rms_map = maps.rms_from_ivar(
            imap_90, cylindrical=True
        ) # convert to RMS noise map

# stamp size and resolution
stamp_width_deg = args.swidth / 60.0        # stamp_width_arcmin: 128.0
pixel = args.pwidth                         # pix_width_arcmin: 0.5
maxr = stamp_width_deg * utils.degree / 2.0 # max radius for projection geometry 

""" 
!! CATALOG TRIMMING BASED ON RMS MAP
"""

if not (args.inject_sim):
    # Remove objects that lie in unobserved regions
    Norig = len(ras)
    with bench.show("cull"):
        coords = np.stack([decs, ras]) * utils.degree
        # Convert catalog coords to pixel coords
        ipixs = rms_map.sky2pix(coords).astype(int)
        Ny, Nx = rms_map.shape
        pixs = []
        # Select pixels that fall within map
        sel = np.logical_and.reduce(
            (ipixs[0] > 0, ipixs[0] < Ny, ipixs[1] > 0, ipixs[1] < Nx)
        )
        ras = ras[sel]
        pixs.append(ipixs[0][sel])
        decs = decs[sel]
        pixs.append(ipixs[1][sel])
        ws = ws[sel]
        zs = zs[sel]
        for key in cdata.keys():
            cdata[key] = cdata[key][sel]
        pixs = np.stack(pixs)
        # Then select pixels where the noise is finite and less than args.max_rms_noise
        nsel = np.logical_and(
            rms_map[pixs[0, :], pixs[1, :]] > 0,
            rms_map[pixs[0, :], pixs[1, :]] < args.max_rms, # max_rms_noise: 50.0
        )
        ras = ras[np.argwhere(nsel)][:, 0]
        decs = decs[np.argwhere(nsel)][:, 0]
        ws = ws[np.argwhere(nsel)][:, 0]
        zs = zs[np.argwhere(nsel)][:, 0]
        for key in cdata.keys():
            cdata[key] = cdata[key][np.argwhere(nsel)][:, 0]
        nsims = len(ras)
        assert len(decs)==nsims
        assert len(ws)==nsims
    del pixs, ipixs

print(f"After applying the noise mask, {Norig} -> {nsims}.")
try:
    print(f"zmin {min(zs)} zmax {max(zs)}")
except:
    pass

# MPI paralellization
comm, rank, my_tasks = mpi.distribute(nsims)


# An MPI statistics collector
s = stats.Stats(comm)

""" 
!! EMPIRICAL POWER SPECTRUM FITTING
"""

def fit_p1d(
    l_edges, cents, p1d, which, xout, bfunc1, bfunc2, rms=None, lmin=None, lmax=None
):
    # function for fitting 1D power spectrum of given stamp
    b1 = bfunc1 if bfunc1 is not None else lambda x: 1
    b2 = bfunc2 if bfunc2 is not None else lambda x: 1
    if args.inject_sim:
        tfunc = lambda x: theory.uCl("TT", x) * b1(x) * b2(x)
    else:
        tfunc = lambda x: theory.lCl("TT", x) * b1(x) * b2(x)

    if args.no_fit_noise:
        # Use fiducial spectrum + RMS noise if no fitting requested
        x = xout
        ret = tfunc(x) + (rms * np.pi / 180.0 / 60.0) ** 2.0

        if which == args.debug_fit:
            pl = io.Plotter("Cell")
            pl.add(cents, p1d, ls="none", marker="o")
            pl.add(xout, ret, ls="none", marker="o")
            pl._ax.set_ylim(1e-7, 1)
            pl._ax.set_xlim(0, 6000)
            pl.done(f"{paths.debugdir}fcl.png")
            sys.exit()

    else:
        # PS fitting
        # Select region for fit
        sel = np.logical_and(cents > lmin, cents < lmax)
        delta_ells = np.diff(l_edges)[sel]
        ells = cents[sel]
        cls = p1d[sel]
        cltt = tfunc(ells)  # fiducial Cltt
        if (which == "act" or which == "act_cross") and (args.act_only_in_hres):
            if which == "act" or which == "act_cross":
                # Get bandpower variance estimate based on cltt + fiducial 1/f + white noise
                w0 = defaults.highres_fiducial_rms
                sigma2 = stats.get_sigma2(
                    ells,
                    cltt,
                    w0,
                    delta_ells,
                    fsky,
                    ell0=defaults.highres_fiducial_lknee,
                    alpha=defaults.highres_fiducial_alpha,
                )
            func = stats.fit_cltt_power(
                ells,
                cls,
                tfunc,
                w0,
                sigma2,
                ell0=defaults.highres_fiducial_lknee,
                alpha=defaults.highres_fiducial_alpha,
                fix_knee=False,
            )
        elif (which == "plc") or (
            (which == "act" or which == "act_cross") and not (args.act_only_in_hres)
        ):
            w0 = defaults.gradient_fiducial_rms if which=='plc' else defaults.highres_fiducial_rms
            sigma2 = stats.get_sigma2(ells, cltt, w0, delta_ells, fsky, ell0=0, alpha=1)
            func = stats.fit_cltt_power(
                ells, cls, tfunc, w0, sigma2, ell0=0, alpha=1, fix_knee=True
            )
        elif which == "apcross":
            w0 = defaults.gradient_fiducial_rms
            w0p = defaults.highres_fiducial_rms
            ell0 = 0
            ell0p = defaults.highres_fiducial_lknee if (args.act_only_in_hres) else 0
            sigma2 = stats.get_sigma2(
                ells,
                cltt,
                w0,
                delta_ells,
                fsky,
                ell0=ell0,
                alpha=0,
                w0p=w0p,
                ell0p=ell0p,
                alphap=defaults.highres_fiducial_alpha if (args.act_only_in_hres) else 1,
                clxx=cltt,
                clyy=cltt,
            )
            func = stats.fit_cltt_power(
                ells,
                cls,
                tfunc,
                w0,
                sigma2,
                ell0=ell0p,
                alpha=defaults.highres_fiducial_alpha if (args.act_only_in_hres) else 1,
                fix_knee=True if not (args.act_only_in_hres) else False,
            )

        ret = func(xout)

        if which == args.debug_fit:
            pl = io.Plotter("Dell")
            ls = np.arange(10000)
            pl.add(ls, func(ls))
            pl.add(ls, tfunc(ls), ls="--")
            pl.add_err(
                cents[sel], p1d[sel], yerr=np.sqrt(sigma2), ls="none", marker="o"
            )
            pl._ax.set_ylim(1e-1, 1e5)
            pl.done(f"{paths.debugdir}fcl.png")
            sys.exit()

    ret[xout < 2] = 0
    assert np.all(np.isfinite(ret))
    return ret


# beam and FWHM
plc_beam_fwhm = defaults.planck_smica_beam_fwhm     # 5 arcmin
ilc_beam_fwhm = defaults.ilc_dr6v3_beam_fwhm        # 1.6 arcmin

# Planck mask
xlmin = args.grad_lmin ; xlmax = args.grad_lmax     # 200, 2000

# ACT mask
ilcmin = args.ilc_lmin ; ilcmax = args.ilc_lmax     # 200, 8000
ylmin = args.hres_lmin ; ylmax = args.hres_lmax     # 200, 6000 -> 3500 

# kappa mask
klmin = args.klmin ; klmax = args.klmax             # 200, 5000 -> 3000

# for binned kappa profile
bin_edges = np.arange(0, args.arcmax, args.arcstep) # 15 arcmin, 1.5 arcmin
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

""" 
!! BINNING
"""

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret


""" 
!! ILC / COADDING
"""

def ilc(modlmap, m1, m2, p11, p22, p12, b1, b2):
    # A simple two array ILC solution
    # Returns beam deconvolved
    sel = np.logical_and(modlmap >= ilcmin, modlmap <= ilcmax)
    nells = modlmap[sel].size
    cov = np.zeros((nells, 2, 2))
    cov[:, 0, 0] = p11[sel]
    cov[:, 1, 1] = p22[sel]
    cov[:, 0, 1] = p12[sel]
    cov[:, 1, 0] = p12[sel]
    ms = np.stack([m1[sel], m2[sel]]).swapaxes(0, 1)
    rs = np.stack([b1[sel], b2[sel]]).swapaxes(0, 1)
    num = np.linalg.solve(cov, ms)
    den = np.linalg.solve(cov, rs)
    tcov = 1.0 / np.einsum("ij,ij->i", rs, den)
    ksolve = np.einsum("ij,ij->i", rs, num) * tcov
    assert np.all(np.isfinite(ksolve))
    ret = m1 * 0
    ret[sel] = ksolve
    tret = p11 * 0
    tret[sel] = tcov
    return ret, tret


""" 
!! LOOP OVER ASSIGNED TASKS
"""

j = 0  # local counter for this MPI task
for task in my_tasks:
    i = task  # global counter for all objects
    cweight = ws[i] if not(args.inject_sim) else 1
    if not(args.inject_sim) and not(args.is_meanfield):
        z = zs[i]
    else:
        z = 0
    cper = int((j + 1) / len(my_tasks) * 100.0)
    if rank == 0:
        print(f"Rank {rank} performing task {task} as index {j} ({cper}% complete.).")

    if not (args.inject_sim):
        coords = np.array([decs[i], ras[i]]) * utils.degree
        """ 
        !! 90 GHz INV-NOISE STAMP
        """
        ivar_90 = reproject.thumbnails_ivar(
            imap_90,
            coords,
            r=maxr,
            res=pixel * utils.arcmin,
            extensive=True,
            proj="tan"
        )
        """ 
        !! REJECT IF NO OBS. OR HIGH NOISE
        """
        if ivar_90 is None:
            print(f"{task} has no ivar 90 stamp")
            continue
        if np.all(ivar_90 < 1e-10):
            print(f"{task} has empty ivar 90 stamp")
            continue
        if np.any(ivar_90 < 1e-10):
            print(f"{task} has high 90 noise")
            if args.debug_anomalies:
                io.plot_img(
                    ivar_90,
                    f"{paths.debugdir}act_90_err_ivar_large_var_{task}.png",
                    arc_width=args.swidth,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue

        """ 
        !! CUT OUT 150 and 90 GHZ STAMPS
        """
        # cut out a stamp from the ACT map (CAR -> tan: gnomonic projection)
        astamp_150 = reproject.thumbnails(
            amap_150,
            coords,
            r=maxr,
            res=pixel * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=True
        )       
        astamp_90 = reproject.thumbnails(
            amap_90,
            coords,
            r=maxr,
            res=pixel * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=True
        )

	# for daynight - night null test
        if args.day_null:
            nastamp_150 = reproject.thumbnails(
                null_map_150,
                coords,
                r=maxr,
                res=pixel * utils.arcmin,
                proj="tan",
                oversample=2,
                pixwin=True
            )
            nastamp_90 = reproject.thumbnails(
                null_map_90,
                coords,
                r=maxr,
                res=pixel * utils.arcmin,
                proj="tan",
                oversample=2,
                pixwin=True
            )       

        """ 
        !! REJECT ANOMALOUS STAMPS
        """
        ##### temporary 1: avoid weird noisy ACT stamps
        if np.any(astamp_150 >= 1e3) or np.any(astamp_90 >= 1e3):
            print(f"{task} has anomalously high ACT 150 or 90")
            if args.debug_anomalies:
                io.plot_img(
                    astamp_150,
                    f"{paths.debugdir}act_150_err_stamp_large_stamp_{task}.png",
                    arc_width=args.swidth,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            if args.debug_anomalies: 
                io.plot_img(
                    astamp_90,
                    f"{paths.debugdir}act_90_err_stamp_large_stamp_{task}.png",
                    arc_width=args.swidth,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue
        try:
            if not (np.all(np.isfinite(astamp_90))):
                raise ValueError
            if not (np.all(np.isfinite(ivar_90))):
                raise ValueError
        except:
            print(f"{task} has anomalous stamp")
            if args.debug_anomalies:
                io.plot_img(
                    astamp_150,
                    f"{paths.debugdir}act_150_err_stamp_an_stamp_{task}.png",
                    arc_width=args.swidth,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            if args.debug_anomalies:
                io.plot_img(
                    astamp_90,
                    f"{paths.debugdir}act_90_err_stamp_an_stamp_{task}.png",
                    arc_width=args.swidth,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue

        if not (args.hres_grad):
            """ 
            !! CUT OUT PLANCK STAMP
            """
            # cut out a stamp from the Planck map (CAR -> tangent)
            pstamp = reproject.thumbnails(
                    pmap, coords, r=maxr, res=pixel * utils.arcmin, proj="tan", oversample=2, pixwin=True if args.ilc_maps else False
            )

            # Check that all the WCS agree
            assert wcsutils.equal(astamp_150.wcs, astamp_90.wcs)
            assert wcsutils.equal(astamp_150.wcs, pstamp.wcs)
            assert wcsutils.equal(astamp_150.wcs, ivar_90.wcs)

            """ 
            !! REJECT ANOMALOUS PLANCK STAMPS
            """
            if pstamp is None:
                print(f"{task} has no planck stamp")
                continue
            if not (np.all(np.isfinite(pstamp))):
                print(f"{task} has anomalous planck stamp; not finite")
                if args.debug_anomalies:
                    io.plot_img(
                        pstamp,
                        f"{paths.debugdir}planck_err_stamp_notfinite_{task}.png",
                        arc_width=args.swidth,
                        xlabel="$\\theta_x$ (arcmin)",
                        ylabel="$\\theta_y$ (arcmin)",
                    )
                continue

            # # Planck unit conversion: K -> uK 
            if not (args.ilc_maps): 
                pstamp = pstamp[0] * 1e6
            else: pstamp = pstamp * 1e6


        else:
            # using hres map for gradient leg 
            gastamp_150 = reproject.thumbnails(
                gamap_150,
                coords,
                r=maxr,
                res=pixel * utils.arcmin,
                proj="tan",
                oversample=2,
                pixwin=True
            )
            gastamp_90 = reproject.thumbnails(
                gamap_90,
                coords,
                r=maxr,
                res=pixel * utils.arcmin,
                proj="tan",
                oversample=2,
                pixwin=True
            )

    else:
        pstamp, astamp_150, astamp_90 = csim.get_obs(task)

    """ 
    !! COSINE TAPER
    """
    if j == 0:
        # get an edge taper map and apodize
        taper = maps.get_taper(
            astamp_150.shape,
            astamp_150.wcs,
            taper_percent=args.tap_per,
            pad_percent=args.pad_per,
            weight=None,
        )
        taper = taper[0]

    # applying this to the stamp makes it have a nice zeroed edge!
    act_stamp_150 = astamp_150 * taper
    act_stamp_90 = astamp_90 * taper

    if args.freq_null:
        act_stamp_fnull = (astamp_150 - astamp_90) * taper

    if not (args.hres_grad):
        plc_stamp = pstamp * taper
    else:
        gact_stamp_150 = gastamp_150 * taper
        gact_stamp_90 = gastamp_90 * taper
    
    if args.inpaint: # only with hres_grad for now 
        """
        If inpainting, we 
        (1) resample the stamp to 64x64 (2 arcmin pixels)
        (2) Inpaint a hole of radius 4 arcmin 
        """
        rmin = 4 * utils.arcmin
        crop_pixels = int(16.  / args.pwidth) # 16 arcminutes wide
        act150 = maps.crop_center(gact_stamp_150,cropy=crop_pixels,cropx=crop_pixels,sel=False)
        act90 = maps.crop_center(gact_stamp_90,cropy=crop_pixels,cropx=crop_pixels,sel=False)
        act_sel150 = maps.crop_center(gact_stamp_150,cropy=crop_pixels,cropx=crop_pixels,sel=True)
        act_sel90 = maps.crop_center(gact_stamp_90,cropy=crop_pixels,cropx=crop_pixels,sel=True)
        Ndown150, Ndown2 = act150.shape[-2:]
        #print(Ndown150, Ndown2) # 32 32
        if Ndown150 != Ndown2: raise Exception
        Ndown90, Ndown2 = act90.shape[-2:]
        if Ndown90 != Ndown2: raise Exception

        if j==0:
            from orphics import pixcov
            pshape = act150.shape
            pwcs = act150.wcs
            beam_fn150 = cutils.load_beam("f150")
            beam_fn90 = cutils.load_beam("f090")
            ipsizemap = enmap.pixsizemap(pshape, pwcs)
            pivar = maps.ivar(pshape, pwcs, defaults.highres_fiducial_rms, ipsizemap=ipsizemap)
            pcov150 = pixcov.tpcov_from_ivar(Ndown150, pivar, theory.lCl, beam_fn150)
            pcov90 = pixcov.tpcov_from_ivar(Ndown90, pivar, theory.lCl, beam_fn90)            
            geo150 = pixcov.make_geometry(pshape, pwcs, rmin, n=Ndown150, deproject=True, iau=False, res=None, pcov=pcov150)
            geo90 = pixcov.make_geometry(pshape, pwcs, rmin, n=Ndown90, deproject=True, iau=False, res=None, pcov=pcov90)

        act150 = pixcov.inpaint_stamp(act150, geo150)
        act90 = pixcov.inpaint_stamp(act90, geo90)
        gact_stamp_150[act_sel150] = act150.copy()
        gact_stamp_90[act_sel90] = act90.copy()

    if args.day_null:
        nact_stamp_150 = nastamp_150 * taper
        nact_stamp_90 = nastamp_90 * taper

    if args.debug_stack:
        sweight = ivar_90.mean()

	    # to obtain tsz profile and covmat for act stamp 
        shape = astamp_150.shape
        wcs = astamp_150.wcs
        modrmap = enmap.modrmap(shape, wcs)        
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=args.hres_lxcut, lycut=args.hres_lycut)
        masked_150 = maps.filter_map(astamp_150, ymask)
        masked_90 = maps.filter_map(astamp_90, ymask)

        if args.no_filter:
            s.add_to_stack('a150_cmb', astamp_150*sweight)
            s.add_to_stack('a90_cmb', astamp_90*sweight)
            sz150 = bin(astamp_150, modrmap * (180 * 60 / np.pi), bin_edges)
            sz150w = bin(astamp_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)
            sz90 = bin(astamp_90, modrmap * (180 * 60 / np.pi), bin_edges)
        else:
            s.add_to_stack('a150_cmb', masked_150*sweight)
            s.add_to_stack('a90_cmb', masked_90*sweight)
            sz150 = bin(masked_150, modrmap * (180 * 60 / np.pi), bin_edges)   
            sz150w = bin(masked_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)
            sz90 = bin(masked_90, modrmap * (180 * 60 / np.pi), bin_edges)

        if args.day_null:    
            nmasked_150 = maps.filter_map(nastamp_150, ymask)
            nmasked_90 = maps.filter_map(nastamp_90, ymask)
            if args.no_filter:
                s.add_to_stack('na150_cmb', nastamp_150*sweight)
                s.add_to_stack('na90_cmb', nastamp_90*sweight)  
                sz150 = bin(nastamp_150, modrmap * (180 * 60 / np.pi), bin_edges)
                sz150w = bin(nastamp_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)
                sz90 = bin(nastamp_90, modrmap * (180 * 60 / np.pi), bin_edges)
            else:
                s.add_to_stack('na150_cmb', nmasked_150*sweight)
                s.add_to_stack('na90_cmb', nmasked_90*sweight)  
                sz150 = bin(nmasked_150, modrmap * (180 * 60 / np.pi), bin_edges)   
                sz150w = bin(nmasked_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)  
                sz90 = bin(nmasked_90, modrmap * (180 * 60 / np.pi), bin_edges)
            
        s.add_to_stats("sz150", sz150)  
        s.add_to_stats("sz150w", sz150w)
        s.add_to_stats("sz90", sz90)         
        s.add_to_stats("szw", (sweight,))
        s.add_to_stats("szw2", (sweight ** 2,))
        s.add_to_stack('acmb_twt',(astamp_90*0+1)*sweight)

        if not (args.hres_grad):
        	# to obtain tsz profile and covmat for planck stamp
            modrmap = enmap.modrmap(pstamp.shape, pstamp.wcs)          
            xmask = maps.mask_kspace(pstamp.shape, pstamp.wcs, lmin=xlmin, lmax=xlmax)
            masked = maps.filter_map(pstamp, xmask)     
            if args.no_filter:
                s.add_to_stack('p_cmb', pstamp)
                psz = bin(pstamp, modrmap * (180 * 60 / np.pi), bin_edges)
            else:
                s.add_to_stack('p_cmb', masked)
                psz = bin(masked, modrmap * (180 * 60 / np.pi), bin_edges)   
            s.add_to_stats("psz_binned", psz)              

        else:
            # to obtain tsz profile and covmat for hres inpainted grad
            modrmap = enmap.modrmap(gact_stamp_150.shape, gact_stamp_150.wcs) 
            xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
            gmasked_150 = maps.filter_map(gact_stamp_150, xmask)
            gmasked_90 = maps.filter_map(gact_stamp_90, xmask)
            
            if args.no_filter:
                s.add_to_stack('ga150_cmb', gact_stamp_150*sweight)
                s.add_to_stack('ga90_cmb', gact_stamp_90*sweight) 
                sz150 = bin(gact_stamp_150, modrmap * (180 * 60 / np.pi), bin_edges)
                sz150w = bin(gact_stamp_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)
                sz90 = bin(gact_stamp_90, modrmap * (180 * 60 / np.pi), bin_edges) 
                sz90w = bin(gact_stamp_90*sweight, modrmap * (180 * 60 / np.pi), bin_edges)
            else:
                s.add_to_stack('ga150_cmb', gmasked_150*sweight)
                s.add_to_stack('ga90_cmb', gmasked_90*sweight)  
                sz150 = bin(gmasked_150, modrmap * (180 * 60 / np.pi), bin_edges)   
                sz150w = bin(gmasked_150*sweight, modrmap * (180 * 60 / np.pi), bin_edges)  
                sz90 = bin(gmasked_90, modrmap * (180 * 60 / np.pi), bin_edges)
                sz90w = bin(gmasked_90*sweight, modrmap * (180 * 60 / np.pi), bin_edges)

            s.add_to_stats("gsz150", sz150)  
            s.add_to_stats("gsz150w", sz150w)
            s.add_to_stats("gsz90", sz90) 
            s.add_to_stats("gsz90w", sz90w)        


        # j = j + 1                        
        # continue # commented for now 

    """ 
    !! STAMP FFTs
    """
    k150 = enmap.fft(act_stamp_150, normalize="phys")
    if not (args.no_90):
        k90 = enmap.fft(act_stamp_90, normalize="phys")

    if args.day_null:
        nk150 = enmap.fft(nact_stamp_150, normalize="phys")
        nk90 = enmap.fft(nact_stamp_90, normalize="phys")

    if args.freq_null:
        fnk = enmap.fft(act_stamp_fnull, normalize="phys")
 
    if not (args.hres_grad):
        kp = enmap.fft(plc_stamp, normalize="phys")
    else:
        gk150 = enmap.fft(gact_stamp_150, normalize="phys")
        gk90 = enmap.fft(gact_stamp_90, normalize="phys")      
       


    if j == 0:

        """ 
        !! INITIALIZE CALCULATIONS BASED ON GEOMETRY
        """
        shape = astamp_150.shape
        wcs = astamp_150.wcs
        modlmap = enmap.modlmap(shape, wcs)

        # High-res beam functions
        if not (args.ilc_maps): 
            bfunc150 = cutils.load_beam("f150")
            bfunc90 = cutils.load_beam("f090")
        else:
            bfunc150 = lambda x: maps.gauss_beam(ilc_beam_fwhm, x)
            bfunc90 = lambda x: maps.gauss_beam(ilc_beam_fwhm, x)

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid
        if not (args.ilc_maps): 
            act_150_kbeam2d = bfunc150(modlmap)
            act_90_kbeam2d = bfunc90(modlmap)
            plc_kbeam2d = maps.gauss_beam(modlmap, plc_beam_fwhm)
        else:
            act_150_kbeam2d = maps.gauss_beam(modlmap, ilc_beam_fwhm)
            act_90_kbeam2d = maps.gauss_beam(modlmap, ilc_beam_fwhm)
            plc_kbeam2d = maps.gauss_beam(modlmap, ilc_beam_fwhm)

        # get theory spectrum - this should be the lensed spectrum!
        ells = np.arange(8000)
        cltt = theory.lCl("TT", ells)

        ## interpolate ells and cltt 1D power spectrum specification
        ## isotropically on to the Fourier 2D space grid
        # build interpolated 2D Fourier CMB from theory and maps
        ucltt = maps.interp(ells, cltt)(modlmap)

        # bin size and range for 1D binned power spectrum
        minell = 2 * maps.minimum_ell(shape, wcs)
        l_edges = np.arange(minell / 2, 8001, minell)
        lbinner = stats.bin2D(modlmap, l_edges)
	    # PS correction factor
        w2 = np.mean(taper ** 2) 
        # fsky for bandpower variance
        fsky = enmap.area(shape, wcs) * w2 / 4.0 / np.pi

        # build Fourier space masks for lensing reconstruction
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(
            shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=args.hres_lxcut, lycut=args.hres_lycut
        )
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        # map of distances from center
        modrmap = enmap.modrmap(shape, wcs)

    # Fourier map -> PS
    pow = lambda x, y: (x * y.conj()).real 

    # measure the binned power spectrum from given stamp
    act_cents, act_p1d_150 = lbinner.bin(pow(k150, k150) / w2)
    if not (args.no_90):
        act_cents, act_p1d_90 = lbinner.bin(pow(k90, k90) / w2)
        act_cents, act_p1d_150_90 = lbinner.bin(pow(k150, k90) / w2)

    if args.freq_null:
        act_cents, act_p1d = lbinner.bin(pow(fnk, fnk) / w2)

    if not (args.hres_grad):
        plc_cents, plc_p1d = lbinner.bin(pow(kp, kp) / w2)
    else:
        act_cents, gact_p1d_150 = lbinner.bin(pow(gk150, gk150) / w2)
        act_cents, gact_p1d_90 = lbinner.bin(pow(gk90, gk90) / w2)
        act_cents, gact_p1d_150_90 = lbinner.bin(pow(gk150, gk90) / w2)

    """ 
    !! FIT POWER SPECTRA
    """
    tclaa_150 = fit_p1d(
        l_edges,
        act_cents,
        act_p1d_150,
        "act",
        modlmap,
        bfunc150,
        bfunc150,
        rms=defaults.highres_fiducial_rms,
        lmin=defaults.highres_fit_ellmin,
        lmax=defaults.highres_fit_ellmax,
    )
    if not (args.no_90):
        tclaa_90 = fit_p1d(
            l_edges,
            act_cents,
            act_p1d_90,
            "act",
            modlmap,
            bfunc90,
            bfunc90,
            rms=defaults.highres_fiducial_rms,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
        )
        tclaa_150_90 = fit_p1d(
            l_edges,
            act_cents,
            act_p1d_150_90,
            "act_cross",
            modlmap,
            bfunc150,
            bfunc90,
            rms=0,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
        )

    if args.freq_null:
       tclaa_fn = fit_p1d(
            l_edges,
            act_cents,
            act_p1d,
            "act",
            modlmap,
            bfunc150,
            bfunc150,
            rms=defaults.highres_fiducial_rms,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
        )            

    if not (args.hres_grad): 
       tclpp = fit_p1d(
            l_edges,
            plc_cents,
            plc_p1d,
            "plc",
            modlmap,
            lambda x: maps.gauss_beam(x, ilc_beam_fwhm if args.ilc_maps else plc_beam_fwhm),
            lambda x: maps.gauss_beam(x, ilc_beam_fwhm if args.ilc_maps else plc_beam_fwhm),
            rms=defaults.gradient_fiducial_rms,
            lmin=defaults.gradient_fit_ellmin,
            lmax=defaults.gradient_fit_ellmax,
        )
    else:
       tclgg_150 = fit_p1d(
            l_edges,
            act_cents,
            gact_p1d_150,
            "act",
            modlmap,
            bfunc150,
            bfunc150,
            rms=defaults.highres_fiducial_rms,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
       )
       tclgg_90 = fit_p1d(
            l_edges,
            act_cents,
            gact_p1d_90,
            "act",
            modlmap,
            bfunc90,
            bfunc90,
            rms=defaults.highres_fiducial_rms,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
       )
       tclgg_150_90 = fit_p1d(
            l_edges,
            act_cents,
            gact_p1d_150_90,
            "act_cross",
            modlmap,
            bfunc150,
            bfunc90,
            rms=0,
            lmin=defaults.highres_fit_ellmin,
            lmax=defaults.highres_fit_ellmax,
       )


    """ 
    !! ILC / COADD
    """

    if not (args.no_90):
        act_kmap, tclaa = ilc(
            modlmap,
            k150,
            k90,
            tclaa_150,
            tclaa_90,
            tclaa_150_90,
            act_150_kbeam2d,
            act_90_kbeam2d,
        ) # beam deconvolved
        
        if args.day_null:
            nact_kmap, _ = ilc(
                modlmap,
                nk150,
                nk90,
                tclaa_150,
                tclaa_90,
                tclaa_150_90,
                act_150_kbeam2d,
                act_90_kbeam2d,
            ) # beam deconvolved
 
        if args.hres_grad:
            gact_kmap, tclgg = ilc(
                modlmap,
                gk150,
                gk90,
                tclgg_150,
                tclgg_90,
                tclgg_150_90,
                act_150_kbeam2d,
                act_90_kbeam2d,
            ) # beam deconvolved
           
        if args.freq_null:
            act_kmap150 = k150 / act_150_kbeam2d 
            act_kmap90 = k90 / act_90_kbeam2d 
            act_kmap = act_kmap150 - act_kmap90
            tclaa = tclaa_fn / (act_150_kbeam2d ** 2.0)
            
        if args.no_150:
            act_kmap = k90 / act_90_kbeam2d 
            tclaa = tclaa_90 / (act_90_kbeam2d ** 2.0)          

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Deconvolve beam
            act_kmap = k150 / act_150_kbeam2d 
            tclaa = tclaa_150 / (act_150_kbeam2d ** 2.0)
        
    act_kmap[~np.isfinite(act_kmap)] = 0
    tclaa[~np.isfinite(tclaa)] = 0

    if args.day_null:
        nact_kmap[~np.isfinite(nact_kmap)] = 0

    if not (args.hres_grad): 
        plc_kmap = kp / plc_kbeam2d
        tclpp = tclpp / (plc_kbeam2d ** 2.0)
        plc_kmap[~np.isfinite(plc_kmap)] = 0
        tclpp[~np.isfinite(tclpp)] = 0

        if args.save_power: 
            cents, btclpp = lbinner.bin(tclpp)
            cents, btclaa = lbinner.bin(tclaa)
            io.save_cols(f'{paths.savedir}/binned_tclpp_{task}.txt',(cents,btclpp))
            io.save_cols(f'{paths.savedir}/binned_tclaa_{task}.txt',(cents,btclaa))

    else:
        gact_kmap[~np.isfinite(gact_kmap)] = 0
        tclgg[~np.isfinite(tclgg)] = 0

    # Fit cross-power of gradient and high-res; not usually used
    if not (args.hres_grad):
        cents, c_ap = lbinner.bin(pow(act_kmap, plc_kmap) / w2)
        tclap = fit_p1d(
            l_edges, cents, c_ap, "apcross", modlmap, None, None, rms=0, lmin=defaults.highres_fit_ellmin, lmax=defaults.gradient_fit_ellmax
        )
    else:
        cents, c_ag = lbinner.bin(pow(act_kmap, gact_kmap) / w2)
        tclag = fit_p1d(
            l_edges, cents, c_ag, "apcross", modlmap, None, None, rms=0, lmin=defaults.highres_fit_ellmin, lmax=defaults.highres_fit_ellmax
        )


    """ 
    !! LENS RECONSTRUCTION
    """
    if args.rand_rot:
        np.random.seed(task)
        act_rmap = enmap.ifft(act_kmap*ymask,normalize='phys').real
        rotmap = enmap.enmap(np.rot90(act_rmap,np.random.randint(1,4)),wcs)
        act_kmap = enmap.fft(rotmap,normalize='phys') * ymask

    # build symlens dictionary for lensing reconstruction
    feed_dict = {
        "uC_T_T": ucltt,  # goes in the lensing response func = lensed theory
        "tC_A_T_A_T": tclaa,  # the approximate ACT power spectrum, ACT beam deconvolved
        "tC_P_T_P_T": tclgg if args.hres_grad else tclpp,  # approximate Planck power spectrum, Planck beam deconvolved
        "tC_A_T_P_T": tclag if args.hres_grad else tclap,  # same lensed theory as above, no instrumental noise
        "tC_P_T_A_T": tclag if args.hres_grad else tclap,  # same lensed theory as above, no instrumental noise
        "X": gact_kmap if args.hres_grad else plc_kmap,  # gradient leg : 2D Planck map, Planck beam deconvolved
        "Y": nact_kmap if args.day_null else act_kmap,  # hres leg : 2D ILC ACT map, ACT beam deconvolved 
    }

    # Sanity check
    for key in feed_dict.keys():
        assert np.all(np.isfinite(feed_dict[key]))

    # ask for reconstruction in Fourier space
    cqe = symlens.QE(
        shape,
        wcs,
        feed_dict,
        estimator="hdv_curl" if args.curl else "hdv",
        XY="TT",
        xmask=xmask,
        ymask=ymask,
        field_names=["P", "A"],
        groups=None,
        kmask=kmask,
    )
    # Fourier space lens reconstruction
    krecon = cqe.reconstruct(feed_dict, xname="X_l1", yname="Y_l2", physical_units=True)
    # not the pixel unit 

    # transform to real space for unweighted stack
    kappa = enmap.ifft(krecon, normalize="phys").real

    """ 
    !! REJECT WEIRD KAPPA
    """

    ##### temporary 3: to get rid of stamps with tSZ cluster in random locations
    if np.any(np.abs(kappa) > 15):
        print(f"{task} has large kappa")
        if args.debug_anomalies:
            io.plot_img(
                kappa,
                f"{paths.debugdir}kappa_err_stamp_large_kappa_{task}.png",
                arc_width=args.swidth,
                xlabel="$\\theta_x$ (arcmin)",
                ylabel="$\\theta_y$ (arcmin)",
            )
        if args.debug_anomalies:
            io.plot_img(
                astamp_150,
                f"{paths.debugdir}act_150_err_stamp_large_kappa_{task}.png",
                arc_width=args.swidth,
                xlabel="$\\theta_x$ (arcmin)",
                ylabel="$\\theta_y$ (arcmin)",
            )
        continue    

    # Save power spectra to look at later
    if args.debug_powers:
        s.add_to_stats("n150", lbinner.bin(tclaa_150)[1])
        if not (args.no_90):
            s.add_to_stats("n90", lbinner.bin(tclaa_90)[1])
            s.add_to_stats("nc", lbinner.bin(tclaa_150_90)[1])
        s.add_to_stats("np", lbinner.bin(tclpp)[1])

    # Unweighted stack
    s.add_to_stack("ustack", kappa)

    """ 
    !! OPTIMAL WEIGHTING
    """
    # Noise curve for lensing obtained from normalization
    Al = cqe.Al

    if args.full_nl:
        # FIXME: The full calculation seems to sometimes be negative
        Nl = symlens.N_l(shape,wcs,feed_dict,estimator="hdv_curl" if args.curl else "hdv",XY='TT',
                         xmask=xmask,ymask=ymask,
                         Al=Al,field_names=['P','A'],kmask=kmask,power_name="t") # !!!
    else:
        # Use approximate noise assuming estimator is optimal (Narrator: it isn't)
        Nl = symlens.N_l_from_A_l_optimal(shape, wcs, Al)
    cents, bnl = lbinner.bin(Nl)
    nmean = bnl[np.logical_and(cents > defaults.kappa_noise_mean_Lmin, cents < defaults.kappa_noise_mean_Lmax)].mean()
    if args.debug_powers:
        s.add_to_stats("nl", bnl)

    if args.save_power:      
        # save theory lensing power spectrum     
        clkk = theory.gCl('kk', ells)
        io.save_cols(f'{paths.savedir}/binned_clkk_{task}.txt',(ells,clkk))
        
        # save noise power spectrum for each stamp : comment out "continue" 
        enmap.write_map(f"{paths.savedir}/noise_{task}.fits", Nl)
        enmap.write_map_geometry(f"{paths.savedir}/map_geometry_{task}.fits", shape, wcs)
        io.save_cols(f'{paths.savedir}/binned_noise_{task}.txt',(cents,bnl))
        
        j = j + 1    
        continue

    if args.debug_nl:
        if np.any(lbinner.bin(Nl)[1]<0):
            ls = lbinner.bin(Nl)[0]
            print(task,ls[lbinner.bin(Nl)[1]<0])
            print("Negative Nls")
        print(Nl.shape)
        pl = io.Plotter('CL')
        pl.add(ells,theory.gCl('kk',ells),color='k')
        pl.add(*lbinner.bin(Nl),ls='--')
        pl.done(f'{paths.debugdir}nlkk.png')
        io.plot_img(np.fft.fftshift(np.log10(Nl)),f'{paths.debugdir}nl2d.png',arc_width=args.swidth)
        print(task)
        io.save_cols(f"{'nl' if args.full_nl else 'al'}{'_curl' if args.curl else ''}.txt",lbinner.bin(Nl))
        sys.exit()

    # Save weighted stacks and statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iNl = (1.0 / Nl) * kmask
    iNl[~np.isfinite(iNl)] = 0
    wkrecon = krecon * iNl
    s.add_to_stack("wk_real", wkrecon.real * cweight)
    s.add_to_stack("wk_imag", wkrecon.imag * cweight)
    s.add_to_stack("wk_iwt", iNl * cweight)

    # inverse variance noise weighting
    ivmean = 1.0 / nmean

    weight = ivmean * cweight
    stack = kappa * weight
    s.add_to_stack("kmap", stack)

    wbinned = bin(stack, modrmap * (180 * 60 / np.pi), bin_edges)
    binned = bin(kappa, modrmap * (180 * 60 / np.pi), bin_edges)
    s.add_to_stats("wk1d", wbinned)
    s.add_to_stats("k1d", binned)
    s.add_to_stats("kw", (weight,))
    s.add_to_stats("kw2", (weight ** 2,))

    # check how many stamps are cut out of given map
    s.add_to_stats("ct_selected", (1,))
    if not(args.is_meanfield) and not(args.inject_sim):
        s.add_to_stats("data", (z,weight,*[cdata[key][i] for key in sorted(cdata.keys())]))                
        s.add_to_stats("redshift", (z,))
        #s.add_to_stats("mass", (cdata['mass'][i],))
        #s.add_to_stats("y0s", (cdata['y0s'][i],))
        #s.add_to_stats("wmass", (cdata['mass'][i] * weight,))  
      
    j = j + 1


# collect from all MPI cores and calculate stacks
s.get_stacks()
s.get_stats()

if rank == 0:

    if args.debug_stack:
       
        twt = s.stacks['acmb_twt']
        a150 = s.stacks['a150_cmb'] / twt
        a90 = s.stacks['a90_cmb'] / twt
        
        if args.day_null:
            a150 = s.stacks['na150_cmb'] / twt
            a90 = s.stacks['na90_cmb'] / twt
              
        # obtain tsz profile and covmat for tsz stack     
        N_sz = s.vectors['szw'].shape[0]
        vsz1 = s.vectors['szw'].sum()
        vsz2 = s.vectors['szw2'].sum()
         
        opt_binned = s.vectors['sz150w'].sum(axis=0) / vsz1
        diff = s.vectors['sz150'] - opt_binned
        cov = np.dot((diff * s.vectors['szw']).T,diff) / (vsz1-(vsz2/vsz1))
        opt_covm = cov/N_sz
        opt_corr = stats.cov2corr(opt_covm)
        opt_errs = np.sqrt(np.diag(opt_covm))

        binned = s.stats['sz150']['mean']
        covm = s.stats['sz150']['covmean']
        corr = stats.cov2corr(s.stats['sz150']['covmean'])
        errs = s.stats['sz150']['errmean']
        
        binned90 = s.stats['sz90']['mean']
        errs90 = s.stats['sz90']['errmean']       

        np.savetxt(f"{paths.savedir}/bin_edges.txt", bin_edges)    
        np.savetxt(f"{paths.savedir}/opt_profile.txt", opt_binned)
        np.savetxt(f"{paths.savedir}/profile.txt", binned)
        np.savetxt(f"{paths.savedir}/profile90.txt", binned90)        
        np.savetxt(f"{paths.savedir}/opt_profile_errs.txt", opt_errs)
        np.savetxt(f"{paths.savedir}/profile_errs.txt", errs)    
        np.savetxt(f"{paths.savedir}/profile_errs90.txt", errs90)         
        np.savetxt(f"{paths.savedir}/opt_covm.txt", opt_covm)
        np.savetxt(f"{paths.savedir}/covm.txt", covm)
        save(f"{paths.savedir}/opt_corr.npy", opt_corr) 
        save(f"{paths.savedir}/corr.npy", corr)    
        save(f"{paths.savedir}/a150_cmb.npy", a150)  
        save(f"{paths.savedir}/a90_cmb.npy", a90)      
        enmap.write_map(f"{paths.savedir}/a150_cmb.fits",a150)
        enmap.write_map(f"{paths.savedir}/a90_cmb.fits",a90)
  
        cwidth = 30.
        crop = int(cwidth / args.pwidth)
        cutils.plot(f"{paths.savedir}/a150_cmb.png",a150,0,0,crop=None,lim=None,label='$\\mu$K')
        cutils.plot(f"{paths.savedir}/a150_cmb_zoom.png",a150,0,0,crop=crop,lim=None,label='$\\mu$K')
        cutils.plot(f"{paths.savedir}/a90_cmb.png",a90,0,0,crop=None,lim=None,label='$\\mu$K')
        cutils.plot(f"{paths.savedir}/a90_cmb_zoom.png",a90,0,0,crop=crop,lim=None,label='$\\mu$K') 
 
        if not (args.hres_grad):                        
            planck = s.stacks['p_cmb']
            binned = s.stats['psz_binned']['mean']
            covm = s.stats['psz_binned']['covmean']
            corr = stats.cov2corr(s.stats['psz_binned']['covmean'])
            errs = s.stats['psz_binned']['errmean']

            np.savetxt(f"{paths.savedir}/plc_profile.txt", binned)
            np.savetxt(f"{paths.savedir}/plc_profile_errs.txt", errs)
            np.savetxt(f"{paths.savedir}/plc_covm.txt", covm)
            save(f"{paths.savedir}/plc_corr.npy", corr)   
            save(f"{paths.savedir}/p_cmb.npy", planck) 
            enmap.write_map(f"{paths.savedir}/p_cmb.fits",planck)
            cutils.plot(f"{paths.savedir}/p_cmb.png",planck,0,0,crop=None,lim=None,label='$\\mu$K')
            cutils.plot(f"{paths.savedir}/p_cmb_zoom.png",planck,0,0,crop=crop,lim=None,label='$\\mu$K')

        else:
            a150 = s.stacks['ga150_cmb'] / twt
            a90 = s.stacks['ga90_cmb'] / twt
             
            opt_binned = s.vectors['gsz150w'].sum(axis=0) / vsz1
            diff = s.vectors['gsz150'] - opt_binned
            cov = np.dot((diff * s.vectors['szw']).T,diff) / (vsz1-(vsz2/vsz1))
            opt_covm = cov/N_sz
            opt_corr = stats.cov2corr(opt_covm)
            opt_errs = np.sqrt(np.diag(opt_covm))

            binned = s.stats['gsz150']['mean']
            covm = s.stats['gsz150']['covmean']
            corr = stats.cov2corr(s.stats['gsz150']['covmean'])
            errs = s.stats['gsz150']['errmean']
            binned90 = s.stats['gsz90']['mean']
            errs90 = s.stats['gsz90']['errmean'] 

            np.savetxt(f"{paths.savedir}/inp_bin_edges.txt", bin_edges)    
            np.savetxt(f"{paths.savedir}/inp_opt_profile.txt", opt_binned)
            np.savetxt(f"{paths.savedir}/inp_profile.txt", binned)
            np.savetxt(f"{paths.savedir}/inp_opt_profile_errs.txt", opt_errs)
            np.savetxt(f"{paths.savedir}/inp_profile_errs.txt", errs)    
            np.savetxt(f"{paths.savedir}/inp_opt_covm.txt", opt_covm)
            np.savetxt(f"{paths.savedir}/inp_covm.txt", covm)
            save(f"{paths.savedir}/inp_opt_corr.npy", opt_corr) 
            save(f"{paths.savedir}/inp_corr.npy", corr)    
            save(f"{paths.savedir}/inp_a150_cmb.npy", a150)  
            save(f"{paths.savedir}/inp_a90_cmb.npy", a90)         
            cutils.plot(f"{paths.savedir}/inp_a150_cmb.png",a150,0,0,crop=None,lim=None,label='$\\mu$K')
            cutils.plot(f"{paths.savedir}/inp_a150_cmb_zoom.png",a150,0,0,crop=crop,lim=None,label='$\\mu$K')
            cutils.plot(f"{paths.savedir}/inp_a90_cmb.png",a90,0,0,crop=None,lim=None,label='$\\mu$K')
            cutils.plot(f"{paths.savedir}/inp_a90_cmb_zoom.png",a90,0,0,crop=crop,lim=None,label='$\\mu$K')  

    # Dump all collected statistics
    with bench.show("dump"):
        if not(args.inject_sim):
            with open(f"{paths.savedir}/cat_data_columns.txt",'w') as f:
                f.write(' '.join(['z','weight',*[key for key in sorted(cdata.keys())]]))
        s.dump(paths.savedir)
        enmap.write_map_geometry(f"{paths.savedir}/map_geometry.fits", shape, wcs)
        enmap.write_map(f"{paths.savedir}/kmask.fits", kmask)
        enmap.write_map(f"{paths.savedir}/modrmap.fits", modrmap)
        np.savetxt(f"{paths.savedir}/bin_edges.txt", bin_edges)
        if not(args.is_meanfield) and not (args.debug_stack):
            np.savetxt(f"{paths.savedir}/profiles.txt",s.vectors['k1d'])
            #np.savetxt(f"{paths.savedir}/z_mass_y.txt", np.c_[s.vectors['redshift'], s.vectors['mass'], s.vectors['y0s']])

    for ctkey in [
        "selected"
    ]:  # ,'large_kappa','no_stamp','high_noise','high_stamp','anomalous']:
        try:
            N = s.vectors[f"ct_{ctkey}"].sum()
            print(f"Number {ctkey} : {N}")
        except:
            pass

    if not (args.debug_stack):
	    # Sanity checks
        N_stamp = s.vectors[f"ct_{ctkey}"].sum()
        assert N_stamp == s.stack_count["kmap"]
        assert N_stamp == s.vectors["kw"].shape[0]

        # if not (args.is_meanfield):
        #     kappa_w = s.vectors['kw'].sum()
        #     w_mass = s.vectors['wmass'].sum(axis=0)/kappa_w
        #     print("weighted SZ mean mass : ", w_mass, "1e14 Msun")   

    # Some debug plots if requested
    if args.debug_powers:
        pl = io.Plotter("CL")
        pl.add(ells, theory.gCl("kk", ells), color="k")
        for Nl in s.vectors["nl"]:
            pl.add(lbinner.centers, Nl, ls="--", alpha=0.2)
        pl.done(f"{paths.savedir}nlkk.png")

    if args.debug_powers:
        pl = io.Plotter("Cell")
        for i in range(s.vectors["n150"].shape[0]):
            pl.add(act_cents, s.vectors["n150"][i])
        pl.done(f"{paths.savedir}n150.png")

        if not (args.no_90):
            pl = io.Plotter("Cell")
            for i in range(s.vectors["n90"].shape[0]):
                pl.add(act_cents, s.vectors["n90"][i])
            pl.done(f"{paths.savedir}n90.png")

            pl = io.Plotter("Cell")
            for i in range(s.vectors["nc"].shape[0]):
                pl.add(act_cents, s.vectors["nc"][i])
            pl.done(f"{paths.savedir}nc.png")

        pl = io.Plotter("Cell")
        for i in range(s.vectors["np"].shape[0]):
            pl.add(plc_cents, s.vectors["np"][i])
        pl.done(f"{paths.savedir}np.png")

    elapsed = time.time() - start_time
    print("\r ::: entire run took %.1f seconds" % elapsed)

