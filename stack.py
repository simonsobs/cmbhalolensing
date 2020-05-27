import numpy as np
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
start_time,paths,defaults,args,tags,rank = cutils.initialize_pipeline_config()
if rank==0:
    print("Paths: ",paths)
    print("Tags: ",tags)
    print("Defaults: ",defaults)
    print("Arguments: ",args)

# Load a fiducial CMB theory object
theory = cosmology.default_theory()

if not (args.inject_sim):
    # Load the catalog
    ras, decs = cutils.catalog_interface(args.cat_type, args.is_meanfield, args.nmax)
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
        fplc_map = paths.data + "planck_smica_nosz_reproj.fits"
        try:
            pmap = enmap.read_map(fplc_map, delayed=False)
        except:
            plc_map = paths.planck_data + "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
            # reproject the Planck map (healpix -> CAR)
            fshape, fwcs = enmap.fullsky_geometry(res=2.0 * utils.arcmin, proj="car")
            pmap = reproject.enmap_from_healpix(
                plc_map, fshape, fwcs, ncomp=1, unit=1, lmax=6000, rot="gal,equ"
            )
            enmap.write_map(fplc_map, pmap)

        # ACT 150 GHz coadd map
        if args.no_sz_sub:
            act_map = (
                paths.coadd_data + f"{tags.apstr}_s08_s18_cmb_f150_{tags.dstr}_srcfree_map.fits"
            )
            amap_150 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])
        else:
            act_map = paths.data + f"modelSubtracted150_{tags.apstr}_{tags.dstr}.fits"
            amap_150 = enmap.read_map(act_map, delayed=False)

        # ACT 90 GHz coadd map
        if args.no_sz_sub:
            act_map = (
                paths.coadd_data + f"{tags.apstr}_s08_s18_cmb_f090_{tags.dstr}_srcfree_map.fits"
            )
            amap_90 = enmap.read_map(act_map, delayed=False, sel=np.s_[0, ...])
        else:
            act_map = paths.data + f"modelSubtracted90_{tags.apstr}_{tags.dstr}.fits"
            amap_90 = enmap.read_map(act_map, delayed=False)

        # Inv var map for 90 GHz
        ivar_map = paths.coadd_data + f"{tags.apstr}_s08_s18_cmb_f090_{tags.dstr}_ivar.fits"
        imap_90 = enmap.read_map(ivar_map, delayed=False, sel=np.s_[0, ...])
        rms_map = maps.rms_from_ivar(
            imap_90, cylindrical=True
        )  # convert to RMS noise map

# stamp size and resolution
stamp_width_deg = args.swidth / 60.0
pixel = args.pwidth
maxr = stamp_width_deg * utils.degree / 2.0

""" 
!! CATALOG TRIMMING BASED ON RMS MAP
"""

if not (args.inject_sim):
    # Remove objects that lie in unobserved regions
    with bench.show("cull"):
        coords = np.stack([decs, ras]) * utils.degree
        # Convert catalog coords to pixel coords
        ipixs = rms_map.sky2pix(coords).astype(np.int)
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
        pixs = np.stack(pixs)
        # Then select pixels where the noise is finite and less than args.max_rms_noise
        nsel = np.logical_and(
            rms_map[pixs[0, :], pixs[1, :]] > 0,
            rms_map[pixs[0, :], pixs[1, :]] < args.max_rms,
        )
        ras = ras[np.argwhere(nsel)][:, 0]
        decs = decs[np.argwhere(nsel)][:, 0]
        nsims = len(ras)
    del pixs, ipixs


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
plc_beam_fwhm = defaults.planck_smica_beam_fwhm

# Planck mask
xlmin = args.grad_lmin ; xlmax = args.grad_lmax

# ACT mask
ilcmin = defaults.ilc_lmin ; ilcmax = defaults.ilc_lmax
ylmin = args.hres_lmin ; ylmax = args.hres_lmax

# kappa mask
klmin = args.klmin ; klmax = args.klmax

# for binned kappa profile
bin_edges = np.arange(0, args.arcmax, args.arcstep)
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
            proj="plain",
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
                    arc_width=args.stamp_width_arcmin,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue

        """ 
        !! CUT OUT 150 and 90 GHZ STAMPS
        """
        # cut out a stamp from the ACT map (CAR -> plain)
        astamp_150 = reproject.thumbnails(
            amap_150,
            coords,
            r=maxr,
            res=pixel * utils.arcmin,
            proj="plain",
            oversample=2,
        )
        astamp_90 = reproject.thumbnails(
            amap_90,
            coords,
            r=maxr,
            res=pixel * utils.arcmin,
            proj="plain",
            oversample=2,
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
                    arc_width=args.stamp_width_arcmin,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            if args.debug_anomalies:
                io.plot_img(
                    astamp_90,
                    f"{paths.debugdir}act_90_err_stamp_large_stamp_{task}.png",
                    arc_width=args.stamp_width_arcmin,
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
                    arc_width=args.stamp_width_arcmin,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            if args.debug_anomalies:
                io.plot_img(
                    astamp_90,
                    f"{paths.debugdir}act_90_err_stamp_an_stamp_{task}.png",
                    arc_width=args.stamp_width_arcmin,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue

        """ 
        !! CUT OUT PLANCK STAMP
        """
        # cut out a stamp from the Planck map (CAR -> plain)
        pstamp = reproject.thumbnails(
            pmap, coords, r=maxr, res=pixel * utils.arcmin, proj="plain", oversample=2
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
                    arc_width=args.stamp_width_arcmin,
                    xlabel="$\\theta_x$ (arcmin)",
                    ylabel="$\\theta_y$ (arcmin)",
                )
            continue

        # Planck unit conversion: K -> uK
        pstamp = pstamp[0] * 1e6

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
    plc_stamp = pstamp * taper

    """ 
    !! STAMP FFTs
    """
    k150 = enmap.fft(act_stamp_150, normalize="phys")
    if not (args.no_90):
        k90 = enmap.fft(act_stamp_90, normalize="phys")
    kp = enmap.fft(plc_stamp, normalize="phys")

    if j == 0:

        """ 
        !! INITIALIZE CALCULATIONS BASED ON GEOMETRY
        """
        shape = astamp_150.shape
        wcs = astamp_150.wcs
        modlmap = enmap.modlmap(shape, wcs)

        # High-res beam functions
        bfunc150 = cutils.load_beam("f150")
        bfunc90 = cutils.load_beam("f090")

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid
        act_150_kbeam2d = bfunc150(modlmap)
        act_90_kbeam2d = bfunc90(modlmap)
        plc_kbeam2d = maps.gauss_beam(modlmap, plc_beam_fwhm)

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
        w2 = np.mean(taper ** 2) # PS correction factor
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

    pow = lambda x, y: (x * y.conj()).real # Fourier map -> PS

    # measure the binned power spectrum from given stamp
    act_cents, act_p1d_150 = lbinner.bin(pow(k150, k150) / w2)
    if not (args.no_90):
        act_cents, act_p1d_90 = lbinner.bin(pow(k90, k90) / w2)
        act_cents, act_p1d_150_90 = lbinner.bin(pow(k150, k90) / w2)
    plc_cents, plc_p1d = lbinner.bin(pow(kp, kp) / w2)

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
    tclpp = fit_p1d(
        l_edges,
        plc_cents,
        plc_p1d,
        "plc",
        modlmap,
        lambda x: maps.gauss_beam(x, plc_beam_fwhm),
        lambda x: maps.gauss_beam(x, plc_beam_fwhm),
        rms=defaults.gradient_fiducial_rms,
        lmin=defaults.gradient_fit_ellmin,
        lmax=defaults.gradient_fit_ellmax,
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
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Deconvolve beam
            act_kmap = k150 / act_150_kbeam2d 
            tclaa = tclaa_150 / act_150_kbeam2d ** 2.0
        act_kmap[~np.isfinite(act_kmap)] = 0
        tclaa[~np.isfinite(tclaa)] = 0

    ## total TT spectrum includes beam-deconvolved noise
    ## so create a total beam-deconvolved spectrum using a Gaussian beam func.
    tclpp = tclpp / (plc_kbeam2d ** 2.0)
    tclaa[~np.isfinite(tclaa)] = 0
    tclpp[~np.isfinite(tclpp)] = 0

    # get beam deconvolved Fourier map for Planck
    plc_kmap = kp / plc_kbeam2d
    act_kmap[~np.isfinite(act_kmap)] = 0
    plc_kmap[~np.isfinite(plc_kmap)] = 0

    # Fit cross-power of gradient and high-res; not usually used
    cents, c_ap = lbinner.bin(pow(act_kmap, plc_kmap) / w2)
    tclap = fit_p1d(
        l_edges, cents, c_ap, "apcross", modlmap, None, None, rms=0, lmin=defaults.highres_fit_ellmin, lmax=defaults.gradient_fit_ellmax
    )

    """ 
    !! LENS RECONSTRUCTION
    """
    # build symlens dictionary for lensing reconstruction
    feed_dict = {
        "uC_T_T": ucltt,  # goes in the lensing response func = lensed theory
        "tC_A_T_A_T": tclaa,  # the fit ACT power spectrum with ACT beam deconvolved
        "tC_P_T_P_T": tclpp,  # approximate Planck power spectrum with Planck beam deconvolved
        "tC_A_T_P_T": tclap,  # same lensed theory as above, no instrumental noise
        "tC_P_T_A_T": tclap,  # same lensed theory as above, no instrumental noise
        "X": plc_kmap,  # Planck map
        "Y": act_kmap,  # ACT map
    }

    # Sanity check
    for key in feed_dict.keys():
        assert np.all(np.isfinite(feed_dict[key]))

    # ask for reconstruction in Fourier space
    cqe = symlens.QE(
        shape,
        wcs,
        feed_dict,
        estimator="hdv",
        XY="TT",
        xmask=xmask,
        ymask=ymask,
        field_names=["P", "A"],
        groups=None,
        kmask=kmask,
    )
    # Fourier space lens reconstruction
    krecon = cqe.reconstruct(feed_dict, xname="X_l1", yname="Y_l2", physical_units=True)

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
                arc_width=args.stamp_width_arcmin,
                xlabel="$\\theta_x$ (arcmin)",
                ylabel="$\\theta_y$ (arcmin)",
            )
        if args.debug_anomalies:
            io.plot_img(
                astamp_150,
                f"{paths.debugdir}act_150_err_stamp_large_kappa_{task}.png",
                arc_width=args.stamp_width_arcmin,
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

    # Unweighte stack
    s.add_to_stack("ustack", kappa)

    """ 
    !! OPTIMAL WEIGHTING
    """
    # Noise curve for lensing obtained from normalization
    Al = cqe.Al

    # FIXME: The full calculation seems to sometimes be negative
    # Nl = symlens.N_l(shape,wcs,feed_dict,estimator='hdv',XY='TT',
    #     xmask=xmask,ymask=ymask,
    #     Al=Al,field_names=['P','A'],kmask=kmask,power_name="t")

    # Use approximate noise assuming estimator is optimal (Narrator: it isn't)
    Nl = symlens.N_l_from_A_l_optimal(shape, wcs, Al)
    cents, bnl = lbinner.bin(Nl)
    nmean = bnl[np.logical_and(cents > defaults.kappa_noise_mean_Lmin, cents < defaults.kappa_noise_mean_Lmax)].mean()
    if args.debug_powers:
        s.add_to_stats("nl", bnl)

    # if np.any(lbinner.bin(Nl)[1]<0):
    #     ls = lbinner.bin(Nl)[0]
    #     print(task,ls[lbinner.bin(Nl)[1]<0])
    #     raise ValueError

    # print(Nl.shape)
    # pl = io.Plotter('CL')
    # pl.add(ells,theory.gCl('kk',ells),color='k')
    # pl.add(*lbinner.bin(Nl),ls='--')
    # pl.done(f'{paths.debugdir}nlkk.png')
    # io.plot_img(np.fft.fftshift(np.log10(Nl)),f'{paths.debugdir}nl2d.png',arc_width=args.stamp_width_arcmin)
    # sys.exit()

    # Save weighted stacks and statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iNl = (1.0 / Nl) * kmask
    iNl[~np.isfinite(iNl)] = 0
    wkrecon = krecon * iNl
    s.add_to_stack("wk_real", wkrecon.real)
    s.add_to_stack("wk_imag", wkrecon.imag)
    s.add_to_stack("wk_iwt", iNl)

    # inverse variance noise weighting
    ivmean = 1.0 / nmean

    weight = ivmean
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
    j = j + 1


# collect from all MPI cores and calculate stacks
s.get_stacks()
s.get_stats()

if rank == 0:
    # Dump all collected statistics
    with bench.show("dump"):
        s.dump(paths.savedir)
        enmap.write_map_geometry(f"{paths.savedir}/map_geometry.fits", shape, wcs)
        enmap.write_map(f"{paths.savedir}/kmask.fits", kmask)
        enmap.write_map(f"{paths.savedir}/modrmap.fits", modrmap)
        np.savetxt(f"{paths.savedir}/bin_edges.txt", bin_edges)

    for ctkey in [
        "selected"
    ]:  # ,'large_kappa','no_stamp','high_noise','high_stamp','anomalous']:
        try:
            N = s.vectors[f"ct_{ctkey}"].sum()
            print(f"Number {ctkey} : {N}")
        except:
            pass
    # Sanity checks
    N_stamp = s.vectors[f"ct_{ctkey}"].sum()
    assert N_stamp == s.stack_count["kmap"]
    assert N_stamp == s.vectors["kw"].shape[0]

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
