import time as t
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pixell import enmap, reproject, utils, wcsutils, bunch
from orphics import mpi, maps, stats, io, cosmology, lensing
# import symlens
from symlens.qe import QE
import argparse
import sys, os

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
print("Paths: ", paths)


# RUN SETTING ------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "save_name", type=str, help="Name you want for your output."
)
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal or agora."
)
parser.add_argument(
    "which_cat", type=str, help="Choose the catalogue type e.g. halo or tsz."
)
parser.add_argument(
    "--cmb", action="store_true", help="CMB for high resolution leg."
)
parser.add_argument(
    "--cmb-tsz", action="store_true", help="CMB+tSZ for high resolution leg."
)
parser.add_argument(
    "--cmb-msub", action="store_true", help="CMB+tSZ-model for high resolution leg."
)
parser.add_argument(
    "--cmb-ksz", action="store_true", help="CMB+kSZ for hres and CMB+kSZ for gradient leg."
)
parser.add_argument(
    "--cmb-cib", action="store_true", help="CMB+CIB(90,150) for hres and CMB+CIB150 for gradient leg."
)
parser.add_argument(
    "--cmb-ksz-tsz", action="store_true", help="CMB+kSZ+tSZ for hres and CMB+kSZ for gradient leg."
)
parser.add_argument(
    "--cmb-cib-tsz", action="store_true", help="CMB+CIB+tSZ(90,150) for hres and CMB+CIB150 for gradient leg."
)
parser.add_argument(
    "--cmb-ksz-msub", action="store_true", help="CMB+kSZ+tSZ-model for hres and CMB+kSZ for gradient leg."
)
parser.add_argument(
    "--cmb-cib-msub", action="store_true", help="CMB+CIB+tSZ(90,150)-model for hres and CMB+CIB150 for gradient leg."
)
parser.add_argument(
    "--cmb-ksz-cib", action="store_true", help="CMB+kSZ+CIB(90,150) for hres and CMB+kSZ+CIB150 for gradient leg."
)
parser.add_argument(
    "--cmb-ksz-cib-tsz", action="store_true", help="CMB+kSZ+CIB+tSZ(90,150) for hres and CMB+kSZ+CIB150 for gradient leg."
)
parser.add_argument(
    "--cmb-ksz-cib-msub", action="store_true", help="CMB+kSZ+CIB+tSZ-model(90,150) for hres and CMB+kSZ+CIB150 for gradient leg."
)
parser.add_argument(
    "--is-meanfield", action="store_true", help="This is a mean-field run."
)
parser.add_argument(
    "--is-test", action="store_true", help="This is a test run for first 10 entries."
)
parser.add_argument(
    "--full-sample", action="store_true", help="Entire sample with SNR > 4 will be used.(Default cut is SNR > 5.5)"
)
parser.add_argument(
    "--highsnr-sample", action="store_true", help="Higher SNR cut sample (SNR > 7) will be used."
)
parser.add_argument(
    "--snmin", type=float, default=None, 
    help="Applying SNR threshold other than 4 (full sample) or 5.5 (cosmo sample). Make sure that corresponding halo catalogue is ready" 
)
parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy lensed map, choose this option. (available for websky and sehgal)"
)
parser.add_argument(
    "--inpaint", type=float, default=None, help="Inpainting for gradient leg. Enter the inpainting radius in arcmin."
)
args = parser.parse_args() 

if args.which_sim == "websky": output_path = paths.websky_output_path
elif args.which_sim == "sehgal": output_path = paths.sehgal_output_path
elif args.which_sim == "agora": output_path = paths.agora_output_path

save_name = args.save_name
save_dir = f"{output_path}/{save_name}"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)

simsuite_path = f"{paths.simsuite_path}/{args.which_sim}_{paths.simsuite_version}/"
cat_path = f"{paths.cat_path}/{paths.nemosim_version}/"
print(" ::: catalogue:", args.which_sim, args.which_cat, "[ simsuite ver:", paths.simsuite_version, "/ nemosim ver:", paths.nemosim_version, "]")

if args.cmb: print(" ::: this is CMB only run")
elif args.cmb_tsz: print(" ::: this is CMB + tSZ run")
elif args.cmb_msub: print(" ::: this is (CMB + tSZ - model) run")
elif args.cmb_ksz: print(" ::: this is CMB + kSZ run")
elif args.cmb_cib: print(" ::: this is CMB + CIB run")
elif args.cmb_ksz_cib: print(" ::: this is CMB + kSZ + CIB run")
elif args.cmb_ksz_tsz: print(" ::: this is CMB + kSZ + tSZ run")
elif args.cmb_cib_tsz: print(" ::: this is CMB + CIB + tSZ run")
elif args.cmb_ksz_msub: print(" ::: this is CMB + kSZ + tSZ - model run")
elif args.cmb_cib_msub: print(" ::: this is CMB + CIB + tSZ - model run")
elif args.cmb_ksz_cib_tsz: print(" ::: this is CMB + tSZ + kSZ + CIB run")
elif args.cmb_ksz_cib_msub: print(" ::: this is BASELINE (CMB + tSZ + kSZ + CIB - model) run")

if args.is_meanfield: print(" ::: this is a mean-field run")


# SIM SETTING ------------------------------------------------------------------

# for grad leg
xlmin = 200   
xlmax = 2000 

# for hres leg
ylmin = 200   
ylmax = 3500   # low lmax cut (high lmax cut is 6000)
lycut = 2 

# for kappa
klmin = 200   
klmax = 3000   # low lmax cut (high lmax cut is 5000)
lstep = 200

# for ILC
ilcmin = 200
ilcmax = 8000
ilc_beam_fwhm150 = 1.5
ilc_beam_fwhm090 = 2.2

# for PS fitting
highres_fiducial_rms = 15
highres_fit_ellmin = 500
highres_fit_ellmax = 8000
gradient_fiducial_rms = 35
gradient_fit_ellmin = 200
gradient_fit_ellmax = 3000
highres_fiducial_lknee = 3000
highres_fiducial_alpha = -4

# for taper
tap_per = 20.0 #12.0
pad_per = 3.0

# for beam
fwhm = 1.5
fwhm_plc = 5.0

# for white noise
nlevel = 15.0  
nlevel_plc = 35.0  

# for stamp cutout 
px = 0.5
width_deg = 120./60.
maxr = width_deg * utils.degree / 2.0

# for radially binned kappa profile 
arcmax = 15.0
arcstep = 1.5  # default is 1.5; test with 2.25 and 3.0
bin_edges = np.arange(0, arcmax, arcstep)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

# for kappa L 
ell_edges = np.arange(klmin, klmax, lstep)
ell_cents = (ell_edges[1:] + ell_edges[:-1]) / 2.0

print(" ::: grad  lmin =", xlmin, "and grad  lmax =", xlmax)
print(" ::: hres  lmin =", ylmin, "and hres  lmax =", ylmax)
print(" ::: kappa Lmin =", klmin, "and kappa Lmax =", klmax)
print(" ::: theta bin size =", arcstep, "arcmin")
print(" ::: L bin size =", lstep)

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret 

if args.inpaint is not None:
    inpaint_radius = args.inpaint # default is 4 arcmin 
    print(" ::: inpainting for gradient leg with a radius of", inpaint_radius, "arcmin")

# READING CATALOGUE ------------------------------------------------------------

if args.which_cat == "halo":
    if args.full_sample:
        cat = cat_path + f"{args.which_sim}_halo_true.txt"
    elif args.highsnr_sample:
        cat = cat_path + f"{args.which_sim}_halo_snr7_true.txt"
    else:
        # cat = cat_path + f"{args.which_sim}_halo_snr5p5.txt"
        cat = cat_path + f"{args.which_sim}_halo_snr5p5_true.txt" # matching true tsz cat below
        
    ras, decs, zs, masses = np.loadtxt(cat, unpack=True)

elif args.which_cat == "tsz_true":
    if args.full_sample:
        cat = cat_path + f"{args.which_sim}_tsz_true.txt"
    elif args.highsnr_sample:
        cat = cat_path + f"{args.which_sim}_tsz_snr7_true.txt"
    else:
        cat = cat_path + f"{args.which_sim}_tsz_snr5p5_true.txt" # M200c, true mass, lensed coordinates for Agora  

    ras, decs, zs, masses = np.loadtxt(cat, unpack=True)    

elif args.which_cat == "tsz":
    if args.which_sim == "websky":
        cat = paths.websky_tsz_cat
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        masses = hdu[1].data["M200m"] # 1e14 Msun: websky halo cat only provides M200m
        snr = hdu[1].data["SNR"] # fixed_SNR
        zs = hdu[1].data["redshift"] 
    else:
        if args.which_sim == "sehgal": cat = paths.sehgal_tsz_cat
        elif args.which_sim == "agora": cat = paths.agora_tsz_cat    
              
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        masses = hdu[1].data["M200c"] # 1e14 Msun  #FIXME: make it as an option
        # masses = hdu[1].data["M200m"] # 1e14 Msun #####
        snr = hdu[1].data["SNR"] # fixed_SNR
        zs = hdu[1].data["redshift"] 

    # SNR cut corresponding to cosmo sample is 5.5
    if not args.full_sample:
        if args.snmin is not None:
            snr_cut = args.snmin
        else:
            snr_cut = 5.5

        keep = (snr > snr_cut)

        ras = ras[keep]
        decs = decs[keep]
        masses = masses[keep]
        zs = zs[keep]
        snr = snr[keep]

    print(f" ::: min and max SNR      = {snr.min():.2f} and {snr.max():.2f} (SNR = {snr.mean():.2f} \xB1 {snr.std():.2f})")
print(f" ::: min and max redshift = {zs.min():.2f} and  {zs.max():.2f} (z = {zs.mean():.2f} \xB1 {zs.std():.2f})") 
print(f" ::: min and max M200c    = {masses.min():.2f} and {masses.max():.2f} (M = {masses.mean():.2f} \xB1 {masses.std():.2f})")


if args.is_meanfield:

    Nx = 100 * len(ras)

    if args.which_cat == "tsz_true": args.which_cat = "tsz"
    # load random catalogue - created by mapcat.py + randcat.py
    cat = cat_path + f"{args.which_sim}_{args.which_cat}_randoms.txt"
    ras, decs = np.loadtxt(cat, unpack=True)    

    # if not args.full_sample:
    ras = ras[:Nx]
    decs = decs[:Nx]

    print(" ::: reading random catalogue for mean-field run")

print(" ::: name of catalogue =", cat)
print(" ::: total number of clusters for stacking =", len(ras)) 


# READING MAPS -----------------------------------------------------------------

if args.which_sim == "websky":
    if args.high_accuracy:
        true = paths.websky_sim_path + paths.websky_kappa4p5_reproj
    else:
        true = paths.websky_sim_path + paths.websky_kappa_reproj
elif args.which_sim == "sehgal":
    if args.high_accuracy:
        true = paths.sehgal_sim_path + paths.sehgal_dkappa_reproj
    else:
        true = paths.sehgal_sim_path + paths.sehgal_kappa_reproj
elif args.which_sim == "agora":
    true = paths.agora_sim_path + paths.agora_kappa_reproj

    

print(" ::: preparing for OBSERVED maps")

h_cmb = f"{simsuite_path}h_ocmb.fits"
g_cmb = f"{simsuite_path}g_ocmb.fits"

print(" ::: reading true kappa map:", true)
print(" ::: reading lensed cmb map for hres:", h_cmb)
print(" ::: reading lensed cmb map for grad:", g_cmb)

true_map = enmap.read_map(true, delayed=False)
h_cmb_map = enmap.read_map(h_cmb, delayed=False)
g_cmb_map = enmap.read_map(g_cmb, delayed=False)

if args.cmb_tsz:

    cmb_tsz150 = f"{simsuite_path}h_ocmb_tsz150.fits"
    cmb_tsz090 = f"{simsuite_path}h_ocmb_tsz090.fits"

    print(" ::: reading lensed cmb + tsz map at 150GHz:", cmb_tsz150)
    print(" ::: reading lensed cmb + tsz map at  90GHz:", cmb_tsz090)

    cmb_tsz_map150 = enmap.read_map(cmb_tsz150, delayed=False)
    cmb_tsz_map090 = enmap.read_map(cmb_tsz090, delayed=False)

if args.cmb_msub: 

    cmb_msub090 = f"{simsuite_path}h_ocmb_msub090_test.fits" 
    cmb_msub150 = f"{simsuite_path}h_ocmb_msub150_test.fits"    

    print(" ::: reading the model subtracted maps!", cmb_msub090)
    print(" ::: reading the model subtracted maps!", cmb_msub150)

    cmb_msub_map090 = enmap.read_map(cmb_msub090, delayed=False)
    cmb_msub_map150 = enmap.read_map(cmb_msub150, delayed=False)

if args.cmb_ksz:

    h_cmb_ksz = f"{simsuite_path}h_ocmb_ksz.fits"
    g_cmb_ksz = f"{simsuite_path}g_ocmb_ksz.fits"

    print(" ::: reading lensed cmb + ksz map for hres:", h_cmb_ksz)
    print(" ::: reading lensed cmb + ksz map for grad:", g_cmb_ksz)

    h_cmb_ksz_map = enmap.read_map(h_cmb_ksz, delayed=False)
    g_cmb_ksz_map = enmap.read_map(g_cmb_ksz, delayed=False)

if args.cmb_cib: 

    h_cmb_cib090 = f"{simsuite_path}h_ocmb_cib090_5.fits"
    h_cmb_cib150 = f"{simsuite_path}h_ocmb_cib150_5.fits"
    g_cmb_cib = f"{simsuite_path}g_ocmb_cib5.fits" 

    print(" ::: reading lensed cmb + cib090 map for hres:", h_cmb_cib090)
    print(" ::: reading lensed cmb + cib150 map for hres:", h_cmb_cib150)
    print(" ::: reading lensed cmb + cib map for grad:", g_cmb_cib)

    h_cmb_cib090_map = enmap.read_map(h_cmb_cib090, delayed=False)
    h_cmb_cib150_map = enmap.read_map(h_cmb_cib150, delayed=False)
    g_cmb_cib_map = enmap.read_map(g_cmb_cib, delayed=False)


if args.cmb_ksz_tsz:

    h_cmb_ksz_tsz090 = f"{simsuite_path}h_ocmb_tsz090_ksz.fits"
    h_cmb_ksz_tsz150 = f"{simsuite_path}h_ocmb_tsz150_ksz.fits"
    g_cmb_ksz = f"{simsuite_path}g_ocmb_ksz.fits"

    print(" ::: reading lensed cmb + ksz + tsz090 map for hres:", h_cmb_ksz_tsz090)
    print(" ::: reading lensed cmb + ksz + tsz150 map for hres:", h_cmb_ksz_tsz150)
    print(" ::: reading lensed cmb + ksz map for grad:", g_cmb_ksz)

    h_cmb_ksz_tsz090_map = enmap.read_map(h_cmb_ksz_tsz090, delayed=False)
    h_cmb_ksz_tsz150_map = enmap.read_map(h_cmb_ksz_tsz150, delayed=False)
    g_cmb_ksz_map = enmap.read_map(g_cmb_ksz, delayed=False)

if args.cmb_cib_tsz:

    h_cmb_cib_tsz090 = f"{simsuite_path}h_ocmb_tsz090_cib5.fits"
    h_cmb_cib_tsz150 = f"{simsuite_path}h_ocmb_tsz150_cib5.fits"
    g_cmb_cib = f"{simsuite_path}g_ocmb_cib5.fits"

    print(" ::: reading lensed cmb + cib + tsz090 map for hres:", h_cmb_cib_tsz090)
    print(" ::: reading lensed cmb + cib + tsz150 map for hres:", h_cmb_cib_tsz150)
    print(" ::: reading lensed cmb + cib map for grad:", g_cmb_cib)

    h_cmb_cib_tsz090_map = enmap.read_map(h_cmb_cib_tsz090, delayed=False)
    h_cmb_cib_tsz150_map = enmap.read_map(h_cmb_cib_tsz150, delayed=False)
    g_cmb_cib_map = enmap.read_map(g_cmb_cib, delayed=False)

if args.cmb_ksz_msub:

    cmb_msub090 = f"{simsuite_path}h_ocmb_msub090_ksz_updated.fits" 
    cmb_msub150 = f"{simsuite_path}h_ocmb_msub150_ksz_updated.fits"    
    g_cmb_ksz = f"{simsuite_path}g_ocmb_ksz.fits"


    print(" ::: reading the model subtracted maps!", cmb_msub090)
    print(" ::: reading the model subtracted maps!", cmb_msub150)
    print(" ::: reading lensed cmb + ksz map for grad:", g_cmb_ksz)

    h_cmb_ksz_msub090_map = enmap.read_map(cmb_msub090, delayed=False)
    h_cmb_ksz_msub150_map = enmap.read_map(cmb_msub150, delayed=False)
    g_cmb_ksz_map = enmap.read_map(g_cmb_ksz, delayed=False)



if args.cmb_cib_msub:

    cmb_msub090 = f"{simsuite_path}h_ocmb_msub090_cib5_updated.fits" 
    cmb_msub150 = f"{simsuite_path}h_ocmb_msub150_cib5_updated.fits" 
    g_cmb_cib = f"{simsuite_path}g_ocmb_cib5.fits"

    print(" ::: reading the model subtracted maps!", cmb_msub090)
    print(" ::: reading the model subtracted maps!", cmb_msub150)
    print(" ::: reading lensed cmb + cib map for grad:", g_cmb_cib)

    h_cmb_cib_msub090_map = enmap.read_map(cmb_msub090, delayed=False)
    h_cmb_cib_msub150_map = enmap.read_map(cmb_msub150, delayed=False)
    g_cmb_cib_map = enmap.read_map(g_cmb_cib, delayed=False)




if args.cmb_ksz_cib:

    h_cmb_ksz_cib090 = f"{simsuite_path}h_ocmb_ksz_cib090_5.fits"
    h_cmb_ksz_cib150 = f"{simsuite_path}h_ocmb_ksz_cib150_5.fits"
    g_cmb_ksz_cib = f"{simsuite_path}g_ocmb_ksz_cib5.fits"

    print(" ::: reading lensed cmb + ksz + cib090 map for hres:", h_cmb_ksz_cib090)
    print(" ::: reading lensed cmb + ksz + cib150 map for hres:", h_cmb_ksz_cib150)
    print(" ::: reading lensed cmb + ksz + cib map for grad:", g_cmb_ksz_cib)

    h_cmb_ksz_cib090_map = enmap.read_map(h_cmb_ksz_cib090, delayed=False)
    h_cmb_ksz_cib150_map = enmap.read_map(h_cmb_ksz_cib150, delayed=False)
    g_cmb_ksz_cib_map = enmap.read_map(g_cmb_ksz_cib, delayed=False)

if args.cmb_ksz_cib_tsz: 

    h_cmb_tsz_ksz_cib090 = f"{simsuite_path}h_ocmb_tsz090_ksz_cib5.fits"
    h_cmb_tsz_ksz_cib150 = f"{simsuite_path}h_ocmb_tsz150_ksz_cib5.fits"
    g_cmb_ksz_cib = f"{simsuite_path}g_ocmb_ksz_cib5.fits"

    print(" ::: reading lensed cmb + tsz090 + ksz + cib090 map for hres:", h_cmb_tsz_ksz_cib090)
    print(" ::: reading lensed cmb + tsz150 + ksz + cib150 map for hres:", h_cmb_tsz_ksz_cib150)
    print(" ::: reading lensed cmb + ksz + cib map for grad:", g_cmb_ksz_cib)

    h_cmb_tsz_ksz_cib090_map = enmap.read_map(h_cmb_tsz_ksz_cib090, delayed=False)
    h_cmb_tsz_ksz_cib150_map = enmap.read_map(h_cmb_tsz_ksz_cib150, delayed=False)
    g_cmb_ksz_cib_map = enmap.read_map(g_cmb_ksz_cib, delayed=False)

if args.cmb_ksz_cib_msub:

    cmb_msub090 = f"{simsuite_path}h_ocmb_msub090_ksz_cib5_updated.fits" 
    cmb_msub150 = f"{simsuite_path}h_ocmb_msub150_ksz_cib5_updated.fits" 
    g_cmb_ksz_cib = f"{simsuite_path}g_ocmb_ksz_cib5.fits"


    print(" ::: reading the model subtracted maps!", cmb_msub090)
    print(" ::: reading the model subtracted maps!", cmb_msub150)
    print(" ::: reading lensed cmb + ksz + cib map for grad:", g_cmb_ksz_cib)

    h_cmb_ksz_cib090_msub_map = enmap.read_map(cmb_msub090, delayed=False)
    h_cmb_ksz_cib150_msub_map = enmap.read_map(cmb_msub150, delayed=False)
    g_cmb_ksz_cib_map = enmap.read_map(g_cmb_ksz_cib, delayed=False)



print(" ::: maps are ready!")



# FOR ILC / COADD ----------------------------------------------------------------

def fit_p1d(
    l_edges, cents, p1d, which, xout, bfunc1, bfunc2, rms=None, lmin=None, lmax=None
):
    # function for fitting 1D power spectrum of given stamp
    b1 = bfunc1 if bfunc1 is not None else lambda x: 1
    b2 = bfunc2 if bfunc2 is not None else lambda x: 1

    tfunc = lambda x: theory.lCl("TT", x) * b1(x) * b2(x)

    # PS fitting
    # Select region for fit
    sel = np.logical_and(cents > lmin, cents < lmax)
    delta_ells = np.diff(l_edges)[sel]
    ells = cents[sel]
    cls = p1d[sel]
    cltt = tfunc(ells)  # fiducial Cltt

    if which == "plc" or which == "act" or which == "act_cross":
        w0 = gradient_fiducial_rms if which=='plc' else highres_fiducial_rms
        sigma2 = stats.get_sigma2(ells, cltt, w0, delta_ells, fsky, ell0=0, alpha=1)
        func = stats.fit_cltt_power(ells, cls, tfunc, w0, sigma2, ell0=0, alpha=1, fix_knee=True)

    elif which == "apcross":
        w0 = gradient_fiducial_rms
        w0p = highres_fiducial_rms
        sigma2 = stats.get_sigma2(ells, cltt, w0, delta_ells, fsky, ell0=0, alpha=0, w0p=w0p, ell0p=0, alphap=1, clxx=cltt, clyy=cltt)
        func = stats.fit_cltt_power(ells, cls, tfunc, w0, sigma2, ell0=0, alpha=1, fix_knee=True)

    ret = func(xout)

    ret[xout < 2] = 0
    assert np.all(np.isfinite(ret))
    return ret


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

    ## NumPy 2.x no longer supports implicit batched solve for (n,2,2)x(n,2),
    ## so we explicitly solve each 2x2 system per ell.
    # num = np.linalg.solve(cov, ms)
    # den = np.linalg.solve(cov, rs)
    num = np.array([np.linalg.solve(cov[i], ms[i]) for i in range(nells)])
    den = np.array([np.linalg.solve(cov[i], rs[i]) for i in range(nells)])

    tcov = 1.0 / np.einsum("ij,ij->i", rs, den)
    ksolve = np.einsum("ij,ij->i", rs, num) * tcov
    assert np.all(np.isfinite(ksolve))
    ret = m1 * 0
    ret[sel] = ksolve
    tret = p11 * 0
    tret[sel] = tcov
    return ret, tret


##### TEST ##### ---------------------------------------------------------------

rstamp = 120.8 * utils.arcmin # to match the stamp size from the sim (241, 241)
res = 0.5 * utils.arcmin

lmin = 200
lmax = 3000 # 20000
rmin = 0.
rmax = 10 * utils.arcmin
rwidth = 1.0 * utils.arcmin

# agora
M200c = 6.37e14 
z = 0.6

# this returns conc = 3.326 using Klypin et al 2016 c(M,z)
# k2dmap is our fixed template on 2D map - this is tapered and filtered  
_,_,_,_,_,_,_,_,k2dmap,_ = lensing.kappa_nfw_profiley(mass=M200c,conc=None,z=z,z_s=1100.,background='critical',delta=200,apply_filter=True,lmin=lmin,lmax=lmax,res=res,rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
# # print(np.shape(k2dmap))
# # io.plot_img(k2dmap, 'kplot.png')

##### TEST ##### ---------------------------------------------------------------




# LOOP OVER ASSIGNED TASKS -----------------------------------------------------

if args.is_test: 
    print(" ::: this is a test run!")
    nsims = 10
else: nsims = len(ras)

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

j = 0  # local counter for this MPI task
for task in my_tasks:
    i = task
    cper = int((j + 1) / len(my_tasks) * 100.0)
    if rank==0: print(f"Rank {rank} performing task {task} as index {j} ({cper}% complete.).")

    coords = np.array([decs[i], ras[i]]) * utils.degree  

    # cut out a stamp from the simulated map
    kstamp = reproject.thumbnails(
        true_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # true kappa

    h_cmb = reproject.thumbnails(
        h_cmb_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True if args.high_accuracy else False # only True for maps made natively in CAR
    ) # lensed cmb for hres

    g_cmb = reproject.thumbnails(
        g_cmb_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True if args.high_accuracy else False # only True for maps made natively in CAR
    ) # lensed cmb for grad

    if args.cmb_tsz:
        h_cmb_tsz150 = reproject.thumbnails(
            cmb_tsz_map150,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz150

        h_cmb_tsz090 = reproject.thumbnails(
            cmb_tsz_map090,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz090

    if args.cmb_msub:
        h_cmb_msub150 = reproject.thumbnails(
            cmb_msub_map150,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz150 - model150        

        h_cmb_msub090 = reproject.thumbnails(
            cmb_msub_map090,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz090 - model090   

    if args.cmb_ksz:
        h_cmb_ksz = reproject.thumbnails(
            h_cmb_ksz_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz for hres

        g_cmb_ksz = reproject.thumbnails(
            g_cmb_ksz_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz for grad

    if args.cmb_cib:
        h_cmb_cib090 = reproject.thumbnails(
            h_cmb_cib090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib090 for hres

        h_cmb_cib150 = reproject.thumbnails(
            h_cmb_cib150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib150 for hres

        g_cmb_cib = reproject.thumbnails(
            g_cmb_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib150 for grad




    if args.cmb_ksz_tsz:
        h_cmb_ksz_tsz090 = reproject.thumbnails(
            h_cmb_ksz_tsz090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + tsz090 for hres

        h_cmb_ksz_tsz150 = reproject.thumbnails(
            h_cmb_ksz_tsz150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + tsz150 for hres

        g_cmb_ksz = reproject.thumbnails(
            g_cmb_ksz_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz for grad

    if args.cmb_cib_tsz:
        h_cmb_cib_tsz090 = reproject.thumbnails(
            h_cmb_cib_tsz090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib + tsz090 for hres

        h_cmb_cib_tsz150 = reproject.thumbnails(
            h_cmb_cib_tsz150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib + tsz150 for hres

        g_cmb_cib = reproject.thumbnails(
            g_cmb_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib for grad

    if args.cmb_ksz_msub:
        h_cmb_ksz_msub090 = reproject.thumbnails(
            h_cmb_ksz_msub090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + tsz090 - model for hres

        h_cmb_ksz_msub150 = reproject.thumbnails(
            h_cmb_ksz_msub150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + tsz150 - model for hres

        g_cmb_ksz = reproject.thumbnails(
            g_cmb_ksz_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz for grad

    if args.cmb_cib_msub:
        h_cmb_cib_msub090 = reproject.thumbnails(
            h_cmb_cib_msub090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib + tsz090 - model for hres

        h_cmb_cib_msub150 = reproject.thumbnails(
            h_cmb_cib_msub150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib + tsz150 - model for hres

        g_cmb_cib = reproject.thumbnails(
            g_cmb_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + cib for grad






    if args.cmb_ksz_cib:
        h_cmb_ksz_cib090 = reproject.thumbnails(
            h_cmb_ksz_cib090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + cib090 for hres

        h_cmb_ksz_cib150 = reproject.thumbnails(
            h_cmb_ksz_cib150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + cib150 for hres

        g_cmb_ksz_cib = reproject.thumbnails(
            g_cmb_ksz_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + cib150 for grad

    if args.cmb_ksz_cib_tsz:
        h_cmb_tsz_ksz_cib090 = reproject.thumbnails(
            h_cmb_tsz_ksz_cib090_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz090 + ksz + cib090 for hres

        h_cmb_tsz_ksz_cib150 = reproject.thumbnails(
            h_cmb_tsz_ksz_cib150_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz150 + ksz + cib150 for hres

        g_cmb_ksz_cib = reproject.thumbnails(
            g_cmb_ksz_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + cib for grad

    if args.cmb_ksz_cib_msub:
        h_cmb_ksz_cib090_msub = reproject.thumbnails(
            h_cmb_ksz_cib090_msub_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz090 + ksz + cib090 - model for hres

        h_cmb_ksz_cib150_msub = reproject.thumbnails(
            h_cmb_ksz_cib150_msub_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + tsz150 + ksz + cib150 - model for hres

        g_cmb_ksz_cib = reproject.thumbnails(
            g_cmb_ksz_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed cmb + ksz + cib for grad

    # initialise calculations based on geometry 
    if j == 0:     

        # get geometry and Fourier info   
        shape = kstamp.shape
        wcs = kstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        modrmap = enmap.modrmap(shape, wcs)

        assert wcsutils.equal(kstamp.wcs, h_cmb.wcs)
        assert wcsutils.equal(kstamp.wcs, g_cmb.wcs)

        if args.cmb_tsz:
            assert wcsutils.equal(kstamp.wcs, h_cmb_tsz150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_tsz090.wcs)

        elif args.cmb_msub:
            assert wcsutils.equal(kstamp.wcs, h_cmb_msub150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_msub090.wcs)    

        elif args.cmb_ksz:
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz.wcs)
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz.wcs) 

        elif args.cmb_cib:
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib090.wcs)      
            assert wcsutils.equal(kstamp.wcs, g_cmb_cib.wcs) 

        elif args.cmb_ksz_tsz:
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_tsz150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_tsz090.wcs)
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz.wcs) 

        elif args.cmb_cib_tsz:
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib_tsz150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib_tsz090.wcs)
            assert wcsutils.equal(kstamp.wcs, g_cmb_cib.wcs) 

        elif args.cmb_ksz_msub:
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_msub150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_msub090.wcs)
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz.wcs) 

        elif args.cmb_cib_msub:
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib_msub150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_cib_msub090.wcs)
            assert wcsutils.equal(kstamp.wcs, g_cmb_cib.wcs) 


        elif args.cmb_ksz_cib:
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_cib150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_cib090.wcs)      
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz_cib.wcs) 

        elif args.cmb_tsz:
            assert wcsutils.equal(kstamp.wcs, h_cmb_tsz_ksz_cib150.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_tsz_ksz_cib090.wcs)      
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz_cib.wcs)        

        elif args.cmb_ksz_cib_msub:
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_cib150_msub.wcs)
            assert wcsutils.equal(kstamp.wcs, h_cmb_ksz_cib090_msub.wcs)      
            assert wcsutils.equal(kstamp.wcs, g_cmb_ksz_cib.wcs)   

        # get an edge taper map and apodize
        taper = maps.get_taper(
            kstamp.shape,
            kstamp.wcs,
            taper_percent=tap_per,
            pad_percent=pad_per,
            weight=None,
        )
        taper = taper[0]  

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid
        beam2d = maps.gauss_beam(modlmap, fwhm)
        beam2d_plc = maps.gauss_beam(modlmap, fwhm_plc)    

        # if args.cmb_tsz or args.cmb_msub or args.cmb_cib or args.cmb_ksz_cib or args.cmb_ksz_cib_tsz or args.cmb_ksz_cib_msub: 
        if not (args.cmb or args.cmb_ksz):

            bfunc150 = lambda x: maps.gauss_beam(ilc_beam_fwhm150, x)
            bfunc90 = lambda x: maps.gauss_beam(ilc_beam_fwhm090, x)    

            act_150_kbeam2d = bfunc150(modlmap)
            act_90_kbeam2d = bfunc90(modlmap)
            plc_kbeam2d = beam2d_plc

        # build Fourier space masks for lensing reconstruction
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax) # grad
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=lycut, lycut=lycut) # hres
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax) # kappa

        # get theory spectrum and build interpolated 2D Fourier CMB from theory and maps      
        theory = cosmology.default_theory()
        ucltt2d = theory.lCl("TT", modlmap)

        if not (args.cmb or args.cmb_ksz):
            # bin size and range for 1D binned power spectrum
            minell = 2 * maps.minimum_ell(shape, wcs)
            l_edges = np.arange(minell / 2, 8001, minell)
            lbinner = stats.bin2D(modlmap, l_edges)
            # PS correction factor
            w2 = np.mean(taper ** 2) 
            # fsky for bandpower variance
            fsky = enmap.area(shape, wcs) * w2 / 4.0 / np.pi

    # same filter as the post-reconstuction for true kappa (also tapered!)
    k_stamp = maps.filter_map(kstamp*taper, kmask)   
    s.add_to_stack("kstamp", k_stamp)
    binned_true = bin(k_stamp, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("tk1d", binned_true)     

    if args.cmb:
        hres = h_cmb
        g_cmb = g_cmb
    elif args.cmb_tsz:
        hres150 = h_cmb_tsz150
        hres90 = h_cmb_tsz090
        g_cmb = g_cmb
    elif args.cmb_msub:
        hres150 = h_cmb_msub150
        hres90 = h_cmb_msub090
        g_cmb = g_cmb
    elif args.cmb_ksz:
        hres = h_cmb_ksz
        g_cmb = g_cmb_ksz
    elif args.cmb_cib:
        hres150 = h_cmb_cib150
        hres90 = h_cmb_cib090      
        g_cmb = g_cmb_cib

    elif args.cmb_ksz_tsz:
        hres150 = h_cmb_ksz_tsz150
        hres90 = h_cmb_ksz_tsz090      
        g_cmb = g_cmb_ksz
    elif args.cmb_cib_tsz:
        hres150 = h_cmb_cib_tsz150
        hres90 = h_cmb_cib_tsz090      
        g_cmb = g_cmb_cib
    elif args.cmb_ksz_msub:
        hres150 = h_cmb_ksz_msub150
        hres90 = h_cmb_ksz_msub090      
        g_cmb = g_cmb_ksz
    elif args.cmb_cib_msub:
        hres150 = h_cmb_cib_msub150
        hres90 = h_cmb_cib_msub090      
        g_cmb = g_cmb_cib


    elif args.cmb_ksz_cib:
        hres150 = h_cmb_ksz_cib150
        hres90 = h_cmb_ksz_cib090      
        g_cmb = g_cmb_ksz_cib
    elif args.cmb_ksz_cib_tsz:
        hres150 = h_cmb_tsz_ksz_cib150
        hres90 = h_cmb_tsz_ksz_cib090      
        g_cmb = g_cmb_ksz_cib
    elif args.cmb_ksz_cib_msub:
        hres150 = h_cmb_ksz_cib150_msub
        hres90 = h_cmb_ksz_cib090_msub    
        g_cmb = g_cmb_ksz_cib


    # filter weird stamps (temporary1)
    if not (args.cmb or args.cmb_ksz):
        if np.any(abs(hres150) > 1e3) or np.any(abs(hres90) > 1e3):
            print(f"{task} has anomalously high hres150 or hres90")
            continue 
    else:
        if np.any(abs(hres) > 1e3):
            print(f"{task} has anomalously high hres")
            continue 


    if not (args.cmb or args.cmb_ksz):
        tapered_hres150 = hres150 * taper
        tapered_hres90 = hres90 * taper
    else:
        tapered_hres = hres * taper





    s.add_to_stack("grad2d_before", g_cmb)
    g_filtered = maps.filter_map(g_cmb, ymask)
    s.add_to_stack("grad2d_before_filtered", g_filtered)
    binned_grad = bin(g_cmb, g_cmb.modrmap() * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("grad1d_before", binned_grad) 

    if args.inpaint:
        if j == 0:
            from inpaint_utils import Inpainter
            grad_inpainter = Inpainter(
                fine_shape=g_cmb.shape,
                fine_wcs=g_cmb.wcs,
                hole_radius_arcmin=inpaint_radius,
                theory=theory,
                beam_fn=lambda x: maps.gauss_beam(fwhm_plc, x),
                noise_uK_arcmin=nlevel_plc,
            )
        g_cmb = grad_inpainter.inpaint(g_cmb)

    # taper stamp    
    tapered_grad = g_cmb * taper  


    s.add_to_stack("grad2d_after", g_cmb)
    g_filtered = maps.filter_map(g_cmb, ymask)
    s.add_to_stack("grad2d_after_filtered", g_filtered)
    binned_grad = bin(g_cmb, g_cmb.modrmap() * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("grad1d_after", binned_grad) 



    if not (args.cmb or args.cmb_ksz):
        # ILC will return a beam-deconvolved Fourier transformed stamp
        k_grad = enmap.fft(tapered_grad, normalize="phys")
        k150 = enmap.fft(tapered_hres150, normalize="phys")
        k90 = enmap.fft(tapered_hres90, normalize="phys")
        assert np.all(np.isfinite(k150)) 
        assert np.all(np.isfinite(k90))  

        # Fourier map -> PS
        pow = lambda x, y: (x * y.conj()).real 

        # measure the binned power spectrum from given stamp
        act_cents, act_p1d_150 = lbinner.bin(pow(k150, k150) / w2)
        act_cents, act_p1d_90 = lbinner.bin(pow(k90, k90) / w2)
        act_cents, act_p1d_150_90 = lbinner.bin(pow(k150, k90) / w2)
        plc_cents, plc_p1d = lbinner.bin(pow(k_grad, k_grad) / w2)

        # fit power spectra 
        tclaa_150 = fit_p1d(
            l_edges,
            act_cents,
            act_p1d_150,
            "act",
            modlmap,
            bfunc150,
            bfunc150,
            rms=highres_fiducial_rms,
            lmin=highres_fit_ellmin,
            lmax=highres_fit_ellmax,
        )

        tclaa_90 = fit_p1d(
            l_edges,
            act_cents,
            act_p1d_90,
            "act",
            modlmap,
            bfunc90,
            bfunc90,
            rms=highres_fiducial_rms,
            lmin=highres_fit_ellmin,
            lmax=highres_fit_ellmax,
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
            lmin=highres_fit_ellmin,
            lmax=highres_fit_ellmax,
        )

        tclpp = fit_p1d(
            l_edges,
            plc_cents,
            plc_p1d,
            "plc",
            modlmap,
            lambda x: maps.gauss_beam(x, fwhm_plc),
            lambda x: maps.gauss_beam(x, fwhm_plc),
            rms=gradient_fiducial_rms,
            lmin=gradient_fit_ellmin,
            lmax=gradient_fit_ellmax,
        )

        # ILC / coadd 
        k_hres, tclaa = ilc(
            modlmap,
            k150,
            k90,
            tclaa_150,
            tclaa_90,
            tclaa_150_90,
            act_150_kbeam2d,
            act_90_kbeam2d,
        ) # beam deconvolved

        k_grad = k_grad / plc_kbeam2d  
        tclpp = tclpp / (plc_kbeam2d ** 2.0)

        # # fit cross-power of gradient and ILC hres 
        # cents, c_ap = lbinner.bin(pow(k_hres, k_grad) / w2)
        # tclap = fit_p1d(
        #     l_edges, 
        #     cents, 
        #     c_ap, 
        #     "apcross", 
        #     modlmap, 
        #     None, 
        #     None, 
        #     rms=0, 
        #     lmin=highres_fit_ellmin, 
        #     lmax=gradient_fit_ellmax
        # ) # shouldn't this be beam deconvolved as well?
        # tclap[~np.isfinite(tclap)] = 0

    else:
        # cmb only hres run doesn't involve ILC
        # total spectrum includes beam-deconvolved noise       
        npower = (nlevel * np.pi/180./60.)**2.
        npower_plc = (nlevel_plc * np.pi/180./60.)**2.

        tclaa = ucltt2d + npower/beam2d**2.
        tclpp = ucltt2d + npower_plc/beam2d_plc**2.

        # get a beam-deconvolved Fourier transformed stamp
        k_grad = enmap.fft(tapered_grad, normalize="phys")/beam2d_plc
        k_hres = enmap.fft(tapered_hres, normalize="phys")/beam2d

    assert np.all(np.isfinite(k_grad)) 
    assert np.all(np.isfinite(k_hres))     

    tclaa[~np.isfinite(tclaa)] = 0
    tclpp[~np.isfinite(tclpp)] = 0

    # build symlens dictionary 
    feed_dict = {
        "uC_T_T" : ucltt2d, 
        "tC_A_T_A_T": tclaa,  # approximate ACT power spectrum, ACT beam deconvolved
        "tC_P_T_P_T": tclpp,  # approximate Planck power spectrum, Planck beam deconvolved
        "tC_A_T_P_T": ucltt2d,  # same lensed theory as above, no instrumental noise
        "tC_P_T_A_T": ucltt2d,  # same lensed theory as above, no instrumental noise
        "X" : k_grad, # grad leg
        "Y" : k_hres, # hres leg
    }  

    # Sanity check
    for key in feed_dict.keys():
        assert np.all(np.isfinite(feed_dict[key]))

    # ask for reconstruction in Fourier space
    cqe = QE(
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
    rkmap = cqe.reconstruct(feed_dict, xname="X_l1", yname="Y_l2", physical_units=True)

    assert np.all(np.isfinite(rkmap))


    # for cross-correlation check ----------------------------------------------

    ikmap = enmap.fft(k_stamp, normalize="phys") # k_stamp is tapered and filtered true kappa
    assert np.all(np.isfinite(ikmap))

    p2d_auto = (ikmap * ikmap.conj()).real
    p2d_cros = (ikmap * rkmap.conj()).real
      
    p1d_auto = bin(p2d_auto, modlmap, ell_edges)
    p1d_cros = bin(p2d_cros, modlmap, ell_edges)

    s.add_to_stats("p1d_auto", p1d_auto) 
    s.add_to_stats("p1d_cros", p1d_cros) 
    s.add_to_stack("true_real", ikmap.real)
    s.add_to_stack("true_imag", ikmap.imag)
    s.add_to_stack("recon_real", rkmap.real)
    s.add_to_stack("recon_imag", rkmap.imag)

    # for cross-correlation check ----------------------------------------------

    ktemp = k2dmap # this is tapered and filtered template (in real space)

    template = enmap.fft(ktemp, normalize="phys") 
    assert np.all(np.isfinite(template))

    p2d_cross_t = (ikmap * template.conj()).real # cross with true kappa  
    p2d_cross_r = (rkmap * template.conj()).real # cross with reconstruction - what about the meanfield here? post-processing!
    p2d_auto_template = (template * template.conj()).real

    p1d_cross_t = bin(p2d_cross_t, modlmap, ell_edges)
    p1d_cross_r = bin(p2d_cross_r, modlmap, ell_edges)
    p1d_auto_template = bin(p2d_auto_template, modlmap, ell_edges)

    s.add_to_stats("p1d_cross_t", p1d_cross_t) 
    s.add_to_stats("p1d_cross_r", p1d_cross_r)
    s.add_to_stats("p1d_auto_template", p1d_auto_template)

    l = ell_cents * ell_cents
    s.add_to_stats("p1d_cross_t_l2", l * p1d_cross_t) 
    s.add_to_stats("p1d_cross_r_l2", l * p1d_cross_r)
    s.add_to_stats("p1d_auto_template_l2", l * p1d_auto_template)

    s.add_to_stack("temp", k2dmap)    
    binned_temp = bin(k2dmap, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("temp1d", binned_temp) 





    # transform to real space
    kappa = enmap.ifft(rkmap, normalize="phys").real    

    # filter weird stamps (temporary2)
    if np.any(np.abs(kappa) > 15):
        print(f"{task} has large kappa")
        continue


    # stack reconstructed kappa     
    s.add_to_stack("lstamp", kappa)    
    binned_kappa = bin(kappa, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("k1d", binned_kappa) 

    # stack stamps pre-reconstruction as well (same filter as hres leg)
    if args.cmb:
        hres = h_cmb
        grad = g_cmb
    if args.cmb_tsz:
        hres = h_cmb_tsz150
        grad = g_cmb
    elif args.cmb_msub:
        hres = h_cmb_msub150
        grad = g_cmb 
    elif args.cmb_ksz:
        hres = h_cmb_ksz
        grad = g_cmb_ksz
    elif args.cmb_cib:
        hres = h_cmb_cib150
        grad = g_cmb_cib

    elif args.cmb_ksz_tsz:
        hres = h_cmb_ksz_tsz150
        grad = g_cmb_ksz
    elif args.cmb_cib_tsz:
        hres = h_cmb_cib_tsz150
        grad = g_cmb_cib
    elif args.cmb_ksz_msub:
        hres = h_cmb_ksz_msub150
        grad = g_cmb_ksz
    elif args.cmb_cib_msub:
        hres = h_cmb_cib_msub150
        grad = g_cmb_cib

    elif args.cmb_ksz_cib:
        hres = h_cmb_ksz_cib150
        grad = g_cmb_ksz_cib
    elif args.cmb_ksz_cib_tsz:
        hres = h_cmb_tsz_ksz_cib150
        grad = g_cmb_ksz_cib
    elif args.cmb_ksz_cib_msub:
        hres = h_cmb_ksz_cib150_msub
        grad = g_cmb_ksz_cib

    hres_stamp = maps.filter_map(hres, ymask) 
    grad_stamp = maps.filter_map(grad, ymask)  
    s.add_to_stack("hres_stamp_st", hres_stamp)
    s.add_to_stack("grad_stamp_st", grad_stamp) 
    s.add_to_stack("hres_noF_stamp_st", hres)
    s.add_to_stack("grad_noF_stamp_st", grad) 

    # save the list of masses and redshifts for matched stack mass fitting
    if not args.is_meanfield:     
        s.add_to_stats("redshift", (zs[i],))
        s.add_to_stats("masses", (masses[i],))

    j = j + 1


            
# COLLECT FROM ALL MPI CORES AND CALCULATE STACKS ------------------------------

s.get_stacks()
s.get_stats()
   
if rank==0:

    if args.is_meanfield:
        save_name = save_name + "_mf" 

    # stacks before lensing reconstruction   
    hres_st = s.stacks["hres_stamp_st"]
    grad_st = s.stacks["grad_stamp_st"]
    hres_noF_st = s.stacks["hres_noF_stamp_st"]
    grad_noF_st = s.stacks["grad_noF_stamp_st"] 
 
    hres_zoom = hres_st[100:140,100:140]  
    grad_zoom = grad_st[100:140,100:140] 
    hres_noF_zoom = hres_noF_st[100:140,100:140]  
    grad_noF_zoom = grad_noF_st[100:140,100:140] 
         
    modrmap = hres_zoom.modrmap()
    modrmap = np.rad2deg(modrmap)*60. 
    
    hbinned = bin(hres_zoom, modrmap, bin_edges)
    gbinned = bin(grad_zoom, modrmap, bin_edges)
    hbinned_noF = bin(hres_noF_zoom, modrmap, bin_edges)
    gbinned_noF = bin(grad_noF_zoom, modrmap, bin_edges)

    io.plot_img(hres_st, f"{save_dir}/{save_name}_0hres.png")  
    io.plot_img(grad_st, f"{save_dir}/{save_name}_0grad.png")             
    io.plot_img(hres_zoom, f"{save_dir}/{save_name}_0hres_zoom.png")   
    io.plot_img(grad_zoom, f"{save_dir}/{save_name}_0grad_zoom.png")  
    np.save(f"{save_dir}/{save_name}_0hres.npy", hres_st) 
    np.save(f"{save_dir}/{save_name}_0grad.npy", grad_st)  

    io.save_cols(f"{save_dir}/{save_name}_0binned_hres.txt", (centers, hbinned))   
    io.save_cols(f"{save_dir}/{save_name}_0binned_grad.txt", (centers, gbinned))      

    io.plot_img(hres_noF_st, f"{save_dir}/{save_name}_0hres_noF.png")  
    io.plot_img(grad_noF_st, f"{save_dir}/{save_name}_0grad_noF.png")             
    io.plot_img(hres_noF_zoom, f"{save_dir}/{save_name}_0hres_noF_zoom.png")   
    io.plot_img(grad_noF_zoom, f"{save_dir}/{save_name}_0grad_noF_zoom.png")  
    np.save(f"{save_dir}/{save_name}_0hres_noF.npy", hres_noF_st) 
    np.save(f"{save_dir}/{save_name}_0grad_noF.npy", grad_noF_st)  
    io.save_cols(f"{save_dir}/{save_name}_0binned_hres_noF.txt", (centers, hbinned_noF))   
    io.save_cols(f"{save_dir}/{save_name}_0binned_grad_noF.txt", (centers, gbinned_noF))  

    # reconstructed lensing field     
    kmap = s.stacks["kstamp"]
    lmap = s.stacks["lstamp"]
    
    kmap_zoom = kmap[100:140,100:140] 
    lmap_zoom = lmap[100:140,100:140] 
            
    modrmap = kmap_zoom.modrmap()
    modrmap = np.rad2deg(modrmap)*60. 

    kbinned = bin(kmap_zoom, modrmap, bin_edges)
    lbinned = bin(lmap_zoom, modrmap, bin_edges)

    io.plot_img(kmap, f"{save_dir}/{save_name}_1tkappa.png")   
    io.plot_img(lmap, f"{save_dir}/{save_name}_1rkappa.png")                 
    io.plot_img(kmap[100:140,100:140], f"{save_dir}/{save_name}_1tkappa_zoom.png")   
    io.plot_img(lmap[100:140,100:140], f"{save_dir}/{save_name}_1rkappa_zoom.png")  
    np.save(f"{save_dir}/{save_name}_1tkappa.npy", kmap)     
    np.save(f"{save_dir}/{save_name}_1rkappa.npy", lmap)                 
    io.save_cols(f"{save_dir}/{save_name}_1binned_tkappa_from2D.txt", (centers, kbinned))
    io.save_cols(f"{save_dir}/{save_name}_1binned_rkappa_from2D.txt", (centers, lbinned))

    tbinned = s.stats["tk1d"]["mean"]
    tcovm = s.stats["tk1d"]["covmean"]
    tcorr = stats.cov2corr(s.stats["tk1d"]["covmean"])
    terrs = s.stats["tk1d"]["errmean"]
    
    binned = s.stats["k1d"]["mean"]
    covm = s.stats["k1d"]["covmean"]
    corr = stats.cov2corr(s.stats["k1d"]["covmean"])
    errs = s.stats["k1d"]["errmean"]

    np.savetxt(f"{save_dir}/{save_name}_1tkappa_errs.txt", terrs)               
    np.savetxt(f"{save_dir}/{save_name}_1rkappa_errs.txt", errs)    
    np.savetxt(f"{save_dir}/{save_name}_1tkappa_covm.txt", tcovm)
    np.savetxt(f"{save_dir}/{save_name}_1rkappa_covm.txt", covm)
    np.save(f"{save_dir}/{save_name}_1tkappa_corr.npy", tcorr)  
    np.save(f"{save_dir}/{save_name}_1rkappa_corr.npy", corr) 
    io.save_cols(f"{save_dir}/{save_name}_1binned_tkappa.txt", (centers, tbinned))
    io.save_cols(f"{save_dir}/{save_name}_1binned_rkappa.txt", (centers, binned)) 

    enmap.write_map(f"{save_dir}/{save_name}_kmask.fits", kmask)   
    np.savetxt(f"{save_dir}/{save_name}_bin_edges.txt", bin_edges)

    if not args.is_meanfield:
        np.savetxt(f"{save_dir}/{save_name}_z_mass1e14.txt", np.c_[s.vectors["redshift"], s.vectors["masses"]])
        np.savetxt(f"{save_dir}/{save_name}_tkappa_1d_ind.txt", np.c_[s.vectors["tk1d"]])
        np.savetxt(f"{save_dir}/{save_name}_rkappa_1d_ind.txt", np.c_[s.vectors["k1d"]])

    # for cross-correlation check ----------------------------------------------

    if args.is_meanfield:
        save_dir = save_dir + "_mf" 
        os.makedirs(save_dir)

    s.dump(f"{save_dir}")

    # for cross-correlation check ----------------------------------------------



elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)
