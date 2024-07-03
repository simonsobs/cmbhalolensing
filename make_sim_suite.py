import time as t
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, wcsutils, bunch
from orphics import maps, io
import healpy as hp
from past.utils import old_div
from websky_cosmo import *
import argparse, sys

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
print("Paths: ", paths)




# RUN SETTING ------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "save_name", type=str, help="Name you want for your output."
)
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal."
)
parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
)
args = parser.parse_args() 


output_path = paths.simsuite_path
save_name = args.save_name
if args.which_sim == "websky": 
    print(" ::: producing maps and catalogues for WEBSKY sim")
    save_name = "websky_" + save_name
    sim_path = paths.websky_sim_path
elif args.which_sim == "sehgal": 
    print(" ::: producing maps and catalogues for SEHGAL sim")
    save_name = "sehgal_" + save_name
    sim_path = paths.sehgal_sim_path
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)






# SIM SETTING ------------------------------------------------------------------

freq_sz = 150
print(" ::: map frequency at %d GHz" %freq_sz)

px = 0.5
width_deg = 120./60.
maxr = width_deg * utils.degree / 2.0

fwhm = 1.5
nlevel = 15.0  






# BEAM CONVOLUTION -------------------------------------------------------------

def apply_beam(imap): 
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    if args.which_sim == "websky": nside = 8192
    elif args.which_sim == "sehgal": nside = 4096
    alm_lmax = nside * 3
    bfunc = lambda x: maps.gauss_beam(fwhm, x)  
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))


# CATALOGUE RESAMPLE -----------------------------------------------------------

def resample_halo(ras, decs, mass, zs, tsz_cat, snr_cut, num_bins):
    # resample websky or sehgal halo catalogue 
    # to match the mass distribution of a given tsz catalogue
    # num_bins: number of mass bins for resampling 

    hdu = fits.open(tsz_cat)
    tsz_mass = hdu[1].data["M200m"] # in units of 1e14 Msun
    snr = hdu[1].data["SNR"]
    tsz_mass = tsz_mass[snr > snr_cut] 
    print(" ::: snr cut = %.1f" %snr_cut) 
    print(" ::: total number of clusters in tsz cat =", len(tsz_mass))  

    # create mass bins based on the tsz masses
    mbins = np.linspace(tsz_mass.min(), tsz_mass.max(), num_bins)
    mhist, mbin_edges = np.histogram(tsz_mass, bins=mbins)

    resampled_ras = []
    resampled_decs = []
    resampled_mass = []
    resampled_zs = []    

    # loop through each mass bin and resample the input catalogue
    for i in range(len(mhist)):
        min_mass = mbin_edges[i]
        max_mass = mbin_edges[i+1]

        # find indices of input cat entries within the current mass bin
        ind0 = np.where((mass > min_mass) & (mass < max_mass))[0]
        bin_ras = ras[ind0]
        bin_decs = decs[ind0]
        bin_mass = mass[ind0]
        bin_zs = zs[ind0]        

        # skip empty bins
        if mhist[i] == 0 or len(bin_mass) == 0:
            continue

        # resample the entries to match the histogran count
        np.random.seed(100)
        ind1 = np.random.choice(len(bin_mass), size=mhist[i], replace=True) 
        resampled_ras.append(bin_ras[ind1])
        resampled_decs.append(bin_decs[ind1])
        resampled_mass.append(bin_mass[ind1])
        resampled_zs.append(bin_zs[ind1])

    ras = np.concatenate(resampled_ras, axis=None)
    decs = np.concatenate(resampled_decs, axis=None)
    mass = np.concatenate(resampled_mass, axis=None)
    zs = np.concatenate(resampled_zs, axis=None)

    print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
    print(" ::: min and max M200m = %.2f and %.2f" %(mass.min(), mass.max()), "(mean = %.2f)" %mass.mean()) 
    print(" ::: total number of halos after resampling =", len(ras)) 

    return ras, decs, mass, zs


# PREPARING MAPS ---------------------------------------------------------------

# paths for reprojected maps
if args.which_sim == "websky": 
    if args.high_accuracy: 
        r_lmap = sim_path + paths.websky_dlensed_reproj
        r_kmap = sim_path + paths.websky_kappa4p5_reproj
        r_tszmap = sim_path + paths.websky_dtsz_reproj
    else: 
        r_lmap = sim_path + paths.websky_lensed_cmb_reproj
        r_kmap = sim_path + paths.websky_kappa_reproj
        r_tszmap = sim_path + paths.websky_tsz_reproj
elif args.which_sim == "sehgal": 
    r_lmap = sim_path + paths.sehgal_lensed_cmb_reproj
    r_kmap = sim_path + paths.sehgal_kappa_reproj
    r_tszmap = sim_path + paths.sehgal_tsz_reproj


# reading lensed cmb map
try:
    lmap = enmap.read_map(r_lmap, delayed=False)
    print(" ::: reading reprojected lensed cmb map:", r_lmap)

except:
    shape, wcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj="car")
    if args.which_sim == "websky": 
        if args.high_accuracy:
            ifile = f"{paths.mat_path}/dlensed.fits"
            lmap = enmap.read_map(ifile)
            print(" ::: reading lensed cmb map:", ifile) 
        else:
            ifile = f"{sim_path}lensed_alm.fits"
            alm = np.complex128(hp.read_alm(ifile, hdu=(1, 2, 3)))
            lmap = curvedsky.alm2map(alm[0,:], enmap.empty(shape, wcs, dtype=np.float64))
            print(" ::: reading lensed alm map and converting to lensed cmb map:", ifile) 

    elif args.which_sim == "sehgal": 
        ifile = f"{sim_path}Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits"
        hmap = hp.read_map(ifile).astype(np.float64)
        lmap = reproject.healpix2map(hmap, shape, wcs)
        print(" ::: reading lensed cmb map:", ifile) 

    enmap.write_map(r_lmap, lmap)
print("lmap", lmap.shape) 
shape, wcs = lmap.shape, lmap.wcs


# reading true kappa map 
try:
    kmap = enmap.read_map(r_kmap, delayed=False)
    print(" ::: reading reprojected true kappa map:", r_kmap)

except:
    if args.which_sim == "websky": 
        if args.high_accuracy:
            ifile = f"{sim_path}kap_lt4.5.fits" # CMB lensing convergence from z<4.5 from halo+field websky          
        else:
            ifile = f"{sim_path}kap.fits" # CMB lensing convergence from 0<z<1100
        hmap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        kmap = reproject.healpix2map(hmap, shape, wcs)[0,:,:]  

    elif args.which_sim == "sehgal":         
        ifile = f"{sim_path}healpix_4096_KappaeffLSStoCMBfullsky.fits"
        hmap = hp.read_map(ifile).astype(np.float64)
        kmap = reproject.healpix2map(hmap, shape, wcs)

    enmap.write_map(r_kmap, kmap)
    print(" ::: reading and reprojecting true kappa map:", ifile) 
print("kmap", kmap.shape)


# reading tsz map
try:
    tszmap = enmap.read_map(r_tszmap, delayed=False)
    print(" ::: reading reprojected tsz map:", r_tszmap) 

except:
    if args.which_sim == "websky": 
        ifile = f"{sim_path}tsz_8192.fits"
        hmap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        ymap = reproject.healpix2map(hmap, shape, wcs)[0,:,:]

    elif args.which_sim == "sehgal":
        ifile = f"{sim_path}tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
        hmap = hp.read_map(ifile).astype(np.float64)
        ymap = reproject.healpix2map(hmap, shape, wcs)

    print(" ::: reading ymap:", ifile) 
    print("ymap", ymap.shape) 

    # convert compton-y to delta-T (in uK) 
    tcmb = 2.726
    tcmb_uK = tcmb * 1e6 #micro-Kelvin
    H_cgs = 6.62608e-27
    K_cgs = 1.3806488e-16

    def fnu(nu):
        """
        nu in GHz
        tcmb in Kelvin
        """
        mu = H_cgs*(1e9*nu)/(K_cgs*tcmb)
        ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
        return ans

    tszmap = fnu(freq_sz) * ymap * tcmb_uK
    print(" ::: converting ymap to tsz map at %d GHz" %freq_sz)
    enmap.write_map(r_tszmap, tszmap)
print("tszmap", tszmap.shape)


assert wcsutils.equal(kmap.wcs, lmap.wcs)  
assert wcsutils.equal(kmap.wcs, tszmap.wcs)


scmb = lmap
scmb_tsz = lmap + tszmap
print(" ::: SIGNAL maps are ready!")

white_noise = maps.white_noise(lmap.shape, lmap.wcs, noise_muK_arcmin=nlevel)

print(" ::: applying beam and adding white noise of %.1f uK" %nlevel)
bcmb = apply_beam(scmb)
ocmb = apply_beam(scmb) + white_noise
ocmb_tsz = apply_beam(scmb_tsz) + white_noise
print(" ::: OBSERVED maps are ready!")

save_scmb = f"{save_dir}scmb.fits"
save_scmb_tsz = f"{save_dir}scmb_tsz.fits"
save_ocmb = f"{save_dir}ocmb.fits"
save_ocmb_tsz = f"{save_dir}ocmb_tsz.fits"
save_bcmb = f"{save_dir}bcmb.fits"

print(" ::: saving all maps")
enmap.write_map(save_scmb, scmb)
enmap.write_map(save_scmb_tsz, scmb_tsz)
enmap.write_map(save_ocmb, ocmb)
enmap.write_map(save_ocmb_tsz, ocmb_tsz)
enmap.write_map(save_bcmb, bcmb)


# PREPARING HALO CATALOGUES ----------------------------------------------------

if args.which_sim == "websky": 

    cat = f"{sim_path}halos.pksc"
    print(" ::: loading the entire websky halo catalogue:", cat) 
    f = open(cat, "r")
    N = np.fromfile(f, count=3, dtype=np.int32)[0]
    catalog = np.fromfile(f, count=N*10, dtype=np.float32)
    catalog = np.reshape(catalog, (N,10))

    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
    R  = catalog[:,6] # Mpc

    # convert to mass, redshift, RA and DEC
    rho        = 2.775e11*omegam*h**2     # Msun/Mpc^3
    M200m      = 4*np.pi/3.*rho*R**3      # in Msun 
    chi        = np.sqrt(x**2+y**2+z**2)  # Mpc
    theta, phi = hp.vec2ang(np.column_stack((x,y,z))) # in radians

    ras      = np.rad2deg(phi)
    decs     = np.rad2deg(np.pi/2. - theta)  
    mass     = M200m / 1e14
    zs       = zofchi(chi)                 

    act_cat = paths.websky_tsz_cat


elif args.which_sim == "sehgal": 

    cat = f"{sim_path}halo_nbody.ascii"
    print(" ::: loading the entire segal halo catalogue:", cat)
    f_coords = open(cat, "r")
    data = np.genfromtxt(f_coords)
    ras, decs = data[:,1], data[:,2] # degrees
    zs = data[:,0]
    mass = data[:,12] # this is M200c
    # mass = data[:,14] # this is M500c
    mass = mass / 1e14    

    act_cat = paths.sehgal_tsz_cat

print(" ::: tsz cat used is:", act_cat)
print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 
print(" ::: min and max M200m = %.2f and %.2f" %(mass.min(), mass.max()), "(mean = %.2f)" %mass.mean()) 
print(" ::: total number of halos = ", len(ras))

ras, decs, mass, zs = resample_halo(ras, decs, mass, zs, act_cat, snr_cut=4, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo.txt", np.c_[ras, decs, zs, mass])

ras, decs, mass, zs = resample_halo(ras, decs, mass, zs, act_cat, snr_cut=5.5, num_bins=30)
np.savetxt(f"{save_dir}{args.which_sim}_halo_snr5p5.txt", np.c_[ras, decs, zs, mass])






elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)




    











