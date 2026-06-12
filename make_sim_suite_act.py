# sim_suite for model image subtracion run 
# this should be done in ACT footprint maps (otherwise, cannot be subtracted)

import time as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, wcsutils, bunch
from orphics import maps, io
import healpy as hp
from past.utils import old_div
import argparse
import sys

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

args = parser.parse_args() 


output_path = paths.simsuite_path
save_name = args.save_name
if args.which_sim == "websky": 
    print(" ::: producing maps for WEBSKY sim")
    save_name = "websky_" + save_name
    sim_path = paths.websky_sim_path
elif args.which_sim == "sehgal": 
    print(" ::: producing maps for SEHGAL sim")
    save_name = "sehgal_" + save_name
    sim_path = paths.sehgal_sim_path
elif args.which_sim == "agora": 
    print(" ::: producing maps for AGORA sim")
    save_name = "agora_" + save_name
    sim_path = paths.agora_sim_path    
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)


# SIM SETTING ------------------------------------------------------------------

px = 0.5

fwhm_plc = 5.0

fwhm150 = 1.5
fwhm090 = 2.2

nlevel = 15.0  
nlevel_plc = 35.0 


# BEAM CONVOLUTION -------------------------------------------------------------

def apply_beam(imap, ifwhm): 
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    if args.which_sim == "websky" or args.which_sim == "agora": nside = 8192
    elif args.which_sim == "sehgal": nside = 4096
    alm_lmax = nside * 3
    bfunc = lambda x: maps.gauss_beam(ifwhm, x)  
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))

# PREPARING MAPS ---------------------------------------------------------------

# reading lensed cmb map

ifile = f"{paths.mat_path}/dlensed_actfoot.fits" # act footprint 
lmap = enmap.read_map(ifile)
print(" ::: reading a map for ACT footprint geometry:", ifile) 
shape, wcs = lmap.shape, lmap.wcs


if args.which_sim == "agora":
    ifile = f"{sim_path}cmb/len/tqu1/agora_tqu1_phiNG_seed1_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits"
    hmap,_,_ = hp.read_map(ifile, field=[0,1,2]).astype(np.float64)
    lmap = reproject.healpix2map(hmap, shape, wcs)
    print(" ::: reading lensed cmb map:", ifile) 

print("lmap", lmap.shape) 
shape, wcs = lmap.shape, lmap.wcs



# reading tsz map

if args.which_sim == "agora":
    ifile = f"{sim_path}/tsz/unl/agora_utszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18.fits" 
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

freq_sz = 90
print(" ::: map frequency at %d GHz" %freq_sz)
tszmap090 = fnu(freq_sz) * ymap * tcmb_uK
print(" ::: converting ymap to tsz map at %d GHz" %freq_sz)
print("tszmap090", tszmap090.shape)

freq_sz = 150
print(" ::: map frequency at %d GHz" %freq_sz)
tszmap150 = fnu(freq_sz) * ymap * tcmb_uK
print(" ::: converting ymap to tsz map at %d GHz" %freq_sz)
print("tszmap150", tszmap150.shape)


# reading ksz map

# r_kszmap = sim_path + paths.agora_ksz_reproj
if args.which_sim == "agora":
    ifile = f"{sim_path}ksz/unl/agora_ukszNG_bahamas80_bnd_unb_1.0e+12_1.0e+18.fits" 
    hmap = hp.read_map(ifile).astype(np.float64)
    kszmap = reproject.healpix2map(hmap, shape, wcs)

# enmap.write_map(r_kszmap, kszmap)
print(" ::: reading and reprojecting ksz map:", ifile) 
print("kszmap", kszmap.shape) 



# reading cib map
if args.which_sim == "agora":

    fluxcut_val = 5. # flux per pixel [mJy]

    def flux_density_to_temp(freq_GHz):
        # get factor for converting delta flux density in [MJy/sr] to delta T in CMB units [uK]
        freq = float(freq_GHz)
        x = freq / 56.8
        return (1.05e3 * (np.exp(x)-1)**2 *
                np.exp(-x) * (freq / 100)**-4)

    freq = 150
    ifile = f"{sim_path}cib/uK/len/act/agora_len_mag_cibmap_act_{freq}ghz_uk.fits" # in uK 
    hmap = hp.read_map(ifile).astype(np.float64)
    print(hmap.min(), hmap.max(), hmap.mean())

    nside = hp.get_nside(hmap) # extract the HEALPix resolution 
    print(nside)

    pixel_solid_angle = hp.nside2pixarea(nside)     # in steradian 
    lim = fluxcut_val * 1e-9 / pixel_solid_angle    # [mJy] to [MJy/sr]
    uK_lim = flux_density_to_temp(freq) * lim       # [MJy/sr] to [uK] 

    print(f"conversion factor: {flux_density_to_temp(freq):.2f}")
    print(f"uK threshold: {uK_lim:.2f}")
    print(f"number of pixels above threshold: {np.sum(hmap > uK_lim)} ({np.sum(hmap > uK_lim)/np.sum(hmap)*100:.6f} percent)")
    print(f"implementing flux cut on map of {fluxcut_val} mJy per pixel")
    hmap[hmap > uK_lim]=0.
    print(hmap.min(), hmap.max(), hmap.mean())

    cibmap150 = reproject.healpix2map(hmap, shape, wcs, method="spline")


    freq = 90
    ifile = f"{sim_path}cib/uK/len/act/agora_len_mag_cibmap_act_{freq}ghz_uk.fits" # in uK 
    hmap = hp.read_map(ifile).astype(np.float64)
    print(hmap.min(), hmap.max(), hmap.mean())

    nside = hp.get_nside(hmap) # extract the HEALPix resolution 
    print(nside)

    pixel_solid_angle = hp.nside2pixarea(nside)     # in steradian 
    lim = fluxcut_val * 1e-9 / pixel_solid_angle    # [mJy] to [MJy/sr]
    uK_lim = flux_density_to_temp(freq) * lim       # [MJy/sr] to [uK] 

    print(f"conversion factor: {flux_density_to_temp(freq):.2f}")
    print(f"uK threshold: {uK_lim:.2f}")
    print(f"number of pixels above threshold: {np.sum(hmap > uK_lim)} ({np.sum(hmap > uK_lim)/np.sum(hmap)*100:.6f} percent)")
    print(f"implementing flux cut on map of {fluxcut_val} mJy per pixel")
    hmap[hmap > uK_lim]=0.
    print(hmap.min(), hmap.max(), hmap.mean())


    cibmap090 = reproject.healpix2map(hmap, shape, wcs, method="spline")

    # enmap.write_map(r_cibmap, cibmap)
    print("cibmap090", cibmap090.shape)
    print("cibmap150", cibmap150.shape)


# assert wcsutils.equal(lmap.wcs, kmap.wcs) 
assert wcsutils.equal(lmap.wcs, tszmap090.wcs) 
assert wcsutils.equal(lmap.wcs, tszmap150.wcs)
assert wcsutils.equal(lmap.wcs, kszmap.wcs)
assert wcsutils.equal(lmap.wcs, cibmap090.wcs) 
assert wcsutils.equal(lmap.wcs, cibmap150.wcs)





scmb_tsz150 = lmap + tszmap150
scmb_tsz090 = lmap + tszmap090 
scmb_tsz150_ksz = lmap + tszmap150 + kszmap
scmb_tsz090_ksz = lmap + tszmap090 + kszmap 
scmb_tsz150_cib = lmap + tszmap150 + cibmap150
scmb_tsz090_cib = lmap + tszmap090 + cibmap090
scmb_tsz150_ksz_cib = lmap + tszmap150 + kszmap + cibmap150
scmb_tsz090_ksz_cib = lmap + tszmap090 + kszmap + cibmap090

print(" ::: SIGNAL maps are ready!")
print(" ::: applying beam and adding white noise")

h_ocmb_tsz150 = apply_beam(scmb_tsz150, fwhm150) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=12)
h_ocmb_tsz090 = apply_beam(scmb_tsz090, fwhm090) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=13)
h_ocmb_tsz150_ksz = apply_beam(scmb_tsz150_ksz, fwhm150) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=12)
h_ocmb_tsz090_ksz = apply_beam(scmb_tsz090_ksz, fwhm090) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=13)
h_ocmb_tsz150_cib = apply_beam(scmb_tsz150_cib, fwhm150) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=12)
h_ocmb_tsz090_cib = apply_beam(scmb_tsz090_cib, fwhm090) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=13)
h_ocmb_tsz150_ksz_cib = apply_beam(scmb_tsz150_ksz_cib, fwhm150) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=12)
h_ocmb_tsz090_ksz_cib = apply_beam(scmb_tsz090_ksz_cib, fwhm090) + maps.white_noise(shape, wcs, noise_muK_arcmin=nlevel, seed=13)



print(" ::: OBSERVED maps are ready!")

# save_h_ocmb_tsz150_ksz = f"{save_dir}h_ocmb_tsz150_ksz_actfoot.fits"
# save_h_ocmb_tsz090_ksz = f"{save_dir}h_ocmb_tsz090_ksz_actfoot.fits"
# enmap.write_map(save_h_ocmb_tsz150_ksz, h_ocmb_tsz150_ksz)
# enmap.write_map(save_h_ocmb_tsz090_ksz, h_ocmb_tsz090_ksz)
# sys.exit()




# reading the model map -- model map is beam convolved 

## only tSZ
# model150 = f"/home3/eunseong/nemo-sims/nemo-sim-kit48/agora_cmb_tsz/clusterModelMaps/clusterModelMap_f150.fits"
# model090 = f"/home3/eunseong/nemo-sims/nemo-sim-kit48/agora_cmb_tsz/clusterModelMaps/clusterModelMap_f090.fits"
## tSZ with kSZ 
model150 = f"/home3/eunseong/nemo-sims/nemo-sim-kit61/20260320_cmb_tsz_ksz/20260320_cmb_tsz_ksz/clusterModelMaps/clusterModelMap_f150.fits"
model090 = f"/home3/eunseong/nemo-sims/nemo-sim-kit61/20260320_cmb_tsz_ksz/20260320_cmb_tsz_ksz/clusterModelMaps/clusterModelMap_f090.fits"
## tSZ with CIB
# model150 = f"/home3/eunseong/nemo-sims/nemo-sim-kit62/20260320_cmb_tsz_cib/20260320_cmb_tsz_cib/clusterModelMaps/clusterModelMap_f150.fits"
# model090 = f"/home3/eunseong/nemo-sims/nemo-sim-kit62/20260320_cmb_tsz_cib/20260320_cmb_tsz_cib/clusterModelMaps/clusterModelMap_f090.fits"
## tSZ with kSZ+CIB
# model150 = f"/home3/eunseong/nemo-sims/nemo-sim-kit63/20260320_cmb_tsz_ksz_cib/20260320_cmb_tsz_ksz_cib/clusterModelMaps/clusterModelMap_f150.fits"
# model090 = f"/home3/eunseong/nemo-sims/nemo-sim-kit63/20260320_cmb_tsz_ksz_cib/20260320_cmb_tsz_ksz_cib/clusterModelMaps/clusterModelMap_f090.fits"



print(" ::: reading model image map:", model150)
print(" ::: reading model image map:", model090)

modelmap150 = enmap.read_map(model150, delayed=False)
modelmap090 = enmap.read_map(model090, delayed=False)
print(modelmap150.shape, modelmap090.shape)



print(" ::: subtracting the model maps!")


# h_ocmb_msub150 = h_ocmb_tsz150 - modelmap150
# h_ocmb_msub090 = h_ocmb_tsz090 - modelmap090
h_ocmb_msub150_ksz = h_ocmb_tsz150_ksz - modelmap150
h_ocmb_msub090_ksz = h_ocmb_tsz090_ksz - modelmap090
# h_ocmb_msub150_cib = h_ocmb_tsz150_cib - modelmap150
# h_ocmb_msub090_cib = h_ocmb_tsz090_cib - modelmap090
# h_ocmb_msub150_ksz_cib = h_ocmb_tsz150_ksz_cib - modelmap150
# h_ocmb_msub090_ksz_cib = h_ocmb_tsz090_ksz_cib - modelmap090


print(" ::: model subtracted maps are ready!")



# save_h_ocmb_msub150 = f"{save_dir}h_ocmb_msub150_test.fits"
# save_h_ocmb_msub090 = f"{save_dir}h_ocmb_msub090_test.fits"
save_h_ocmb_msub150_ksz = f"{save_dir}h_ocmb_msub150_ksz_updated.fits"
save_h_ocmb_msub090_ksz = f"{save_dir}h_ocmb_msub090_ksz_updated.fits"
# save_h_ocmb_msub150_cib = f"{save_dir}h_ocmb_msub150_cib5_updated.fits"
# save_h_ocmb_msub090_cib = f"{save_dir}h_ocmb_msub090_cib5_updated.fits"
# save_h_ocmb_msub150_ksz_cib = f"{save_dir}h_ocmb_msub150_ksz_cib5_updated.fits"
# save_h_ocmb_msub090_ksz_cib = f"{save_dir}h_ocmb_msub090_ksz_cib5_updated.fits"



# enmap.write_map(save_h_ocmb_msub150, h_ocmb_msub150)
# enmap.write_map(save_h_ocmb_msub090, h_ocmb_msub090)
enmap.write_map(save_h_ocmb_msub150_ksz, h_ocmb_msub150_ksz)
enmap.write_map(save_h_ocmb_msub090_ksz, h_ocmb_msub090_ksz)
# enmap.write_map(save_h_ocmb_msub150_cib, h_ocmb_msub150_cib)
# enmap.write_map(save_h_ocmb_msub090_cib, h_ocmb_msub090_cib)
# enmap.write_map(save_h_ocmb_msub150_ksz_cib, h_ocmb_msub150_ksz_cib)
# enmap.write_map(save_h_ocmb_msub090_ksz_cib, h_ocmb_msub090_ksz_cib)



print(" ::: all maps are saved! yayyyyyy")
 






elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)




    











