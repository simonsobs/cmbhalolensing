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

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_paths.yml"))
sim_spec = bunch.Bunch(io.config_from_yaml("input/sim_info.yml"))
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
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
)
args = parser.parse_args() 


output_path = paths.simsuite_path
sim_name = args.which_sim
if args.high_accuracy:
    sim_name += "_high_acc"
sim_info = sim_spec[sim_name]
save_name = args.save_name
print(f" ::: producing maps for {sim_name.upper()} sim")
save_name = sim_name + "_" + save_name
sim_path = paths[f"{sim_name}_sim_path"]
 
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)


# SIM SETTING ------------------------------------------------------------------

freq_sz = 150
print(" ::: map frequency at %d GHz" %freq_sz)

px = 0.5
fwhm = 1.5
nlevel = 15.0  


# BEAM CONVOLUTION -------------------------------------------------------------

def apply_beam(imap): 
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    # if args.which_sim == "websky" or args.which_sim == "agora": nside = 8192
    nside = sim_info['nside']
    alm_lmax = nside * 3
    bfunc = lambda x: maps.gauss_beam(fwhm, x)  
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))



# PREPARING MAPS ---------------------------------------------------------------

# paths for reprojected maps

r_lmap = sim_path + paths[f"{sim_name}_cmb_reproj"]
r_kmap = sim_path + paths[f"{sim_name}_kappa_reproj"]
r_tszmap = sim_path + paths[f"{sim_name}_tsz_reproj"]

def file_to_map(fname, shape=None, wcs=None):
    try:            #first try opening as a 2d enmap
        omap = enmap.read_map(fname)
    
    except:

        try:        #try with healpy as a map
            hmap = hp.read_map(fname).astype(np.float64)
            omap = reproject.healpix2map(hmap, shape, wcs)

        except:
            
            try:        #load with healpy as alms with multiple hdus
                alm = np.complex128(hp.read_alm(fname, hdu=(1, 2, 3)))
                omap = curvedsky.alm2map(alm[0,:], enmap.empty(shape, wcs, dtype=np.float64))

            except:     #load with healpy as alms with one hdu
                hmap  = hp.read_alm(fname).astype(np.complex128)
                omap = curvedsky.alm2map(hmap, enmap.empty(shape, wcs, dtype=np.float64))

    return omap


# reading lensed cmb map
try:
    lmap = enmap.read_map(r_lmap, delayed=False)
    print(" ::: reading reprojected lensed cmb map:", r_lmap)

except:
    shape, wcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj="car")
    ifile = sim_path + sim_info['lensed_cmb']
    lmap = file_to_map(ifile, shape, wcs) #should preserve high accuracy (doesn't read shape and wcs)


#     if args.which_sim == "websky": 
#         if args.high_accuracy:
#             ifile = f"{paths.mat_path}/dlensed.fits"
#             lmap = enmap.read_map(ifile)
#             print(" ::: reading lensed cmb map:", ifile) 
#         else:
#             ifile = f"{sim_path}lensed_alm.fits"
#             alm = np.complex128(hp.read_alm(ifile, hdu=(1, 2, 3)))
#             lmap = curvedsky.alm2map(alm[0,:], enmap.empty(shape, wcs, dtype=np.float64))
#             print(" ::: reading lensed alm map and converting to lensed cmb map:", ifile) 

#     elif args.which_sim == "sehgal": 
#         ifile = f"{sim_path}Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits"
#         hmap = hp.read_map(ifile).astype(np.float64)
#         lmap = reproject.healpix2map(hmap, shape, wcs)
#         print(" ::: reading lensed cmb map:", ifile) 

#     elif args.which_sim == "agora":
#         ifile = f"{sim_path}cmb/len/tqu1/agora_tqu1_phiNG_seed1_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits"
#         hmap,_,_ = hp.read_map(ifile, field=[0,1,2]).astype(np.float64)
#         lmap = reproject.healpix2map(hmap, shape, wcs)
#         print(" ::: reading lensed cmb map:", ifile) 

    enmap.write_map(r_lmap, lmap)
print("lmap", lmap.shape) 
shape, wcs = lmap.shape, lmap.wcs


# reading true kappa map 
try:
    kmap = enmap.read_map(r_kmap, delayed=False)
    print(" ::: reading reprojected true kappa map:", r_kmap)

except:
    ifile = sim_path + sim_info['true_kappa']
    kmap = file_to_map(ifile, shape, wcs)

#     if args.which_sim == "websky": 
#         if args.high_accuracy:
#             ifile = f"{sim_path}kap_lt4.5.fits" # CMB lensing convergence from z<4.5 from halo+field websky          
#         else:
#             ifile = f"{sim_path}kap.fits" # CMB lensing convergence from 0<z<1100
#         hmap = hp.read_map(ifile).astype(np.float64)
#         ymap = reproject.healpix2map(hmap, shape, wcs)

#     elif args.which_sim == "sehgal":         
#         ifile = f"{sim_path}healpix_4096_KappaeffLSStoCMBfullsky.fits"
#         hmap = hp.read_map(ifile).astype(np.float64)
#         kmap = reproject.healpix2map(hmap, shape, wcs)

#     elif args.which_sim == "agora":
#         ifile = f'{sim_path}../../kappa_alm_recon_agora_phiNG_phi1_seed1.fits'
#         hmap  = hp.read_alm(ifile).astype(np.complex128)
#         kmap = curvedsky.alm2map(hmap, enmap.empty(shape, wcs, dtype=np.float64))


    enmap.write_map(r_kmap, kmap)
    print(" ::: reading and reprojecting true kappa map:", ifile) 
print("kmap", kmap.shape)


# reading tsz map
try:
    tszmap = enmap.read_map(r_tszmap, delayed=False)
    print(" ::: reading reprojected tsz map:", r_tszmap) 

except:
    ifile = sim_path + sim_info['tsz']
    ymap = file_to_map(ifile, shape, wcs)

#     if args.which_sim == "websky": 
#         ifile = f"{sim_path}tsz_8192.fits"
#         hmap = hp.read_map(ifile).astype(np.float64)
#         ymap = reproject.healpix2map(hmap, shape, wcs)

#     elif args.which_sim == "sehgal":
#         ifile = f"{sim_path}tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits"
#         hmap = hp.read_map(ifile).astype(np.float64)
#         ymap = reproject.healpix2map(hmap, shape, wcs)

#     elif args.which_sim == "agora":
#         ifile = f"{sim_path}tsz/len/agora_ltszNG_bahamas78_bnd_unb_1.0e+12_1.0e+18_lensed.fits"
#         hmap = hp.read_map(ifile).astype(np.float64)
#         ymap = reproject.healpix2map(hmap, shape, wcs)

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






elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)




    











