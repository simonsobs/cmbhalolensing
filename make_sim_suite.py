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
parser.add_argument("freq", type=int, help="Frequency for tSZ and CIB")

parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
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

freq_sz = args.freq
print(" ::: map frequency at %d GHz" %freq_sz)

px = 0.5
fwhm = 1.5
nlevel = 15.0  






# BEAM CONVOLUTION -------------------------------------------------------------

def apply_beam(imap): 
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    if args.which_sim == "websky" or args.which_sim == "agora": nside = 8192
    elif args.which_sim == "sehgal": nside = 4096
    alm_lmax = nside * 3
    bfunc = lambda x: maps.gauss_beam(fwhm, x)  
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))



# PREPARING MAPS ---------------------------------------------------------------

# paths for reprojected maps
if args.which_sim == "websky": 
    if args.high_accuracy: 
        r_lmap = sim_path + paths.websky_dlensed_reproj
        r_kmap = sim_path + paths.websky_kappa4p5_reproj
        r_tsz090map = sim_path + paths.websky_dtsz090_reproj
        r_tsz150map = sim_path + paths.websky_dtsz150_reproj
        r_kszmap = sim_path + paths.websky_dksz_reproj
        r_cib093map = sim_path + paths.websky_dcib093_reproj
        r_cib145map = sim_path + paths.websky_dcib145_reproj
    else: 
        r_lmap = sim_path + paths.websky_cmb_reproj
        r_kmap = sim_path + paths.websky_kappa_reproj
        r_tszmap = sim_path + paths.websky_tsz_reproj
elif args.which_sim == "sehgal": 
    r_lmap = sim_path + paths.sehgal_cmb_reproj
    r_kmap = sim_path + paths.sehgal_kappa_reproj
    r_tszmap = sim_path + paths.sehgal_tsz_reproj
elif args.which_sim == "agora":
    r_lmap = sim_path + paths.agora_cmb_reproj
    r_kmap = sim_path + paths.agora_kappa_reproj
    r_tsz090map = sim_path + paths.agora_tsz090_reproj
    r_tsz150map = sim_path + paths.agora_tsz150_reproj
    r_cib093map = sim_path + paths.agora_cib090_reproj
    r_cib145map = sim_path + paths.agora_cib150_reproj
    r_cib220map = sim_path + paths.agora_cib220_reproj
    r_kszmap = sim_path + paths.agora_ksz_reproj


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

    elif args.which_sim == "agora":
        ifile = f"{sim_path}cmb/len/tqu1/agora_tqu1_phiNG_seed1_lmax16000_nside8192_interp1.6_method1_pol_1_lensedmap.fits"
        hmap,_,_ = hp.read_map(ifile, field=[0,1,2]).astype(np.float64)
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

    elif args.which_sim == "agora":
        ifile = f'{sim_path}../../kappa_alm_recon_agora_phiNG_phi1_seed1.fits'
        hmap  = hp.read_alm(ifile).astype(np.complex128)
        kmap = curvedsky.alm2map(hmap, enmap.empty(shape, wcs, dtype=np.float64))


    enmap.write_map(r_kmap, kmap)
    print(" ::: reading and reprojecting true kappa map:", ifile) 
print("kmap", kmap.shape)


# reading tsz map
try:
    if (args.which_sim == "websky") or (args.which_sim == "agora"):
        if freq_sz == 90: r_tszmap = r_tsz090map
        elif freq_sz == 150: r_tszmap = r_tsz150map

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

    elif args.which_sim == "agora":
        ifile = f"{sim_path}tsz/len/mdpl2_ltszNG_bahamas80_rot_sum_4_176_bnd_unb_1.0e+12_1.0e+18_v103021_lmax24000_nside8192_interp1.0_method1_1_lensed_map.fits"
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


# reading ksz map
try:
    kszmap = enmap.read_map(r_kszmap, delayed=False)
    print(" ::: reading reprojected ksz map:", r_kszmap) 

except:
    if args.which_sim == "websky": 
        ifile = f"{sim_path}ksz.fits"
        hmap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        kszmap = reproject.healpix2map(hmap, shape, wcs)[0,:,:]

    if args.which_sim == "agora":
        ifile = f"{sim_path}/ksz/mdpl2_lkszNG_bahamas80_rot_sum_4_176_bnd_unb_1.0e+12_1.0e+18_v103021_lmax24000_nside8192_interp1.0_method1_1_lensed_map.fits"
        hmap = hp.read_map(ifile).astype(np.float64)
        kszmap = reproject.healpix2map(hmap, shape, wcs)

    enmap.write_map(r_kszmap, kszmap)
    print(" ::: reading and reprojecting ksz map:", ifile) 
print("kszmap", kszmap.shape) 


# reading cib map
try:
    if (args.which_sim == "websky") or (args.which_sim == "agora"):
        if freq_sz == 90: r_cibmap = r_cib093map
        elif freq_sz == 150: r_cibmap = r_cib145map

    cibmap = enmap.read_map(r_cibmap, delayed=False)
    print(" ::: reading reprojected cib map:", r_cibmap)

except:
    if args.which_sim == "websky": 

        # frequency for CIB 
        if freq_sz == 90: ifile = f"{sim_path}cib_nu0093.fits"       
        elif freq_sz == 150: ifile = f"{sim_path}cib_nu0145.fits"
        
        hmap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        cibmap_i = reproject.healpix2map(hmap, shape, wcs)[0,:,:]

        print(" ::: reading cib map:", ifile)
        print("cibmap", np.shape(cibmap_i))

        
        tcmb = 2.726
        tcmb_uK = tcmb * 1e6 #micro-Kelvin
        kboltz = 1.3806503e-23 #MKS
        hplanck = 6.626068e-34 #MKS
        clight = 299792458.0 #MKS

        def ItoDeltaT(nu):
            # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
            #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
            #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
            nu *= 1e9
            X = hplanck*nu/(kboltz*tcmb)
            dBnudT = (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/tcmb_uK * 1e26
            return 1./dBnudT

        # frequency for CIB 
        if freq_sz == 150: freq_cib = 145
        elif freq_sz == 90: freq_cib = 93

        cibmap = ItoDeltaT(freq_cib) * cibmap_i * 1.e6 # MJy/sr -> Jy/sr -> Tcmb

        print(" ::: converting cib map at %d GHz" %freq_cib)

    if args.which_sim == "agora":

        # frequency for CIB (already in cmb units)
        if freq_sz == 90: ifile = f"{sim_path}/cib/act/len/uk/mdpl2_len_mag_cibmap_act_90_uk.fit"       
        elif freq_sz == 150: ifile = f"{sim_path}cib/act/len/uk/mdpl2_len_mag_cibmap_act_150_uk.fits"
        
        hmap = hp.read_map(ifile).astype(np.float64)
        cibmap = reproject.healpix2map(hmap, shape, wcs)

        print(" ::: reading and reprojecting cib map:", ifile)
    enmap.write_map(r_cibmap, cibmap)
print("cibmap", cibmap.shape)

assert wcsutils.equal(kmap.wcs, lmap.wcs)  
assert wcsutils.equal(kmap.wcs, tszmap.wcs)
assert wcsutils.equal(kmap.wcs, kszmap.wcs)
assert wcsutils.equal(kmap.wcs, cibmap.wcs)

scmb = lmap
scmb_cib = lmap + cibmap
scmb_ksz_cib = lmap + kszmap + cibmap

scmb_tsz = lmap + tszmap
scmb_tsz_cib = lmap + tszmap + cibmap
scmb_tsz_ksz_cib = lmap + tszmap + kszmap + cibmap

print(" ::: SIGNAL maps are ready!")

white_noise = maps.white_noise(lmap.shape, lmap.wcs, noise_muK_arcmin=nlevel)

print(" ::: applying beam and adding white noise of %.1f uK" %nlevel)

bcmb = apply_beam(scmb)
bcmb_ksz_cib = apply_beam(scmb_ksz_cib)

ocmb = apply_beam(scmb) + white_noise
ocmb_cib = apply_beam(scmb_cib) + white_noise
ocmb_ksz_cib = apply_beam(scmb_ksz_cib) + white_noise

ocmb_tsz = apply_beam(scmb_tsz) + white_noise
ocmb_tsz_cib = apply_beam(scmb_tsz_cib) + white_noise
ocmb_tsz_ksz_cib = apply_beam(scmb_tsz_ksz_cib) + white_noise

print(" ::: OBSERVED maps are ready!")

save_scmb = f"{save_dir}scmb.fits"
save_scmb_cib = f"{save_dir}scmb_cib.fits"
save_scmb_ksz_cib = f"{save_dir}scmb_ksz_cib.fits"

save_scmb_tsz = f"{save_dir}scmb_tsz.fits"
save_scmb_tsz_cib = f"{save_dir}scmb_tsz_cib.fits"
save_scmb_tsz_ksz_cib = f"{save_dir}scmb_tsz_ksz_cib.fits"

save_ocmb = f"{save_dir}ocmb.fits"
save_ocmb_cib = f"{save_dir}ocmb_cib.fits"
save_ocmb_ksz_cib = f"{save_dir}ocmb_ksz_cib.fits"

save_ocmb_tsz = f"{save_dir}ocmb_tsz.fits"
save_ocmb_tsz_cib = f"{save_dir}ocmb_tsz_cib.fits"
save_ocmb_tsz_ksz_cib = f"{save_dir}ocmb_tsz_ksz_cib.fits"

save_bcmb = f"{save_dir}bcmb.fits"
save_bcmb_ksz_cib = f"{save_dir}bcmb_ksz_cib.fits"

print(" ::: saving all maps")

enmap.write_map(save_scmb, scmb)
enmap.write_map(save_scmb_cib, scmb_cib)
enmap.write_map(save_scmb_ksz_cib, scmb_ksz_cib)

enmap.write_map(save_scmb_tsz, scmb_tsz)
enmap.write_map(save_scmb_tsz_cib, scmb_tsz_cib)
enmap.write_map(save_scmb_tsz_ksz_cib, scmb_tsz_ksz_cib)

enmap.write_map(save_ocmb, ocmb)
enmap.write_map(save_ocmb_cib, ocmb_cib)
enmap.write_map(save_ocmb_ksz_cib, ocmb_ksz_cib)

enmap.write_map(save_ocmb_tsz, ocmb_tsz)
enmap.write_map(save_ocmb_tsz_cib, ocmb_tsz_cib)
enmap.write_map(save_ocmb_tsz_ksz_cib, ocmb_tsz_ksz_cib)

enmap.write_map(save_bcmb, bcmb)
enmap.write_map(save_bcmb_ksz_cib, bcmb_ksz_cib)






elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)