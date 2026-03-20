import time as t
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, wcsutils, bunch
from orphics import maps, io
import healpy as hp
from past.utils import old_div
import argparse

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
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
    "freq_sz", type=int, help="Set frequency for maps in GHz (90 or 150)."
)
parser.add_argument(
    "--flux_cut", type=float, help="If using flux cut, set max in mJy."
)
parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
)
parser.add_argument(
    "--planck_like", action="store_true", help="Set sim settings to be same as Planck SMICA."
)
args = parser.parse_args() 


output_path = paths.simsuite_path
sim_name = args.which_sim
if args.high_accuracy:
    sim_name += "_high_acc"
    print("high accuracy not currently doing anything special")
    ###TODO: high accuracy settings
sim_info = sim_spec[sim_name]
save_name = args.save_name
print(f" ::: producing maps for {sim_name.upper()} sim")
save_name = sim_name + "_" + save_name
sim_path = paths[f"{sim_name}_sim_path"]
 
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)


# SIM SETTING ------------------------------------------------------------------

print(f" ::: map frequency at {args.freq_sz} GHz")

px = 0.5
if args.planck_like:
    fwhm = 5.
    nlevel = 35.
else:
    nlevel = 15.
    if args.freq_sz == 90:
        fwhm = 2.2
    elif args.freq_sz == 150:
        fwhm = 1.5


# UNIT CONVERSIONS and FLUX CUT --------------------------------------------------------------------

tcmb = 2.726 #Kelvin
tcmb_uK = tcmb * 1e6 #micro-Kelvin
H_cgs = 6.62608e-27
K_cgs = 1.3806488e-16
kboltz = 1.3806503e-23 #MKS
hplanck = 6.626068e-34 #MKS
clight = 299792458.0 #MKS

# convert compton-y to delta-T (in uK) 
def fnu(nu):
    """
    nu in GHz
    tcmb in Kelvin
    """
    mu = H_cgs*(1e9*nu)/(K_cgs*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

def ItoDeltaT(nu):
    # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
    #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
    #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
    nu *= 1e9
    X = hplanck*nu/(kboltz*tcmb)
    dBnudT = (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/tcmb_uK * 1e26
    return 1./dBnudT

if args.flux_cut is not None:
    flux_cut_uK = ItoDeltaT(args.freq_sz) * args.flux_cut *1e-3 / (px*utils.arcmin)**2
    print(f" ::: map flux maximum set at {args.flux_cut} mJy, or {flux_cut_uK} uK")


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
r_tszmap = sim_path + paths[f"{sim_name}_tsz{args.freq_sz}_reproj"]
r_kszmap = sim_path + paths[f"{sim_name}_ksz_reproj"]
r_cib093map = sim_path + paths[f"{sim_name}_cib093_reproj"]
r_cib145map = sim_path + paths[f"{sim_name}_cib145_reproj"]

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

    print(" ::: reading ymap:", ifile) 
    print("ymap", ymap.shape) 

    tszmap = fnu(args.freq_sz) * ymap * tcmb_uK
    print(" ::: converting ymap to tsz map at %d GHz" %args.freq_sz)
    enmap.write_map(r_tszmap, tszmap)

if args.flux_cut is not None:
    print(" ::: tszmap pixels above cut:", np.count_nonzero(tszmap>flux_cut_uK))
    tszmap[tszmap>flux_cut_uK] = flux_cut_uK

print("tszmap", tszmap.shape)

# reading ksz map
try:
    kszmap = enmap.read_map(r_kszmap, delayed=False)
    print(" ::: reading reprojected ksz map:", r_kszmap)

except:
    ifile = sim_path + sim_info['ksz']
    kszmap = file_to_map(ifile, shape, wcs)

    enmap.write_map(r_kszmap, kszmap)
    print(" ::: reading and reprojecting ksz map:", ifile) 

print("kszmap", kszmap.shape)
if args.flux_cut is not None:
    print(" ::: kszmap pixels above cut:", np.count_nonzero(kszmap>flux_cut_uK))
    kszmap[kszmap>flux_cut_uK] = flux_cut_uK

# reading cib map
try:
    if args.freq_sz == 90: r_cibmap = r_cib093map
    elif args.freq_sz == 150: r_cibmap = r_cib145map

    cibmap = enmap.read_map(r_cibmap, delayed=False)
    print(" ::: reading reprojected cib map:", r_cibmap)

except:
    # frequency for CIB
    if args.freq_sz == 90: ifile = sim_path + sim_info['cib_nu093']
    elif args.freq_sz == 150: ifile = sim_path + sim_info['cib_nu145']

    cibmap_i = file_to_map(ifile, shape, wcs)
    
    if sim_name == "websky":
        cibmap_i*=1e6 # MJy/sr -> Jy/sr


    print(" ::: reading cib map:", ifile)
    print("cibmap", np.shape(cibmap_i))

    # frequency for CIB 
    if args.freq_sz == 150: freq_cib = 145
    elif args.freq_sz == 90: freq_cib = 93

    cibmap = ItoDeltaT(freq_cib) * cibmap_i 

    print(" ::: converting cib map at %d GHz" %freq_cib)
    enmap.write_map(r_cibmap, cibmap)

if args.flux_cut is not None:
    print(" ::: cibmap pixels above cut:", np.count_nonzero(cibmap>flux_cut_uK))
    cibmap[cibmap>flux_cut_uK] = flux_cut_uK

print("cibmap", cibmap.shape)

# check and make signal/observed maps

def make_save_maps(lmap, fgmaps = None, fgnames = None, save_beam = False):
    smap = lmap
    map_names = ""

    if fgmaps:
        for fgmap, name in zip(fgmaps, fgnames):
            smap += fgmap
            map_names += f"_{name}"

    bmap = apply_beam(smap)

    white_noise = maps.white_noise(smap.shape, smap.wcs, noise_muK_arcmin=nlevel)
    omap = bmap + white_noise

    if save_beam:
        enmap.write_map(f"{save_dir}bcmb{map_names}.fits", bmap)

    enmap.write_map(f"{save_dir}scmb{map_names}.fits", smap)
    enmap.write_map(f"{save_dir}ocmb{map_names}.fits", omap)


assert wcsutils.equal(kmap.wcs, lmap.wcs)  
assert wcsutils.equal(kmap.wcs, tszmap.wcs)

make_save_maps(lmap, save_beam = True)
make_save_maps(lmap, fgmaps = [tszmap], fgnames = ['tsz'])


assert wcsutils.equal(kmap.wcs, kszmap.wcs)
assert wcsutils.equal(kmap.wcs, cibmap.wcs)

make_save_maps(lmap, fgmaps = [kszmap + cibmap], fgnames = ['ksz_cib'], save_beam = True)
make_save_maps(lmap, fgmaps = [tszmap, kszmap + cibmap], fgnames = ['tsz', 'ksz_cib'])

elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)