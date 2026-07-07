import time as t
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


# RUN SETTING --------------------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "save_name", type=str, help="Name you want for your output."
)
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal or agora."
)
parser.add_argument(
    "footprint", type=str, help="Choose the footprint e.g. full or act. The fullsky map is needed for halo sample."
)


args = parser.parse_args() 

output_path = paths.simsuite_path
save_name = args.save_name
if args.which_sim == "agora": 
    print(" ::: producing maps for AGORA sim")
    save_name = "agora_" + save_name
    sim_path = paths.agora_sim_path    
save_dir = f"{output_path}/{save_name}/"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir) 


# SIM SETTING --------------------------------------------------------------------------------

px = 0.5

fwhm090 = 2.2
fwhm150 = 1.5
fwhm_plc = 5.0

nlevel = 15.0  
nlevel_plc = 35.0 

# flux per pixel [mJy] cut for CIB map
fluxcut_val = 5.0


# BEAM CONVOLUTION ---------------------------------------------------------------------------

def apply_beam(imap, ifwhm): 
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    if args.which_sim == "websky" or args.which_sim == "agora": nside = 8192
    elif args.which_sim == "sehgal": nside = 4096
    alm_lmax = nside * 3
    bfunc = lambda x: maps.gauss_beam(ifwhm, x)  
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convolved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convolved_alm, enmap.empty(imap.shape, imap.wcs))


# PREPARING MAP GEOMETRY ---------------------------------------------------------------------

if args.footprint == "full":
    shape, wcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj="car")
    print(" ::: full sky geometry")

elif args.footprint == "act":
    ifile = f"{paths.mat_path}/dlensed_actfoot.fits"  
    imap = enmap.read_map(ifile)
    print(" ::: ACT footprint geometry") 
    shape, wcs = imap.shape, imap.wcs


# PREPARING TRUE KAPPA -----------------------------------------------------------------------

if args.which_sim == "agora":
    ifile = f"{sim_path}{paths.agora_true_kappa}"
    hmap  = hp.read_alm(ifile).astype(np.complex128)
    kmap = curvedsky.alm2map(hmap, enmap.empty(shape, wcs, dtype=np.float64))

    print(" ::: reading and reprojecting true kappa map:", ifile) 
    print("kmap", kmap.shape)


# PREPARING LENSED CMB MAP -------------------------------------------------------------------

if args.which_sim == "agora":
    ifile = f"{sim_path}{paths.agora_lensed_cmb}"
    hmap,_,_ = hp.read_map(ifile, field=[0,1,2]).astype(np.float64)
    lmap = reproject.healpix2map(hmap, shape, wcs)

    print(" ::: reading and reprojecting lensed cmb map:", ifile) 
    print("lmap", lmap.shape) 


# PREPARING TSZ MAP (UNLENSED) ---------------------------------------------------------------

if args.which_sim == "agora":
    ifile = f"{sim_path}{paths.agora_tsz}"
    hmap = hp.read_map(ifile).astype(np.float64)
    ymap = reproject.healpix2map(hmap, shape, wcs)

    print(" ::: reading and reprojecting ymap:", ifile) 
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

def y_to_tsz(y_map, freq):
    # convert compton-y map to tSZ temperature map
    print(f" ::: converting ymap to tsz map at {freq:.0f} GHz")
    return fnu(freq) * y_map * tcmb_uK

tszmap090 = y_to_tsz(ymap, 90.0)    
tszmap150 = y_to_tsz(ymap, 150.0)


# PREPARING KSZ MAP (UNLENSED) ---------------------------------------------------------------

if args.which_sim == "agora":
    ifile = f"{sim_path}{paths.agora_ksz}" 
    hmap = hp.read_map(ifile).astype(np.float64)
    kszmap = reproject.healpix2map(hmap, shape, wcs)

    print(" ::: reading and reprojecting ksz map:", ifile) 
    print("kszmap", kszmap.shape) 



# PREPARING CIB MAP (LENSED; can't find the unlensed map T_T) ---------------------------------
# flux cut routine is copied from Karen's script for now! 

if args.which_sim == "agora":

    def flux_density_to_temp(freq_GHz):
        # get factor for converting delta flux density in [MJy/sr] to delta T in CMB units [uK]
        freq = float(freq_GHz)
        x = freq / 56.8
        return (1.05e3 * (np.exp(x)-1)**2 *
                np.exp(-x) * (freq / 100)**-4)

    def fluxcut_to_cib_map(freq, fluxcut_mJy):

        if freq == 90:
            ifile = f"{sim_path}{paths.agora_cib090}" # in uK 
        elif freq == 150:
            ifile = f"{sim_path}{paths.agora_cib150}" # in uK 

        hmap = hp.read_map(ifile).astype(np.float64)
        print(f"\n{freq} GHz")
        print("before cut:", hmap.min(), hmap.max(), hmap.mean())

        nside = hp.get_nside(hmap)                      # extract the HEALPix resolution 
        pixel_solid_angle = hp.nside2pixarea(nside)     # in steradian 
        lim = fluxcut_mJy * 1e-9 / pixel_solid_angle    # [mJy] to [MJy/sr]
        uK_lim = flux_density_to_temp(freq) * lim       # [MJy/sr] to [uK] 

        print(f"conversion factor: {flux_density_to_temp(freq):.2f}")
        print(f"uK threshold: {uK_lim:.2f}")
        print(f"number of pixels above threshold: {np.sum(hmap > uK_lim)} ({np.sum(hmap > uK_lim)/np.sum(hmap)*100:.6f} percent)")
        print(f"implementing flux cut on map of {fluxcut_mJy} mJy per pixel")

        hmap[hmap > uK_lim]=0.
        print("after cut:", hmap.min(), hmap.max(), hmap.mean())

        return reproject.healpix2map(hmap, shape, wcs)

    cibmap090 = fluxcut_to_cib_map(90, fluxcut_val)
    cibmap150 = fluxcut_to_cib_map(150, fluxcut_val)


for m in (
    kmap,
    tszmap090,
    tszmap150,
    kszmap,
    cibmap090,
    cibmap150,
):
    assert wcsutils.equal(lmap.wcs, m.wcs)


# ADDING MAPS --------------------------------------------------------------------------------

signal_maps = {
    "cmb": lmap,
    "cmb_ksz": lmap + kszmap,
    "cmb_cib090": lmap + cibmap090,
    "cmb_cib150": lmap + cibmap150,
    "cmb_ksz_cib090": lmap + kszmap + cibmap090,
    "cmb_ksz_cib150": lmap + kszmap + cibmap150,
    "cmb_tsz090": lmap + tszmap090,
    "cmb_tsz150": lmap + tszmap150,
    "cmb_tsz090_ksz": lmap + tszmap090 + kszmap,
    "cmb_tsz150_ksz": lmap + tszmap150 + kszmap,
    "cmb_tsz090_cib": lmap + tszmap090 + cibmap090,
    "cmb_tsz150_cib": lmap + tszmap150 + cibmap150,
    "cmb_tsz090_ksz_cib": lmap + tszmap090 + kszmap + cibmap090,
    "cmb_tsz150_ksz_cib": lmap + tszmap150 + kszmap + cibmap150,
}

print(" ::: SIGNAL only maps are ready!")



# APPLYING BEAM AND ADDING NOISE -------------------------------------------------------------

def make_observed_map_for_recon(smap, fwhm, noise_level, seed):
    return (
        apply_beam(smap, fwhm)
        + maps.white_noise(
            shape, wcs,
            noise_muK_arcmin=noise_level,
            seed=seed,
        )
    )

obs_map_details = {
    "g_ocmb":                ("cmb",                fwhm_plc, nlevel_plc, 100),
    "g_ocmb_ksz":            ("cmb_ksz",            fwhm_plc, nlevel_plc, 100),
    "g_ocmb_cib":            ("cmb_cib150",         fwhm_plc, nlevel_plc, 100),
    "g_ocmb_ksz_cib":        ("cmb_ksz_cib150",     fwhm_plc, nlevel_plc, 100),
    "h_ocmb":                ("cmb",                fwhm150,  nlevel,     101),
    "h_ocmb_ksz":            ("cmb_ksz",            fwhm150,  nlevel,     101),
    "h_ocmb_cib090":         ("cmb_cib090",         fwhm090,  nlevel,     102),
    "h_ocmb_cib150":         ("cmb_cib150",         fwhm150,  nlevel,     103),
    "h_ocmb_ksz_cib090":     ("cmb_ksz_cib090",     fwhm090,  nlevel,     102),
    "h_ocmb_ksz_cib150":     ("cmb_ksz_cib150",     fwhm150,  nlevel,     103),
    "h_ocmb_tsz090":         ("cmb_tsz090",         fwhm090,  nlevel,     102),
    "h_ocmb_tsz150":         ("cmb_tsz150",         fwhm150,  nlevel,     103),
    "h_ocmb_tsz090_ksz":     ("cmb_tsz090_ksz",     fwhm090,  nlevel,     102),
    "h_ocmb_tsz150_ksz":     ("cmb_tsz150_ksz",     fwhm150,  nlevel,     103),
    "h_ocmb_tsz090_cib":     ("cmb_tsz090_cib",     fwhm090,  nlevel,     102),
    "h_ocmb_tsz150_cib":     ("cmb_tsz150_cib",     fwhm150,  nlevel,     103),
    "h_ocmb_tsz090_ksz_cib": ("cmb_tsz090_ksz_cib", fwhm090,  nlevel,     102),
    "h_ocmb_tsz150_ksz_cib": ("cmb_tsz150_ksz_cib", fwhm150,  nlevel,     103),
}

omaps = {
    name: make_observed_map_for_recon(
        signal_maps[signal_name],
        fwhm,
        noise_level,
        seed,
    )
    for name, (signal_name, fwhm, noise_level, seed)
    in obs_map_details.items()
}

print(" ::: applying corresponding beams and adding flatwhite noise")
print(" ::: OBSERVED maps for lensing reconstruction are ready!")


if args.footprint == "act":

    ivar090 = enmap.read_map(paths.ivar090) 
    ivar150 = enmap.read_map(paths.ivar150) 

    def make_noise(ivar, seed):
        noise = maps.modulated_noise_map(ivar, seed=seed)[0]
        noise = np.nan_to_num(noise, posinf=0, neginf=0)
        assert np.all(np.isfinite(noise))
        return noise

    def make_observed_map(sky_map, fwhm, noise_map):
        return apply_beam(sky_map, fwhm) + noise_map

    freq_info = {
        "090": (fwhm090, make_noise(ivar090, seed=1)),
        "150": (fwhm150, make_noise(ivar150, seed=0)),
    }

    dr5white_omap_names = [
        "cmb_tsz090",
        "cmb_tsz150",
        "cmb_tsz090_ksz",
        "cmb_tsz150_ksz",
        "cmb_tsz090_cib",
        "cmb_tsz150_cib",
        "cmb_tsz090_ksz_cib",
        "cmb_tsz150_ksz_cib",
    ]

    dr5white_omaps = {
        f"h_ocmb{name[3:]}": make_observed_map(
            signal_maps[name],
            *freq_info["090" if "090" in name else "150"],
        )
        for name in dr5white_omap_names
    }

    print(" ::: applying corresponding beams and adding dr5white noise")
    print(" ::: OBSERVED maps for model subtraction are ready!")



# SAVING MAPS --------------------------------------------------------------------------------

suffix = "_fullsky" if args.footprint == "full" else ""

enmap.write_map(f"{save_dir}/true_kappa{suffix}.fits", kmap)

for name, omap in omaps.items():
    enmap.write_map(f"{save_dir}/{name}{suffix}.fits", omap)

if args.footprint == "act":
    for name, dr5white_omap in dr5white_omaps.items():
        enmap.write_map(f"{save_dir}/{name}{suffix}_dr5white.fits", dr5white_omap)

print(" ::: all maps are saved! yayyyyyy")



elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)