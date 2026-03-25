import numpy as np
from pixell import enmap, reproject, utils, curvedsky, bunch
from orphics import io
import healpy as hp
import argparse

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
sim_spec = bunch.Bunch(io.config_from_yaml("input/sim_info.yml"))
print("Paths: ", paths)

parser = argparse.ArgumentParser()
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal or agora."
)
args = parser.parse_args()

sim_name = args.which_sim
sim_path = paths[f"{sim_name}_sim_path"]
hp_sims = sim_spec[sim_name]
r_sims = paths[sim_name]
components = r_sims.keys()
print(f"components in {sim_name}: {components}")


### CAR settings
px = 0.5
shape, wcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj="car")

print(f"CAR maps:\n pixel size: {px}\n shape: {shape}\n wcs: {wcs}")

def file_to_map(fname, shape, wcs):
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


### REPROJECTING AND SAVING MAPS
for comp in components:

    hp_path = sim_path+hp_sims[comp]
    r_path = sim_path+r_sims[comp]

    print(f"::: reading and reprojecting {comp} map in {hp_path}")

    r_map = file_to_map(hp_path, shape, wcs)
    enmap.write_map(r_path, r_map)

    print(f"::: {comp} saved in {r_path}")