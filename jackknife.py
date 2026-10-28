import healpy as hp
import numpy as np
from orphics import io#,maps,cosmology,stats,catalogs
from pixell import enmap,utils,bunch
from somapy.maps import split_equal_area

import os,sys
import utils as cutils

""" 
Loads a mask (or map) and splits it into equal area.
Splits given catalogs into same regions and 
calculates the mean using jackknifing.
"""
mask_path = sys.argv[1]
stack_path = sys.argv[2]
mf_path = sys.argv[3]
n_regions = 25

paths = cutils.paths
defaults = bunch.Bunch(io.config_from_yaml("input/defaults.yml"))
data_choice = bunch.Bunch(io.config_from_yaml("input/data.yml"))

### get equal area regions from mask
mask = enmap.read_map(mask_path)
region_map = split_equal_area(mask, n_regions)
io.plot_img(region_map, f'{mask_path.split(".")[-2]}_regions.png')
enmap.write_map(f'{mask_path.split(".")[-2]}_regions.fits', region_map)

### load data
#TODO: just write this into utils.py postprocess eventually?
s_stack, shape_stack, wcs_stack, kmask, modrmap, bin_edges,data,profs = load_dumped_stats(stack_path,get_extra=True)
if mf_path != "":
    s_mf, shape_mf, wcs_mf = load_dumped_stats(mf_path)
#TODO: update load_dumped_stats to also get other saved info: coords, redshifts, etc.

#TODO: mask coords based on region numbers
#TODO: calculate mf-sub stack for each region
#TODO: get mean and jackknife errors

