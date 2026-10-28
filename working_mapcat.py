from __future__ import print_function
import healpy as hp
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,utils,bunch
import numpy as np
import os,sys
import utils as cutils
# import recon_sim as sim

"""
Loads a catalog
Maps it
Smooths it
Thresholds it
Projects it onto ACT
This gives a mask of 1s and 0s from which a random catalog can be made
"""


paths = cutils.paths
defaults = bunch.Bunch(io.config_from_yaml("input/defaults.yml"))
data_choice = bunch.Bunch(io.config_from_yaml("input/data.yml"))

#cat_type = "wise_panstarrs"
#cat_type = "madcows_photz"
cat_type = sys.argv[1]
meanfield = True

# cat_type = "sdss_redmapper"
# meanfield = True

io.mkdir(paths.scratch)
shape,wcs = enmap.fullsky_geometry(res=1 * utils.degree)
if cat_type == "websky_cmass":
    cat = "/home3/nehajo/projects/cmbhalolensing/data/sim_cats/websky_cmasslike_deccut.npy"
    data = np.load(cat, allow_pickle=True).item()
    ras = data['ra']
    decs = data['dec']
    zs = data['z']
    
elif cat_type == "agora_cmass":
    cat = "/data7/nehajo/CMASS/sim_cats/agora_lensed_cmasslike_deccut.npy"
    data = np.load(cat, allow_pickle=True).item()
    ras = data['ra']
    decs = data['dec']
    zs = data['z']

else: ras,decs,zs,ws,data = cutils.catalog_interface(cat_type,is_meanfield=meanfield)
cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree))
io.plot_img(cmap,f'{paths.scratch}{cat_type}_cat_mask.png')


### get act mask
act_mask = enmap.read_map(paths.act_data + "DR6_lensing/masks/act_mask_20220316_GAL060_rms_70.00_d2sk.fits") 
io.plot_img(act_mask, f'{paths.scratch}act_mask.png')

### project catalog mask to ACT geometry
shape, wcs = act_mask.shape, act_mask.wcs
omap = enmap.project(cmap,shape,wcs,order=1)

### get intersection of catalog and ACT masks
imask = np.logical_and(omap, act_mask)
imask = imask.astype(np.int8)
imap = enmap.enmap(imask, wcs=wcs)

io.plot_img(imap, f'{paths.scratch}{cat_type}_act_mask.png')
enmap.write_map(f'{paths.scratch}{cat_type}_act_mask.fits', imap)

