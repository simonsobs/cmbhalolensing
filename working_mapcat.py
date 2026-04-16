from __future__ import print_function
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
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree),1e-3)
io.hplot(cmap,f'{paths.scratch}{cat_type}_cat_counts')

### get ivar map to make binary mask and grow mask to account for stamp radius
ivar = enmap.read_map(paths.act_data + data_choice.hres_ivar)
ivar_mask = maps.get_masked_ivar(ivar, 
                                 grow_arcmin=defaults.stamp_width_arcmin/2)
io.plot_img(ivar_mask,f'{paths.scratch}{cat_type}_ivar')

### project catalog mask to ACT geometry
shape, wcs = ivar.shape, ivar.wcs
omap = enmap.project(cmap,shape,wcs,order=0)
io.plot_img(omap,f'{paths.scratch}{cat_type}_counts')

### get intersection of catalog and ACT masks
imask = np.logical_and(omap, ivar_mask)
imask = imask.astype(np.int8)
imap = enmap.enmap(imask, wcs=wcs)
io.plot_img(imap, f'{paths.scratch}{cat_type}_mask')
enmap.write_map(f'{paths.scratch}{cat_type}_mask.fits', imap)

