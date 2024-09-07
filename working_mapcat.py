from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,utils
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
#cat_type = "wise_panstarrs"
#cat_type = "madcows_photz"
cat_type = sys.argv[1]
meanfield = False

# cat_type = "sdss_redmapper"
# meanfield = True

shape,wcs = enmap.fullsky_geometry(res=1 * utils.degree)
if cat_type == "websky_cmass":
    cat = "/home3/nehajo/projects/cmbhalolensing/data/sim_cats/websky_cmasslike_deccut.txt"
    data = np.load(cat, allow_pickle=True).item()
    ras = data['ra']
    decs = data['dec']
    zs = data['z']
    
elif cat_type == "agora_cmass":
    cat = "/home3/nehajo/projects/cmbhalolensing/data/sim_cats/agora_cmasslike_deccut.npy"
    data = np.load(cat, allow_pickle=True).item()
    ras = data['ra']
    decs = data['dec']
    zs = data['z']

else: ras,decs,zs,ws,data = cutils.catalog_interface(cat_type,is_meanfield=meanfield)
cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree),1e-3)
io.hplot(cmap,f'{paths.scratch}{cat_type}_counts')

shape,wcs = enmap.read_map_geometry(paths.act_data + f"../maps/dr5/act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits")
omap = enmap.project(cmap,shape,wcs,order=0)
io.plot_img(omap,f'{paths.scratch}{cat_type}_pcounts')
enmap.write_map(f'{paths.scratch}{cat_type}_mask.fits',omap)

