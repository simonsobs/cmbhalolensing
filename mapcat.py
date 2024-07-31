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
if cat_type == "cmass":
    cat = "/data5/sims/websky/cmass_hod/websky_cmass_galaxy_catalog_full_sky.txt"
    ras, decs, zs = np.loadtxt(cat, unpack=True)
    dec_cut = np.where(np.logical_and(decs<10, decs>-10))
    ras = ras[dec_cut]
    decs = decs[dec_cut]
else: ras,decs,zs,ws,data = cutils.catalog_interface(cat_type,is_meanfield=meanfield)
cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree),1e-3)
io.hplot(cmap,'counts')


shape,wcs = enmap.read_map_geometry(paths.coadd_data + f"act_planck_dr5.01_s08s18_AA_f090_night_map_srcfree.fits")
omap = enmap.project(cmap,shape,wcs,order=0)
io.plot_img(omap,'pcounts')
enmap.write_map(f'{paths.scratch}{cat_type}_mask.fits',omap)
