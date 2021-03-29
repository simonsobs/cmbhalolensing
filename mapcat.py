from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,utils
import numpy as np
import os,sys
import utils as cutils

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
ras,decs,_,_,_ = cutils.catalog_interface(cat_type,is_meanfield=meanfield)
cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree),1e-3)
io.hplot(cmap,f'counts_{cat_type}')


shape,wcs = enmap.read_map_geometry(paths.coadd_data + f"act_planck_s08_s18_cmb_f150_daynight_srcfree_map.fits")
omap = enmap.project(cmap,shape,wcs,order=0)
io.plot_img(omap,f'pcounts_{cat_type}')
enmap.write_map(f'{paths.scratch}{cat_type}_mask.fits',omap)
