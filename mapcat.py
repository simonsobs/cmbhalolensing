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
data_choice = cutils.data_choice
#cat_type = "wise_panstarrs"
#cat_type = "madcows_photz"
#cat_type = "sdss_redmapper"
cat_type = sys.argv[1]
meanfield = False

shape,wcs = enmap.fullsky_geometry(res=1 * utils.degree)
ras,decs,_ = cutils.catalog_interface(cat_type,is_meanfield=meanfield)

cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
cmap = maps.binary_mask(enmap.smooth_gauss(cmapper.counts,2 * utils.degree),1e-3)
io.hplot(cmap,'counts')


shape,wcs = enmap.read_map_geometry(paths.coadd_data + data_choice.hres_150)
omap = enmap.project(cmap,shape,wcs,order=0)
io.plot_img(omap,'pcounts')
enmap.write_map(f'{paths.data}{cat_type}_mask.fits',omap)