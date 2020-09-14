from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,utils
import numpy as np
import os,sys
import utils as cutils

"""
Given the mask saved from mapcat.py, generate and save
a random catalog.
"""


paths = cutils.paths
#cat_type = "wise_panstarrs"
#cat_type = "madcows_photz"
cat_type = args.sys[1]
Nx = int(args.sys[2])

ras,decs,_,_,_ = cutils.catalog_interface(cat_type,is_meanfield=False)
N = Nx * len(ras)

mask = enmap.read_map(f'{paths.scratch}{cat_type}_mask.fits')
shape,wcs = mask.shape,mask.wcs
Npix = mask[mask>0].size
inds = np.random.choice(Npix,size=N,replace=False)

pixs = enmap.pixmap(shape,wcs)

print(pixs.shape)

coords = mask.pix2sky(pixs[:,mask>0][:,inds]) / utils.degree
io.save_cols(paths.data+f"{cat_type}_randoms.txt",(coords[1],coords[0]))
print(coords.shape)
cmapper = catalogs.CatMapper(coords[1],coords[0],shape=shape,wcs=wcs)
io.hplot(enmap.downgrade(cmapper.counts,16),'randcounts',mask=0)

