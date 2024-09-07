from __future__ import print_function
from orphics import maps, io, cosmology, stats, catalogs
from pixell import enmap, utils, bunch
import numpy as np
import os, sys
import utils as cutils
from astropy.io import fits
import healpy as hp

"""
Given the mask saved from mapcat.py, generate and save
a random catalog.
"""


paths = cutils.paths
sim_paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))

cat_type = sys.argv[1] # halo / tsz / cmass
sim_type = sys.argv[2] # websky / sehgal / agora

nemo = False

try:
    sim_version = sys.argv[3] # nemo-sim-kit version name  
    output_path = f"{sim_paths.scratch}/{sim_version}/"
except: output_path = f"{sim_paths.scratch}/"

Nx = 10

try:
    # data catalogues are available through utils.py but not sim data catalogues yet
    # ras, decs, _,_,_ = cutils.catalog_interface(cat_type, is_meanfield=False)

    if sim_type == "websky": 
        cat = sim_paths.websky_tsz_cat
    elif sim_type == "sehgal":
        cat = sim_paths.sehgal_tsz_cat
    elif sim_type == "agora":
        cat = sim_paths.agora_tsz_cat
    hdu = fits.open(cat)
    ras = hdu[1].data["RADeg"]
    decs = hdu[1].data["DECDeg"]

except:
    if cat_type == "cmass":
        if sim_type == "agora":
            cat = "/home3/nehajo/projects/cmbhalolensing/data/sim_cats/agora_cmasslike_deccut.npy"
            data = np.load(cat, allow_pickle=True).item()
            ras = data['ra']
            decs = data['dec']
            zs = data['z']
        elif sim_type == "websky":
            cat = "/home3/nehajo/projects/cmbhalolensing/data/sim_cats/websky_cmasslike_deccut.npy"
            data = np.load(cat, allow_pickle=True).item()
            ras = data['ra']
            decs = data['dec']
            zs = data['z']
    else:
        cat = output_path + f"{sim_type}_halo.txt"
        ras, decs, _, _ = np.loadtxt(cat, unpack=True)

N = Nx * len(ras)
print(" ::: total number of sample = ", Nx, "x", len(ras))

if cat_type == "halo": 

    dec_min=-np.pi/2
    dec_max=np.pi/2
    ra_min=0
    ra_max=2.*np.pi

    coords = np.zeros((2,N))
    dmin = np.cos(np.pi/2 - dec_min)
    dmax = np.cos(np.pi/2 - dec_max)
    coords[0,:] = (np.pi/2 - np.arccos(np.random.uniform(dmin, dmax, N))) / utils.degree
    coords[1,:] = np.random.uniform(ra_min, ra_max, N) / utils.degree

    io.save_cols(output_path+f"{sim_type}_{cat_type}_{N}x_randoms.txt",(coords[1], coords[0]))

elif cat_type == "tsz" or cat_type == "cmass": 

    mask = enmap.read_map(f'{paths.scratch}{sim_type}_{cat_type}_mask.fits')
    # mask = enmap.read_map(f'{paths.data}mask/websky_act_mask.fits')
    shape, wcs = mask.shape, mask.wcs
    Npix = mask[mask>0].size
    inds = np.random.choice(Npix, size=N, replace=False)

    pixs = enmap.pixmap(shape,wcs)
    print(pixs.shape)

    coords = mask.pix2sky(pixs[:,mask>0][:,inds], safe=False) / utils.degree
    print(coords.shape)

    cmapper = catalogs.CatMapper(coords[1], coords[0], shape=shape, wcs=wcs)
    io.hplot(enmap.downgrade(cmapper.counts, 16), f'{paths.scratch}{sim_type}_{cat_type}_{Nx}x_randcounts', mask=0)

    print(coords.shape)
    print(coords[0].shape)
    print(coords[1].shape)
    io.save_cols(output_path+f"{sim_type}_{cat_type}_{Nx}x_randoms.txt",(coords[1], coords[0]))

print(" ::: min and max rand ra: ", coords[1].min(), coords[1].max())
print(" ::: min and max rand dec: ", coords[0].min(), coords[0].max())
print(coords.shape)