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

cat_type = sys.argv[1] # halo / tsz / cmass_like / cmass_dr12
sim_type = sys.argv[2] # websky / sehgal / agora / none
Nx = sys.argv[3] # integer or none
meanfield = True
nemo = False

try:
    sim_version = sys.argv[4] # nemo-sim-kit version name  
    output_path = f"{sim_paths.scratch}/{sim_version}/"
except: output_path = f"{paths.scratch}/"

if cat_type == "tsz":
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


elif cat_type == "cmass_like":
    if sim_type == "agora":
        cat = "/data7/nehajo/CMASS/sim_cats/agora_lensed_cmasslike_deccut.npy"
        data = np.load(cat, allow_pickle=True).item()
        ras = data['ra']
        decs = data['dec']
        zs = data['z']
    elif sim_type == "websky":
        cat = "/data7/nehajo/CMASS/sim_cats/websky_cmasslike_deccut.npy"
        data = np.load(cat, allow_pickle=True).item()
        ras = data['ra']
        decs = data['dec']
        zs = data['z']

elif cat_type == "cmass_dr12":
    ras,decs,zs,ws,data = cutils.catalog_interface(cat_type,is_meanfield=meanfield)
else:
    cat = output_path + f"{sim_type}_halo.txt"
    ras, decs, _, _ = np.loadtxt(cat, unpack=True)

if Nx != "none":
    N = Nx * len(ras)
else: N = len(ras)
print(" ::: total number of sample = ", N)

def select_based_on_mask(ras,decs,mask,threshold=0.99,ws=None, zs=None):
    """
    Filters ra,dec based on whether it falls within a mask
    """
    shape, wcs = mask.shape, mask.wcs

    print(ras.min(), ras.max(), decs.min(), decs.max())
    coords = np.stack([decs, ras]) * utils.degree
    pixs = enmap.sky2pix(shape,wcs,coords).astype(int)

    # First select those that fall within geometry
    Ny,Nx = shape
    sel = np.logical_and.reduce([pixs[0]>=0,pixs[1]>=0,pixs[0]<Ny,pixs[1]<Nx])
    print("selected", sel.shape)
    print(ras[sel].shape)
    pixs = pixs[:,sel]
    print("pixs", pixs.shape)
    ras = ras[sel]
    decs = decs[sel]
    zs = zs[sel]
    ws = ws[sel]

    # Select only inds above threshold
    sel2 = mask[pixs[0],pixs[1]]>threshold
    ras = ras[sel2]
    decs = decs[sel2]
    zs = zs[sel2]
    ws = ws[sel2]
    return ras, decs, ws, zs

def randoms_from_mask(N, mask):
    shape, wcs = mask.shape, mask.wcs
    Npix = mask[mask>0].size
    inds = np.random.choice(Npix, size=N, replace=False)

    pixs = enmap.pixmap(shape,wcs)
    print(pixs.shape)

    coords = mask.pix2sky(pixs[:,mask>0][:,inds], safe=False) / utils.degree
    print(coords.shape)
    return coords

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

elif cat_type == "tsz" or cat_type == "cmass_like": 

    mask = enmap.read_map(f'{paths.scratch}{sim_type}_{cat_type}_mask.fits')
    # mask = enmap.read_map(f'{paths.data}mask/websky_act_mask.fits')
    shape, wcs = mask.shape, mask.wcs
    coords = randoms_from_mask(N, mask)
    cmapper = catalogs.CatMapper(coords[1], coords[0], shape=shape, wcs=wcs)
    io.hplot(enmap.downgrade(cmapper.counts, 16), f'{paths.scratch}{sim_type}_{cat_type}_{Nx}x_randcounts', mask=0)

    print(coords.shape)
    print(coords[0].shape)
    print(coords[1].shape)
    io.save_cols(output_path+f"{sim_type}_{cat_type}_{Nx}x_randoms.txt",(coords[1], coords[0]))

elif cat_type == "cmass_dr12":
    # use given random catalogs instead of generating randoms
    # mask made from mapcat.py -- either act mask or intersect with catalog mask
    mask = enmap.read_map(f'{paths.scratch}{cat_type}_mask.fits', delayed=False)[0]
    shape, wcs = mask.shape, mask.wcs
    ras, decs, ws, zs = select_based_on_mask(ras, decs, mask, ws=ws, zs=zs)

    if Nx == "none":
        io.save_cols(output_path+f"{cat_type}_all_randoms.txt", (ras, decs, ws, zs))
    
    else:
        if N > len(ras):
            decs, ras = randoms_from_mask(N, mask)
            cmapper = catalogs.CatMapper(decs, ras, shape=shape, wcs=wcs)
            io.hplot(enmap.downgrade(cmapper.counts, 16), f'{paths.scratch}{cat_type}_{Nx}x_randcounts', mask=0)

        io.save_cols(output_path+f"{cat_type}_Nx_randoms.txt", (ras, decs, zs, ws))
        
    coords = np.array([decs, ras])
print(" ::: min and max rand ra: ", coords[1].min(), coords[1].max())
print(" ::: min and max rand dec: ", coords[0].min(), coords[0].max())
print(coords.shape)