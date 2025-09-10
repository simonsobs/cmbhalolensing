from pixell import enmap,utils as u,lensing as plensing,bench,bunch,wcsutils,curvedsky as cs
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys, shutil, os, warnings
import healpy as hp


daynight = "daynight"
apod_deg = 1.0
lmax = 24000



mask = enmap.read_map("/data5/act/masks/dr6v4_lensing_20240919b_masks/baseline/dr6v4_lensing_20240919_night_enhanced_mask_80.fits")
mask = maps.cosine_apodize(mask,apod_deg)
enmap.write_map('mask.fits',mask)
print("w2 (fsky): ",maps.wfactor(2,mask))


with bench.show("srcmask"):
    shape,wcs = mask.shape,mask.wcs
    rra,rdec = np.loadtxt("/data5/act/catalogs/inpaint_catalogs_20241002/union_catalog_regular_20241002.csv",delimiter=',',unpack=True)
    lra,ldec = np.loadtxt("/data5/act/catalogs/inpaint_catalogs_20241002/union_catalog_large_20241002.csv",delimiter=',',unpack=True)

    smask = (maps.mask_srcs(shape,wcs,(rdec,rra),6.0) * maps.mask_srcs(shape,wcs,(ldec,lra),10.0)).astype(bool)


# Final bmask
bmask = mask*0 + 1
bmask[mask<0.99] = 0

bmask = bmask * smask
enmap.write_map('bmask.fits',bmask)

for freq in ['f090','f150','f220']:
    print(freq)
    if freq!='f220' : continue
    
    fseason = "s08" if freq!='f220' else "s17"
    mapfunc = lambda freq: f"/data5/act/coadds/20240323_simple/act_planck_{fseason}_s22_{freq}_{daynight}_map_srcfree.fits"
    ivarfunc = lambda freq: f"/data5/act/coadds/20240323_simple/act_planck_{fseason}_s22_{freq}_{daynight}_ivar.fits"
    cmapfunc = lambda freq: f"/data5/act/iabril/nemo/candidatesModelMap_{freq}_SNR5.fits"
    
    # cluster subtract
    with bench.show("readmaps"):
        clmap = enmap.read_map(cmapfunc(freq)) if freq!='f220' else 0.
        imap = enmap.read_map(mapfunc(freq),sel=np.s_[0])  - clmap
        if not(np.all(np.isfinite(imap))):
            print("WARNING: ",imap[~np.isfinite(imap)].size * 100./imap.size, "% bad pixels.")
            imap[~np.isfinite(imap)] = 0
            

    with bench.show("read ivars"):
        ivar = enmap.read_map(ivarfunc(freq),sel=np.s_[0])
    rms = maps.rms_from_ivar(ivar,safe=False)
    allrms = rms[np.logical_and(ivar>0,bmask==1)]
    print(f"{freq}, {allrms.mean():.1f} +- {allrms.std():.1f} uK-arcmin")
    io.hist(allrms,bins=np.linspace(3.,80.,120),save_file=f'hist_rms_{freq}.png',verbose=True)


    # inpaint
    with bench.show("gapfill"):
        omap = maps.gapfill_edge_conv_flat(imap, ~smask, ivar=ivar)

    if not(np.all(np.isfinite(omap))):
        print("WARNING: ",omap[~np.isfinite(omap)].size * 100./omap.size, "% bad pixels.")
        print("Npix: ", omap[~np.isfinite(omap)].size)
        omap[~np.isfinite(omap)] = 0
        
    # if not(np.all(np.isfinite(omap))): raise ValueError
    enmap.write_map(f'inpainted_subtracted_{freq}.fits',omap)
    omap = omap * mask
    io.hplot(omap,f'omap_{freq}',downgrade=2)

    
    # alm transform
    with bench.show("alm"):
        alm = cs.map2alm(omap,lmax=lmax)

    if not(all(np.isfinite(alm))): raise ValueError
    hp.write_alm(f'inpainted_subtracted_alm_{freq}.fits',alm,overwrite=True)
        


