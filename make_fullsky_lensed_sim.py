#%%
from pixell import enmap,lensing as plensing, curvedsky as cs, utils as u
import healpy as hp
import numpy as np
from enlib import bench
from orphics import io,cosmology,maps
import os,sys
#%%




wpath = "/data5/sims/websky/data/kap_lt4.5.fits"
kfile = "phi_alm.fits"
try:
    phi_alm = hp.read_alm(kfile)
    print("Read saved phi alm file...")
except:
    print("Making phi alm file...")
    kmap = hp.read_map(wpath)
    nside = hp.npix2nside(kmap.size)
    lmax = nside*3+1
    kalm = hp.map2alm(kmap,lmax=lmax)
    ells = np.arange(lmax+2)
    fls = 2/ells/(ells+1.)
    fls[ells<2] = 0
    phi_alm = hp.almxfl(kalm,fls)
    hp.write_alm(kfile,phi_alm,overwrite=True)



#for res in [0.25,None]:
for res in [0.25]:
    # if not(res is None): continue #!!!

    if res is None:
        sfact = 2
        temp = "/data5/sims/websky/dr5_clusters/model_MFMF_pass2_cmb-tsz-ksz-cib_f150.fits"
        footstr = "_actfoot"
    else:
        footstr = ""

    shape,wcs = enmap.band_geometry(np.asarray((-89,89))*u.degree,res=res*u.arcmin,proj='car')
    if not(res is None):
        pass
    else:
        oshape,owcs = enmap.read_map_geometry(temp)
        #shape,wcs = enmap.scale_geometry(oshape,owcs,sfact)
        #print("Scaling geometry...")
        #print(oshape)
        #print(shape)


    theory = cosmology.default_theory()

    seed = 1
    #shape = (3,)+shape
    lmax = 10000
    #ps = np.zeros((3,3,lmax))
    ells = np.arange(lmax)
    # ps[0,0] = theory.uCl('TT',ells)
    # ps[1,0] = theory.uCl('TE',ells)
    # ps[0,1] = theory.uCl('TE',ells)
    # ps[1,1] = theory.uCl('EE',ells)
    ps = theory.uCl('TT',ells)

    # if res is None:
    #     cmb_alm = hp.read_alm("/data5/sims/websky/data/unlensed_alm.fits")
    # else:
    #     cmb_alm = cs.rand_alm(ps)
    cmb_alm = hp.read_alm("/data5/sims/websky/data/unlensed_alm.fits")

    
    with bench.show("lens"):
        omaps = plensing.lens_map_curved(shape, wcs, phi_alm, cmb_alm, output="lu", verbose=True, delta_theta=None)

    enmap.write_map(f"lensed{footstr}.fits",omaps[0])
    enmap.write_map(f"unlensed{footstr}.fits",omaps[1])
    lmap = omaps[0]
    umap = omaps[1]



    # %%
    odlmap = enmap.downgrade(lmap,2)
    # %%
    odumap = enmap.downgrade(umap,2)
    # %%
    if res is None:
        dlmap = enmap.extract(odlmap,oshape,owcs)
        dumap = enmap.extract(odumap,oshape,owcs)
    else:
        dlmap = odlmap
        dumap = odumap
        
    enmap.write_map(f"dlensed{footstr}.fits",dlmap)
    enmap.write_map(f"dunlensed{footstr}.fits",dumap)
    # %%
    # %%
