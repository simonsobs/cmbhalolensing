from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs,mpi
from pixell import enmap,reproject,pointsrcs,utils
import numpy as np
import os,sys
from tilec import fg as tfg
from enlib import bunch,bench
import ptfit

zmin = 0.1
snmin = 20.
target_fwhm = 2.2

df = catalogs.df_from_fits('confirmed_catalog.fits',['SNR','redshift','name','RADeg','decDeg'],['sn','z','name','ra','dec'])
df = df.drop(df[df.z < zmin].index)
df = df.drop(df[df.sn < snmin].index)
df.sort_values(by=['sn'],inplace=True,ascending=False,ignore_index=True)

width_arcmin = 20.
res_arcmin = 0.5

njobs = len(df)
comm,rank,my_tasks = mpi.distribute(njobs)

map90 = "/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f090_daynight_srcfree_map.fits"
div90 = "/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f090_daynight_ivar.fits"
map150 = "/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f150_daynight_srcfree_map.fits"
div150 = "/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f150_daynight_ivar.fits"

theory = cosmology.default_theory()
ells = np.arange(10000)
ps = theory.lCl('TT',ells)[None,None]

for task in my_tasks:
    sn = df.sn[task]
    print(task,sn)
    ra = df.ra[task] * utils.degree
    dec = df.dec[task] * utils.degree

    map90_sel = reproject.cutout(map90, width=np.deg2rad(width_arcmin/60.), ra=ra, dec=dec,
                                 res=np.deg2rad(res_arcmin/60.), return_slice=True,sindex=0)
    if map90_sel is None: continue
    div90_sel = reproject.cutout(div90, width=np.deg2rad(width_arcmin/60.), ra=ra, dec=dec,
                                 res=np.deg2rad(res_arcmin/60.), return_slice=True,sindex=0)
    map150_sel = reproject.cutout(map150, width=np.deg2rad(width_arcmin/60.), ra=ra, dec=dec,
                                 res=np.deg2rad(res_arcmin/60.), return_slice=True,sindex=0)
    div150_sel = reproject.cutout(div150, width=np.deg2rad(width_arcmin/60.), ra=ra, dec=dec,
                                 res=np.deg2rad(res_arcmin/60.), return_slice=True,sindex=0)

    s90 = enmap.read_map(map90,sel=map90_sel)
    s150 = enmap.read_map(map150,sel=map150_sel)

    sel = maps.crop_center(s90,1,sel=True) 
    s90[sel] = np.nan
    io.plot_img(s90,f"m90_{df.name[task].decode().replace(' ','_')}_snr_{sn:.1f}.png",cmap='gray',lim=300)
    sel = maps.crop_center(s150,1,sel=True) 
    s150[sel] = np.nan
    io.plot_img(s150,f"m150_{df.name[task].decode().replace(' ','_')}_snr_{sn:.1f}.png",cmap='gray',lim=300)
    continue

    i90 = enmap.read_map(div90,sel=div90_sel)
    i150 = enmap.read_map(div150,sel=div150_sel)


    shape,wcs = s90.shape,s90.wcs
    tap_per = 12.0
    pad_per = 3.0
    frac = 1. - ((tap_per + pad_per ) / 100.)
    taper,_ = maps.get_taper(shape,wcs,taper_percent = tap_per,pad_percent = pad_per,weight=None)
    modlmap = enmap.modlmap(shape,wcs)
    target_beam = maps.gauss_beam(modlmap,target_fwhm)
    beam_ratio = target_beam / maps.gauss_beam(modlmap,1.4)
    r150 = maps.filter_map(s150*taper,beam_ratio)
    beam_ratio = target_beam / maps.gauss_beam(modlmap,2.2)
    r90 = maps.filter_map(s90 * taper,beam_ratio)
    
    dmap = maps.get_central(r150 - r90,frac)
    div = maps.get_central(1./(1./i150 + 1./i90),frac)

    io.plot_img(dmap,f"ymap_{df.name[task].decode().replace(' ','_')}_snr_{sn:.1f}.png",cmap='gray',lim=300)

    beam = np.deg2rad(maps.sigma_from_fwhm(target_fwhm))
    pfitter = ptfit.Pfit(dmap.shape,dmap.wcs,beam=beam,div=div,ps=ps)

    chi = np.inf
    sim = None
    for k,f in enumerate([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]):
        fwhm2 = f/60.
        beam2 = np.deg2rad(maps.sigma_from_fwhm(fwhm2))
        with bench.show("fit"):
            pflux,cov,isim,chisquare = pfitter.fit(dmap if k==0 else None,dec,ra,beam2)
        print(f,chisquare)
        if chisquare>chi: break
        chi = chisquare
        sim = isim
    if sim is None: raise ValueError
        

    io.plot_img(sim,f"ysim_{df.name[task].decode().replace(' ','_')}_snr_{sn:.1f}.png",lim=300,verbose=False,cmap='gray')
    io.plot_img(dmap-sim,f"ydiff_{df.name[task].decode().replace(' ','_')}_snr_{sn:.1f}.png",lim=300,verbose=False,cmap='gray')



sys.exit()
fwhm = 1.4/60.
beam = np.deg2rad(maps.sigma_from_fwhm(fwhm))



# pflux,cov,fit = ptfit.ptsrc_fit(stamp,np.deg2rad(dec),np.deg2rad(ra),beam=beam,rbeam=beam2,div=div,ps=ps)

with bench.show("init"):
    pfitter = ptfit.Pfit(stamp.shape,stamp.wcs,beam=beam,div=div,ps=ps)

for k,f in enumerate([1.0,2.0,3.0,4.0,5.0,6.0,7.0]):
    fwhm2 = f/60.
    beam2 = np.deg2rad(maps.sigma_from_fwhm(fwhm2))
    with bench.show("fit"):
        pflux,cov,fit,chisquare = pfitter.fit(stamp if k==0 else None,np.deg2rad(dec),np.deg2rad(ra),beam2)

    print(f,chisquare)
    sim = fit
    io.plot_img(sim,f"sim_{f}.png",lim=300,verbose=False)
    io.plot_img(stamp-sim,f"diff_{f}.png",lim=300,verbose=False)



