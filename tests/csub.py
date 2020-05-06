from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs
from pixell import enmap,reproject,pointsrcs
import numpy as np
import os,sys
from tilec import fg as tfg
from enlib import bunch
import ptfit

imap_90 = enmap.read_map("/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f090_daynight_srcfree_map.fits",sel=np.s_[0,...])

imap_150 = enmap.read_map("/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f150_daynight_srcfree_map.fits",sel=np.s_[0,...])

ivar = enmap.read_map("/home/r/rbond/sigurdkn/project/actpol/map_coadd/20200228/release/act_planck_s08_s18_cmb_f090_daynight_ivar.fits",sel=np.s_[0,...])
print(ivar.shape)

cols = bunch.Bunch(catalogs.load_fits("confirmed_catalog.fits",['RAdeg','DECdeg','SNR','fixed_y_c','deltaT_c','name'],hdu_num=1,Nmax=None))
print(cols.SNR.size)
#sys.exit()
for i in range(cols.SNR.size):

    width_arcmin = 30.
    res_arcmin = 0.5
    nu_ghz = 150.

    ra = cols.RAdeg[i]
    dec = cols.DECdeg[i]
    stamp = reproject.cutout(imap_90, width=np.deg2rad(width_arcmin/60.), ra=np.deg2rad(ra), dec=np.deg2rad(dec), 
           res=np.deg2rad(res_arcmin/60.))
    
    if stamp is None: continue
    sel = maps.crop_center(stamp,1,sel=True) 
    cstamp = stamp.copy()
    cstamp[sel] = np.nan
    io.plot_img(cstamp,f"stamp_090_{cols.name[i]}_ra_{ra:.4f}_dec_{dec:.4f}.png",lim=300)

    stamp = reproject.cutout(imap_150, width=np.deg2rad(width_arcmin/60.), ra=np.deg2rad(ra), dec=np.deg2rad(dec), 
           res=np.deg2rad(res_arcmin/60.))
    
    if stamp is None: continue
    sel = maps.crop_center(stamp,1,sel=True) 
    cstamp = stamp.copy()
    cstamp[sel] = np.nan
    io.plot_img(cstamp,f"stamp_150_{cols.name[i]}_ra_{ra:.4f}_dec_{dec:.4f}.png",lim=300)

sys.exit()

#print(cols.SNR.max())
i = np.argwhere(cols.SNR==cols.SNR.max())[0,0]
#i = np.argwhere(np.logical_and(np.abs(cols.DECdeg)<60., np.logical_and(cols.SNR>50,cols.SNR<60)))[0,0]
#i = np.argwhere(np.logical_and(np.abs(cols.DECdeg)<60., np.logical_and(cols.SNR>10,cols.SNR<20)))[0,0]
#i = np.argwhere(np.logical_and(np.abs(cols.DECdeg)<60., np.logical_and(cols.SNR>35,cols.SNR<40)))[0,0]
#i = np.argwhere(np.logical_and(np.abs(cols.DECdeg)<60., np.logical_and(cols.SNR>20,cols.SNR<30)))[0,0]
#i = np.argwhere(np.logical_and(np.abs(cols.DECdeg)<60., np.logical_and(cols.SNR>40,cols.SNR<50)))[0,0]

print(cols.name[i])
ra = cols.RAdeg[i]
dec = cols.DECdeg[i]
#ra = cols.BCG_RADeg[i]
#dec = cols.BCG_DECDeg[i]
print(ra,dec)
y_c = cols.fixed_y_c[i]*1e-5

width_arcmin = 20.
res_arcmin = 0.5
nu_ghz = 150.

_,stamp = reproject.postage_stamp(imap, ra, dec, width_arcmin,
                  res_arcmin, proj='gnomonic', return_cutout=True,
                  npad=3, rotate_pol=False)

_,div_stamp = reproject.postage_stamp(ivar, ra, dec, width_arcmin,
                  res_arcmin, proj='gnomonic', return_cutout=True,
                  npad=3, rotate_pol=False)

sel = maps.crop_center(stamp,1,sel=True) 
stamp[sel] = np.nan
io.plot_img(stamp,"stamp.png",lim=300)
io.plot_img(div_stamp,"div.png")

fwhm = 1.4/60.
beam = np.deg2rad(maps.sigma_from_fwhm(fwhm))

fwhm2 = 6.0/60.
beam2 = np.deg2rad(maps.sigma_from_fwhm(fwhm2))

theory = cosmology.default_theory()
ells = np.arange(10000)
ps = theory.lCl('TT',ells)[None,None]
pflux,cov,fit = ptfit.ptsrc_fit(stamp,np.deg2rad(dec),np.deg2rad(ra),beam=beam,rbeam=beam2,div=div_stamp,ps=ps)
sim = fit
io.plot_img(sim,"sim.png",lim=300)
io.plot_img(stamp-sim,"diff.png",lim=300)

#




sys.exit()
tsz = y_c*tfg.get_mix(nu_ghz, 'tSZ')
tsz = cols.deltaT_c[i]

print(ra,dec)
print(np.deg2rad(ra),np.deg2rad(dec))

srcs = np.zeros((1,3))
srcs[0] = np.array((np.deg2rad(dec),np.deg2rad(ra),tsz))
sim = pointsrcs.sim_srcs(stamp.shape, stamp.wcs, srcs, beam)
io.plot_img(sim,"sim.png",lim=300)
io.plot_img(stamp-sim,"diff.png",lim=300)

print(tsz)
    
