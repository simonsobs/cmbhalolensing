import time as t
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, wcsutils
from orphics import mpi, maps, stats, io, cosmology
import sys
import healpy as hp
from symlens import qe

start = t.time() 

sim_path = "/global/project/projectdirs/act/data/msyriac/dr5_cluster_sims"
my_sim_path = "/global/cscratch1/sd/eunseong/data/webskysim"


omegab = 0.049
omegac = 0.261
omegam = omegab + omegac
h      = 0.68
rho    = 2.775e11 * omegam * h**2 # Msun/Mpc^3

f = open(f'{my_sim_path}/halos.pksc')
# only take first N entries for testing (there are ~8e8 halos total...)
N = 20000

catalog=np.fromfile(f, count=N*10, dtype=np.float32)
catalog=np.reshape(catalog, (N,10))

x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
R  = catalog[:,6] # Mpc

# convert to mass, RA and DEc
M200m       = 4*np.pi/3.*rho*R**3  # this is M200m (mean density 200 times mean) in Msun 
theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians

ms   = M200m/1e14
ras  = np.rad2deg(phi)
decs = np.rad2deg(np.pi/2. - theta)

print("Number of halos = ", N)

# select clusters above the certain mass limit 
ras0 = []
decs0 = []
ms0 = []
i = 0
for i in range(ras.size):
    if ms[i] > 3.:
        ras0.append(ras[i])
        decs0.append(decs[i])
        ms0.append(ms[i])

ras = np.array(ras0)
decs = np.array(decs0)	
ms = np.array(ms0)
print("Number of halos above the given mass = ", len(ras)) 


# stamp size and resolution 
px = 0.5
width = 120./60.
maxr = width * utils.degree / 2.0

# read maps 
fshape, fwcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj='car') 
imap = f'{my_sim_path}/kap_lt4.5.fits'
kmap = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=6000, rot=None)
print(kmap.shape) 

filename = f'{my_sim_path}/lensed_alm.fits'
alm = np.complex128(hp.read_alm(filename, hdu=(1, 2, 3)))
#print(alm.shape) (3, 34043626)
ncomp = 1
omap = enmap.empty((ncomp,)+fshape[-2:], fwcs, dtype=np.float64)
lmap = curvedsky.alm2map(alm, omap, spin=[0,2], oversample=2.0, method="auto")
print(lmap.shape) 

tap_per = 12.0
pad_per = 3.0

fwhm = 1.4
nlevel = 20

xlmin = 200 
xlmax = 2000
ylmin = 200
ylmax = 6000
ylcut = 2
klmin = 200
klmax = 5000

# change number of clusters here
nsims = len(ras)

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

for task in my_tasks:
    print(rank,task)
    
    i = task
    coords = np.array([decs[i], ras[i]]) * utils.degree  
    
    # cut out a stamp from the simulated map (CAR -> plain)
    kstamp = reproject.thumbnails(
        kmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="plain",
        oversample=2,
        depix=True
    )  
    
    lstamp = reproject.thumbnails(
        lmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="plain",
        oversample=2,
        depix=True
    )  

    kstamp = kstamp[0]
    lstamp = lstamp[0]
 
    if i == 0:   
        # get an edge taper map and apodize
        taper = maps.get_taper(
            l_stamp.shape,
            l_stamp.wcs,
            taper_percent=tap_per,
            pad_percent=pad_per,
            weight=None,
        )
        taper = taper[0] 
    
        # get geometry and Fourier info   
        shape = kstamp.shape
        wcs = kstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        
        assert wcsutils.equal(kstamp.wcs, lstamp.wcs)
        
        # evaluate the 2D Gaussian beam on an isotropic Fourier grid 
        beam2d = maps.gauss_beam(modlmap, fwhm) 
        
        # build Fourier space masks for lensing reconstruction
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=ylcut, lycut=ylcut)
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)    

        # get theory spectrum and build interpolated 2D Fourier CMB from theory and maps      
        theory = cosmology.default_theory()
        ucltt2d = theory.lCl('TT', modlmap)

        # total spectrum includes beam-deconvolved noise       
        npower = (nlevel * np.pi/180./60.)**2.
        tcltt2d = ucltt2d + npower/beam2d**2.
 
        
    # apply beam     
    k_stamp = maps.filter_map(kstamp, beam2d)
    l_stamp = maps.filter_map(lstamp, beam2d)
    
    # same filter as the post-reconstuction 
    fk_stamp = maps.filter_map(k_stamp, kmask)   
    s.add_to_stack('kstamp', fk_stamp) 
 
    
    tapered_stamp = l_stamp * taper
    
    # get a beam deconvolved Fourier stamp
    k_map = enmap.fft(tapered_stamp, normalize="phys")/beam2d
    assert np.all(np.isfinite(k_map))
    
    # build symlens dictionary 
    feed_dict = {
        'uC_T_T' : ucltt2d, 
        'tC_T_T' : tcltt2d, 
        'X' : k_map,
        'Y' : k_map,
    }   

    # do lensing reconstruction in Fourier space    
    rkmap = qe.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY="TT", xmask=xmask, ymask=ymask, kmask=kmask, physical_units=True)
    
    assert np.all(np.isfinite(rkmap))
        
    # transform to real space
    kappa = enmap.ifft(rkmap, normalize='phys').real    
    
    s.add_to_stack("lstamp", kappa)
        
s.get_stacks()
    
if rank==0:
    kmap = s.stacks['kstamp']
    io.plot_img(kmap, f'wsky_outs/kappa.png')   
    io.plot_img(kmap[100:140,100:140], f'wsky_outs/kappa_zoom.png')   
    
    lmap = s.stacks['lstamp']
    io.plot_img(lmap, f'wsky_outs/rec_kappa.png')   
    io.plot_img(lmap[100:140,100:140], f'wsky_outs/rec_kappa_zoom.png')    


elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)