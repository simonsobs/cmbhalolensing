from pixell import enmap, reproject, enplot
from astropy.io import fits
from orphics import maps
import numpy as np
import matplotlib.pyplot as plt
#import soapack.interfaces as sints
import symlens
import time
import sys

start_total = time.time() 

#mask = sints.get_act_mr3_crosslinked_mask('deep6', version='180323')
#dm = sints.ACTc7v5(region=mask)
#imap = dm.get_coadd("S1","D6","PA1",ncomp=1)

# input map 
act_map = "data/ACTPol_148_D6_PA1_S1_1way_I.fits"
imap = enmap.read_map(act_map)
# 1-way inverse-variance co-added field map I indicates calibrated T map with sources restored 

# mask for input map
act_mask = "data/mask_05.00arcmin_0.015Jy_new.fits"
mask = enmap.read_map(act_mask)
# the mask is unity everywhere except in circles of 10 arcmin diameter around sources detected in the data where it is zero 

# chosen by me for now
dec0, ra0 = np.deg2rad([-8, 27])
dec1, ra1 = np.deg2rad([0, 42])

# construct a coordinate box using the convention [[dec_from,ra_from],[[dec_to,ra_to]] 
box = [[dec0, ra0], [dec1, ra1]]

imap_cut = imap.submap(box)
mask_cut = mask.submap(box)

amap = imap_cut*mask_cut
#enplot.show(enplot.plot(smap, range=300, mask=0))


# Planck tSZ deprojected map; so sz sources have been projected out
# I assume that sources are masked as well?
planck_nosz = "data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
pmap = reproject.enmap_from_healpix(planck_nosz, amap.shape, amap.wcs, ncomp=1, unit=1, lmax=6000,rot="gal,equ")
# so this would be a gradient map 

#enplot.show(enplot.plot(pmap, range=300, mask=0))


#catalogue_name = "data/E-D56Clusters.fits"
catalogue_name = "data/AdvACT_190220_confirmed.fits"  # new ACT catalogue yeahhhh
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg']
decs = hdu[1].data['DECDeg']

N = len(ras) 
#print(N)
#print(smap.shape) 
	

def stacking(N, input_map, ras, decs):
    
    start_stack = time.time()   
    
    stack = 0
    count = 0
    
    for i in range(N):
    
        ## extract a postage stamp from a larger map
        ## by reprojecting to a coordinate system centered on the given position 
        ## (a tangent plane projection) 
        
        stamp_width = 120.
        stamp = reproject.postage_stamp(input_map, ras[i], decs[i], stamp_width, 0.5)
    
        ## R.A. and Dec. are in degrees
        ## width_arcmin : stamp dimension in arcmin
        ## res_arcmin : width of pixel in arcmin
        ## therefore this makes a stamp of 120 x 120 arcmin (2 x 2 degree)
        ## consisting of 240 x 240 pixels of width 0.5 arcmins
        ## each pixel represents half arcmin
        
          
        if stamp is None: continue
        ready = stamp[0]
    
        ## if we want to do any sort of harmonic analysis 
        ## we require periodic boundary conditions
        ## we can prepare an edge taper map on the same footprint as our map of interest

        # get an edge taper map and apodize
        taper = maps.get_taper(ready.shape, taper_percent=12.0, pad_percent=3.0, weight=None)
             
        # applying this to the stamp makes it have a nice zeroed edge!    
        tapered_stamp = ready*taper[0]
        
        ## all outputs are 2D arrays in Fourier space
        ## so you will need some way to bin it in annuli
        ## a map of the absolute wavenumbers is useful for this : enmap.modlmap
    
        # get geometry and Fourier info
        shape, wcs = tapered_stamp.shape, tapered_stamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        
        # build a Gaussian beam transfer function 
        beam_arcmin = 1.4
        fwhm = np.deg2rad(beam_arcmin/60.)
        # evaluate the beam on an isotropic Fourier grid 
        kbeam2d = np.exp(-(fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))    
    
        ## lensing noise curves require CMB power spectra
        ## this could be from camb theory data or actual map
    
        # get theory spectrum - this should be the lensed spectrum!
        ells, dltt = np.loadtxt("data/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt = dltt/ells/(ells + 1.)*2.*np.pi
    
        ## interpolate ells and cltt 1D power spectrum specification 
        ## isotropically on to the Fourier space grid
    	
        # build interpolated 2D Fourier CMB theory 
        ucltt2d = np.interp(modlmap, ells, cltt)

        ## total TT spectrum includes beam-deconvolved noise
        ## so create a Planck-like total beam-deconvolved spectrum using a Gaussian beam func.
    	
        # total noise power spectrum for filters 
        noise_arcmin = 10.
        cl_noise = (noise_arcmin*np.pi/180./60.)**2.
        tcltt2d = ucltt2d + np.nan_to_num(cl_noise/kbeam2d**2.)
    	
        ## need to have a Fourier space mask in hand 
        ## that enforces what multipoles in the CMB map are included 
    
        # build a Fourier space mask 
        kmask = modlmap*0+1
        kmask[modlmap < 500] = 0
        kmask[modlmap > 3000] = 0
    	
        ## the noise was specified for a beam deconvolved map 
        ## so we deconvolve the beam from our map

        # get a beam deconvolved Fourier map
        kmap = np.nan_to_num(enmap.fft(tapered_stamp, normalize='phys')/kbeam2d)
    
        # build symlens dictionary 
        feed_dict = {
            'uC_T_T' : ucltt2d, # goes in the lensing response func 
            'tC_T_T' : tcltt2d, # goes in the lensing filters 
            'X' : kmap,
            'Y' : kmap,
        }
    
        # ask for reconstruction in Fourier space
        krecon = symlens.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY='TT', xmask=kmask, ymask=kmask, kmask=kmask, physical_units=True)
    
        # transform to real space
        kappa = enmap.ifft(krecon, normalize='phys').real
    
        # stacking the stamps   
        stack += kappa
        
        # to  check actually how many stamp are cut
        count += 1

    # averging the stack of stamps
    stack /= count
    
    elapsed_stack = time.time() - start_stack
    print("\r stacking took %.1f seconds " % elapsed_stack)

    return(stack, count, stamp_width)

# choose the input map ***    
input_map = amap

# staking at cluster positions 
field, N_stamp, stamp_width = stacking(N, input_map, ras, decs)
print("\r number of stamps : %d " % N_stamp)

# to find random positions in the map
dec00, ra00 = np.rad2deg([dec0, ra0])
dec11, ra11 = np.rad2deg([dec1, ra1])
width = stamp_width/60. + .5

# iterations for mean field
N_iterations = 400  
meanf = np.zeros(field.shape)
k = 0  

# stacking at random positions : choose the input map ***
while (k < N_iterations):
    
    rd_decs = np.random.rand(N_stamp)*(dec11 - dec00 - width) + dec00 + width/2.
    rd_ras = np.random.rand(N_stamp)*(ra11 - ra00 - width) + ra00 + width/2.
    
    rd_field, N_rd_stamp,_ = stacking(N_stamp, input_map, rd_ras, rd_decs)
    #print("\r number of clusters : %d " % N_rd_stamp)
    if N_rd_stamp is N_stamp: print("\r (the number of stamp matches in iteration {})".format(k+1))

    
    meanf += rd_field
    
    #print("\r iteration complete: %d of %d  " % ((k+1), N_iterations))
    #sys.stdout.write("\r iteration complete: %d of %d " % ((k+1), N_iterations))
    #sys.stdout.flush()
    k += 1


# averging over the number of iterations
meanf /= N_iterations

# mean field subtraction
final = field - meanf

elapsed_total = int((time.time() - start_total)/60.0)
print("\r\r entire run took {} minutes ".format(elapsed_total))

plt.imshow(field); plt.title('stacked lensing reconstruction'); plt.colorbar(); plt.show()
plt.imshow(meanf); plt.title('mean field'); plt.colorbar(); plt.show()
plt.imshow(field); plt.title('after mean field subtraction');  plt.colorbar(); plt.show()
plt.imshow(field[100:140,100:140]); plt.title('zoom');  plt.colorbar(); plt.show()

