from pixell import enmap, reproject
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import symlens
from orphics import maps

map_temp = "ACTPol_148_D6_PA1_S1_1way_I.fits"
imap = enmap.read_map(map_temp)
# 1-way inverse-variance co-added field map 
# I indicates calibrated T map with sources restored 

mask_temp = "mask_05.00arcmin_0.015Jy_new.fits"
mask = enmap.read_map(mask_temp)
# the mask is unity everywhere except in circles of 10 arcmin diameter 
# around sources detected in the data where it is zero 

dec0, ra0 = np.deg2rad([-8, 27])
dec1, ra1 = np.deg2rad([0, 42])

box = [[dec0, ra0], [dec1, ra1]]

imap_cut = imap.submap(box)
mask_cut = mask.submap(box)

smap = imap_cut*mask_cut

catalogue_name = "E-D56Clusters.fits"
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg']
decs = hdu[1].data['DECDeg']

N = len(ras) 

def stacking(N, smap, ras, decs):

    stack = 0
    count = 0
    
    for i in range(N):
    
        stamp = reproject.postage_stamp(smap, ras[i], decs[i], 120., 0.5)
    
        if stamp is None: continue
            		
        ready = stamp[0]
    
        # get an edge taper map and apodize
        taper = maps.get_taper(ready.shape, taper_percent=12.0)
        tstamp = ready*taper[0]

        # get geometry and Fourier info
        shape, wcs = tstamp.shape, tstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)

        # build a beam 
        beam_arcmin = 1.4
        fwhm = np.deg2rad(beam_arcmin/60.)
        kbeam2d = np.exp(-(fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))
        
        # get theory spectrum - this should be the lensed spectrum! 
        ells, dltt = np.loadtxt("pixell_tutorials/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt = dltt/ells/(ells + 1.)*2.*np.pi
    	
        # build interpolated 2D Fourier CMB theory 
        ucltt2d = np.interp(modlmap, ells, cltt)
    	
        # total noise power spectrum for filters 
        noise_arcmin = 10.
        cl_noise = (noise_arcmin*np.pi/180./60.)**2.
        tcltt2d = ucltt2d + np.nan_to_num(cl_noise/kbeam2d**2.)
    	
        # build a Fourier space mask 
        kmask = modlmap*0+1
        kmask[modlmap < 500] = 0
        kmask[modlmap > 3000] = 0
        
        # get a beam deconvolved Fourier map
        kmap = np.nan_to_num(enmap.fft(tstamp, normalize='phys')/kbeam2d)

        # build symlens dictionary 
        feed_dict = {
            'uC_T_T' : ucltt2d, # goes in the lensing response func 
            'tC_T_T' : tcltt2d, # goes in the lensing filters 
            'X' : kmap,
            'Y' : kmap,
        }
    
        # ask for reconstruction in Fourier space
        krecon = symlens.reconstruct(shape, wcs, feed_dict, estimator='hu_ok', XY='TT', xmask=kmask, ymask=kmask, kmask=kmask, physical_units=True)
    
        # transform to real space
        kappa = enmap.ifft(krecon, normalize='phys').real
    
        # stacking the stamps :D       
        stack += kappa
        count += 1

    stack /= count    
    print("\r number of clusters : %d " % count)

    return(stack, count)


# staking clusters
field, N_stamp = stacking(N, smap, ras, decs)

dec00, ra00 = np.rad2deg([dec0, ra0])
dec11, ra11 = np.rad2deg([dec1, ra1])

N_iterations = 5
k = 0    
meanf = np.zeros(field.shape)

# summing up the reconstructed stamp of random positions  
while (k < N_iterations):
    
    rd_decs = np.random.rand(N_stamp)*(dec11 - dec00 - 2.) + dec00 + 1.
    rd_ras = np.random.rand(N_stamp)*(ra11 - ra00 - 2.) + ra00 + 1.
    
    randf, count_check = stacking(N_stamp, smap, rd_ras, rd_decs)
    meanf += randf
    
    print("\r iteration complete: %d of %d  " % ((k+1), N_iterations))
    k += 1

# averging 
meanf /= N_iterations

# subtraction
final = field - meanf

plt.imshow(final)
plt.colorbar()
plt.show()

