from pixell import enmap, reproject, enplot
from astropy.io import fits
from orphics import maps
import numpy as np
import matplotlib.pyplot as plt
import symlens
import time
from scipy.optimize import curve_fit
import healpy as hp


start_total = time.time() 

# Planck tSZ deprojected map; so sz sources have been projected out
plc_map = "data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
hmap = hp.read_map(plc_map)

# Planck binary mask
plc_mask = "data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
hmask = hp.read_map(plc_mask)

# now the map is masked 
pmap = hmap*hmask


# ACT coadd map;
act_map = '/global/project/projectdirs/act/data/coadds/act_s08_s18_cmb_f150_daynight_srcfree_map.fits'
amap = enmap.read_map(act_map)
print(amap.shape)

ivar_map = '/global/project/projectdirs/act/data/coadds/act_s08_s18_cmb_f150_daynight_srcfree_ivar.fits'
imap = enmap.read_map(ivar_map)

#enplot.show(enplot.plot(amap[0], downgrade=4, colorbar=True))
#plot = enplot.plot(amap[0], downgrade=4, colorbar=True)
#enplot.write("coadd_map", plot)


# new ACT catalogue yeahhhh 3200 clusters
catalogue_name = "data/AdvACT_190220_confirmed.fits" 
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg']
decs = hdu[1].data['DECDeg']

N_cluster = len(ras) 
N_stamp = []

N_cluster = 10


def fit_p1d(cents, p1d):

    # remove nans in cls (about 10%)        
    mask_nan = ~np.isnan(p1d) 
    ells = cents[mask_nan]
    cltt = p1d[mask_nan]

    logy = np.log10(cltt)
       
    def line(x, a, b):
        return a*0.999**x + b

    popt, pcov = curve_fit(line, ells, logy, maxfev=1000)
    print(popt)


    # fill in the cls 
    clarr = np.zeros(ells.size)       

    i = 0
    for i in range(ells.size):          
        clarr[i] = line(ells[i], *popt)

    clarr = 10.**clarr

    #plt.plot(ells, ells*(ells+1)*clarr1/(2*np.pi), 'k--', alpha=0.5)
    #plt.plot(ells, ells*(ells+1)*clarr2/(2*np.pi), 'k--', alpha=0.5)
    plt.plot(ells, ells*(ells+1)*clarr/(2*np.pi), 'r-')
    plt.yscale('log')
    plt.xlabel("$\ell$")
    plt.ylabel("$\ell(\ell+1)C_{\ell}/2\pi\,$ [$\mu$K$^2$]") 
    #plt.ylim([1, 5e4])
    #plt.title('%d' % (count+1))
    #plt.savefig('plots/why/%d_ps.png' % (count+1)); plt.clf()
    plt.show()

    return ells, clarr



def stacking(N, ras, decs):
    
    start_stack = time.time()   
    
    stack = 0
    count = 0
    weights = 0
    
    for i in range(N):
    
        ## extract a postage stamp from a larger map
        ## by reprojecting to a coordinate system centered on the given position 
        ## (a tangent plane projection) 
        
        stamp_width = 120.
        pixel = 0.5

        stamp = reproject.postage_stamp(amap, ras[i], decs[i], stamp_width, pixel)    
        ivar = reproject.postage_stamp(imap, ras[i], decs[i], stamp_width, pixel)

        ## R.A. and Dec. are in degrees
        ## width_arcmin : stamp dimension in arcmin
        ## res_arcmin : width of pixel in arcmin
        ## therefore this makes a stamp of 120 x 120 arcmin (2 x 2 degree)
        ## consisting of 240 x 240 pixels of width 0.5 arcmins
        ## each pixel represents half arcmin
                 
        if stamp is None: continue
        ivar = ivar[0]

        if np.any(ivar <= 1e-4): continue
        astamp = stamp[0]
        
        # remove noisy stamps due to the galaxy
        if np.any(astamp >= 1000): continue
        #print(ras[i], decs[i])

        #plt.imshow(astamp); plt.colorbar(); plt.savefig('plots/why/%d_stamp.png' % (count+1)); plt.clf() # plt.show()
        #plt.imshow(ivar); plt.colorbar(); plt.savefig('plots/why/%d_ivar.png' % (count+1)); plt.clf() # plt.show()
        #plt.imshow(astamp); plt.colorbar(); plt.show()
  
        ## if we want to do any sort of harmonic analysis 
        ## we require periodic boundary conditions
        ## we can prepare an edge taper map on the same footprint as our map of interest

        # get an edge taper map and apodize
        taper = maps.get_taper(astamp.shape, taper_percent=12.0, pad_percent=3.0, weight=None)
        taper = taper[0]

        # applying this to the stamp makes it have a nice zeroed edge!    
        act_stamp = astamp*taper
        
        ## all outputs are 2D arrays in Fourier space
        ## so you will need some way to bin it in annuli
        ## a map of the absolute wavenumbers is useful for this : enmap.modlmap
    
        # get geometry and Fourier info
        shape, wcs = astamp.shape, astamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        
        # build a Gaussian beam transfer function 
        act_beam = 1.4 
        act_fwhm = np.deg2rad(act_beam/60.)
        
        # evaluate the beam on an isotropic Fourier grid 
        act_kbeam2d = np.exp(-(act_fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))    
    
        ## lensing noise curves require CMB power spectra
        ## this could be from camb theory data or actual map
        ## it is from camb at the moment 
    
        # get theory spectrum - this should be the lensed spectrum!
        ells, dltt = np.loadtxt("data/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt = dltt/ells/(ells + 1.)*2.*np.pi

        # measure the binned power spectrum from given stamp 
        act_edges = np.arange(100,8001,20)
        act_cents, act_p1d = maps.binned_power(astamp, bin_edges=act_edges, mask=taper) 

        plt.plot(act_cents, act_cents*(act_cents+1)*act_p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none') 

        # fit 1D power spectrum 
        act_ells, act_cltt = fit_p1d(act_cents, act_p1d)

        # cut out stamp from given Planck map as well
        pstamp = maps.get_planck_cutout(pmap, ras[i], decs[i], stamp_width, pixel)

        if pstamp is None: continue
                 
	# checking stamps - would 80% be good enough? 
        true = np.nonzero(pstamp)[0]
        ntrue = true.size
        req = 0.8*(2.*stamp_width)**2.
        if ntrue < req: continue

        # K -> uK 
        pstamp = pstamp*1e6

        #plt.imshow(pstamp); plt.colorbar(); plt.show()

        # taper the stamp for Fourier transform 
        plc_stamp = pstamp*taper
        
         # build a Gaussian beam transfer function 
        plc_beam = 5.
        plc_fwhm = np.deg2rad(plc_beam/60.)

        # evaluate the beam on an isotropic Fourier grid 
        plc_kbeam2d = np.exp(-(plc_fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))  

        # measure the binned power spectrum from given stamp 
        plc_edges = np.arange(20,4001,20)
        plc_cents, plc_p1d = maps.binned_power(pstamp, bin_edges=plc_edges, mask=taper)  

        plt.plot(plc_cents, plc_cents*(plc_cents+1)*plc_p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none')

        # fit 1D power spectrum 
        plc_ells, plc_cltt = fit_p1d(plc_cents, plc_p1d) 
  
        ## interpolate ells and cltt 1D power spectrum specification 
        ## isotropically on to the Fourier 2D space grid
    	
        # build interpolated 2D Fourier CMB theory 
        ucltt = maps.interp(ells, cltt)(modlmap)
        tclaa = maps.interp(act_ells, act_cltt)(modlmap)
        tclpp = maps.interp(plc_ells, plc_cltt)(modlmap)

        ## total TT spectrum includes beam-deconvolved noise
        ## so create a total beam-deconvolved spectrum using a Gaussian beam func.
        tclaa = tclaa/(act_kbeam2d**2.)
        tclpp = tclpp/(plc_kbeam2d**2.)
  	
        # total power spectrum for filters 
        #noise_arcmin = 10.
        #cl_noise = (noise_arcmin*np.pi/180./60.)**2.
        #tcltt2d = ucltt2d + np.nan_to_num(cl_noise/kbeam2d**2.)
   	
        ## need to have a Fourier space mask in hand 
        ## that enforces what multipoles in the CMB map are included 

        # build a Fourier space mask    
        xmask = maps.mask_kspace(shape, wcs, lmin=100, lmax=2000)
        ymask = maps.mask_kspace(shape, wcs, lmin=500, lmax=6000, lxcut=20, lycut=20)
        kmask = maps.mask_kspace(shape, wcs, lmin=40, lmax=5000)

        ## the noise was specified for a beam deconvolved map 
        ## so we deconvolve the beam from our map

        # get a beam deconvolved Fourier map
        plc_kmap = np.nan_to_num(enmap.fft(plc_stamp, normalize='phys')/plc_kbeam2d)
        act_kmap = np.nan_to_num(enmap.fft(act_stamp, normalize='phys')/act_kbeam2d)
    
        # build symlens dictionary 
        feed_dict = {
            'uC_T_T' : ucltt, # goes in the lensing response func = lensed theory 
            'tC_A_T_A_T' : tclaa, # the fit ACT power spectrum with ACT beam deconvolved
            'tC_P_T_P_T' : tclpp, # approximate Planck power spectrum with Planck beam deconvolved 
            'tC_A_T_P_T' : ucltt, # same lensed theory as above, no instrumental noise  
            'X' : plc_kmap, # Planck map
            'Y' : act_kmap  # ACT map
        }
    
        # ask for reconstruction in Fourier space
        krecon = symlens.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY='TT', xmask=xmask, ymask=ymask, field_names=['P','A'], xname='X_l1', yname='Y_l2', kmask=kmask, physical_units=True)

    
        #pr2d = (krecon*np.conj(krecon)).real
        #plt.imshow(np.log10(np.fft.fftshift(pr2d))); plt.colorbar(); plt.show()

        # transform to real space
        kappa = enmap.ifft(krecon, normalize='phys').real

        #print(maps.minimum_ell(shape, wcs)) : 180

        kedges = np.arange(20,8001,400)
        kcents, kp1d = maps.binned_power(kappa, bin_edges=kedges)   

        #print(kp1d)
        #plt.plot(kcents, kcents*(kcents+1)*kp1d/(2.*np.pi)); plt.show()

        if np.any(np.abs(kappa) > 15): continue    # clusters 
        #print(count+1, 'kappa max =', np.max(np.abs(kappa)))


        # noise weighting and staking the stamps
        ivmean = np.mean(ivar)
        ivmean = 1./ivmean    
        stack += kappa*ivmean
        weights += ivmean

        # stacking the stamps   
        # stack += kappa

        # to  check actually how many stamp are cut
        count += 1

        #plt.imshow(kappa); plt.colorbar(); plt.savefig('plots/why/%d_kappa.png' % count); plt.clf() #plt.show()
        #plt.imshow(kappa); plt.colorbar(); plt.show()

        # for mean field stacking counts 
        if N_stamp is not None and count == N_stamp: break

    # averging the stack of stamps
    # stack /= count
    stack /= weights 

    #plt.imshow(stack); plt.show()
    
    elapsed_stack = time.time() - start_stack
    print("\r ::: stacking took %.1f seconds " % elapsed_stack)

    return(stack, count, stamp_width)


# staking at cluster positions 
field, N_stamp, stamp_width = stacking(N_cluster, ras, decs)
print("\r ::: number of stamps of clusters : %d " % N_stamp)


# to find random positions in the map
dec0, dec1, ra0, ra1 = [-62.0, 22.0, -180.0, 180.0]
width = stamp_width/60.


# iterations for mean field
N_iterations = 4
meanf = np.zeros(field.shape)

# in case that a stamp is not created in a given location 
N_bigger = N_stamp + 5000

k = 0  
# stacking at random positions 
while (k < N_iterations):
    
    rd_decs = np.random.rand(N_bigger)*(dec1 - dec0 - width) + dec0 + width/2.
    rd_ras = np.random.rand(N_bigger)*(ra1 - ra0 - width) + ra0 + width/2.
    
    rd_field, N_rd_stamp, _ = stacking(N_bigger, rd_ras, rd_decs)
    print("\r ::: number of stamps of random positions : %d " % N_rd_stamp)

    #enmap.write_fits('test/%d_iter.fits'% (k+1), rd_field)
    print("\r ::: iteration complete: %d of %d  " % ((k+1), N_iterations))
	
    meanf += rd_field

    k += 1
    #plt.imshow(meanf); plt.colorbar(); plt.savefig('plots/why/mean.png') #plt.show()


# averging over the number of iterations
meanf /= N_iterations

# mean field subtraction
final = field - meanf

elapsed_total = int((time.time() - start_total)/60.0)
print("\r ::: entire run took {} minutes ".format(elapsed_total))

plt.imshow(field); plt.title('stacked lensing reconstruction'); plt.colorbar(); plt.savefig('plots/70field_noise.png'); plt.clf() 
plt.imshow(meanf); plt.title('mean field (%d iterations)' % N_iterations); plt.colorbar(); plt.savefig('plots/71meanf_noise.png'); plt.clf()
plt.imshow(final); plt.title('after mean field subtraction'); plt.colorbar(); plt.savefig('plots/72final_noise.png'); plt.clf()
plt.imshow(final[100:140,100:140]); plt.title('zoom'); plt.colorbar(); plt.savefig('plots/73zoom_noise.png'); plt.clf()


