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



def fit_act_ps(cents, p1d):

    # remove nans in cls (about 10%)        
    mask_nan = ~np.isnan(p1d) 
    ells = cents[mask_nan]
    cltt = p1d[mask_nan]

    # fit the binned power spectrum to the power law
    cut1 = np.argmax(ells > 4500)  
    cut2 = np.argmax(ells > 5000) 

    logx1 = np.log10(ells[:cut1])
    logy1 = np.log10(cltt[:cut1])
    logx2 = np.log10(ells[cut2:])
    logy2 = np.log10(cltt[cut2:])

    def line(x, a, b):
        return a*x + b

    popt1, pcov1 = curve_fit(line, logx1, logy1)
    popt2, pcov2 = curve_fit(line, logx2, logy2)
    #perr1 = np.sqrt(np.diag(pcov1))
    #perr2 = np.sqrt(np.diag(pcov2))
    #print(popt1, pcov1, perr1)
    #print(popt2, pcov2, perr2)

    ## logy = a*logx + b
    ## y = 10**b * x**a    

    amp1 = 10.**popt1[1]
    amp2 = 10.**popt2[1]
    ind1 = popt1[0]
    ind2 = popt2[0]

    # fill in the cls 
    clarr = np.zeros(ells.size)
    clarr1 = np.zeros(ells.size)
    clarr2 = np.zeros(ells.size)        

    j = 0
    while (j < ells.size):          
        clarr1[j] = amp1*(ells[j]**ind1)
        clarr2[j] = amp2*(ells[j]**ind2)
        j += 1   

    j = 0
    while (j < ells.size):  
        if clarr1[j] <= clarr2[j]: break
        j += 1
        cross = j


    k = 0
    while (k < ells.size): 
        if k < cross:
            clarr[k] = clarr1[k]
        else:
            clarr[k] = clarr2[k]
        k += 1

    #plt.plot(ells, ells*(ells+1)*clarr1/(2*np.pi), 'k--', alpha=0.5)
    #plt.plot(ells, ells*(ells+1)*clarr2/(2*np.pi), 'k--', alpha=0.5)
    plt.plot(ells, ells*(ells+1)*clarr/(2*np.pi), 'r-')
    plt.yscale('log')
    #plt.xlabel("$\ell$")
    #plt.ylabel("$\ell(\ell+1)C_{\ell}/2\pi\,$ [$\mu$K$^2$]") 
    #plt.title('%d' % (count+1))
    #plt.savefig('plots/why/%d_ps.png' % (count+1)); plt.clf()
    plt.show()

    return ells, clarr


def fit_plc_ps(cents, p1d):
         
    # remove nans in cls (about 10%)        
    mask = ~np.isnan(p1d)
    ells = cents[mask]
    cltt = p1d[mask]

    # fit the binned power spectrum to the power law
    cut = np.argmax(ells > 3500)
        
    logx = np.log10(ells[0:cut])
    logy = np.log10(cltt[0:cut])

    def line0(x, a, b):
        return a*x + b

    popt, pcov = curve_fit(line0, logx, logy)
    #perr = np.sqrt(np.diag(pcov))
    #print(popt, pcov, perr)

    ## logy = a*logx + b
    ## y = 10**b * x**a    

    amp = 10.**popt[1]
    ind = popt[0]


    # fill in the cls 
    clarr = np.zeros(ells.size)

    j=0
    while (j < ells.size):
        clarr[j] = amp*(ells[j]**ind)
        j += 1          

    plt.plot(ells, ells*(ells+1)*clarr/(2*np.pi), 'r-')
    plt.yscale('log')
    #plt.xlabel("$\ell$")
    #plt.ylabel("$\ell(\ell+1)C_{\ell}/2\pi\,$ [$\mu$K-rad]$^2$")
    plt.show()


    return ells, clarr




def stacking(N, ras, decs):
    
    start_stack = time.time()   
    
    stack = 0
    count = 0
    
    for i in range(N):
    
        ## extract a postage stamp from a larger map
        ## by reprojecting to a coordinate system centered on the given position 
        ## (a tangent plane projection) 
        
        stamp_width = 120.
        stamp = reproject.postage_stamp(amap, ras[i], decs[i], stamp_width, 0.5)    
        ivar = reproject.postage_stamp(imap, ras[i], decs[i], stamp_width, 0.5)

        ## R.A. and Dec. are in degrees
        ## width_arcmin : stamp dimension in arcmin
        ## res_arcmin : width of pixel in arcmin
        ## therefore this makes a stamp of 120 x 120 arcmin (2 x 2 degree)
        ## consisting of 240 x 240 pixels of width 0.5 arcmins
        ## each pixel represents half arcmin
        
          
        if stamp is None: continue
        ivar = ivar[0]
        #print('ivar min and max before = ', np.min(ivar), np.max(ivar))
        if np.any(ivar <= 1e-4): continue
        #ready = stamp[0]	
        #if np.all(ivar > 0): ready = stamp[0]
        #else: continue
        #if (np.min(ivar) < 0): continue
        ready = stamp[0]
        
        if np.any(ready >= 1000): continue
	
        #print(count+1, 'stamp max', np.max(ready))
	
        #print(ras[i], decs[i])
        #plt.imshow(ready); plt.colorbar(); plt.savefig('plots/why/%d_stamp.png' % (count+1)); plt.clf() # plt.show()
        #plt.imshow(ivar); plt.colorbar(); plt.savefig('plots/why/%d_ivar.png' % (count+1)); plt.clf() # plt.show()
        #print(count+1, 'ivar min and max after = ', np.min(ivar), np.max(ivar))

        plt.imshow(ready); plt.colorbar(); plt.show()

    
        ## if we want to do any sort of harmonic analysis 
        ## we require periodic boundary conditions
        ## we can prepare an edge taper map on the same footprint as our map of interest

        # get an edge taper map and apodize
        taper = maps.get_taper(ready.shape, taper_percent=12.0, pad_percent=3.0, weight=None)
        taper = taper[0]
        # applying this to the stamp makes it have a nice zeroed edge!    
        tapered_stamp = ready*taper
        
        ## all outputs are 2D arrays in Fourier space
        ## so you will need some way to bin it in annuli
        ## a map of the absolute wavenumbers is useful for this : enmap.modlmap
    
        # get geometry and Fourier info
        shape, wcs = ready.shape, ready.wcs
        modlmap = enmap.modlmap(shape, wcs)

        #print(ready.shape, ready.wcs, modlmap.shape)
        
        # build a Gaussian beam transfer function 
        beam_arcmin = 1.4 
        fwhm = np.deg2rad(beam_arcmin/60.)
        
        # evaluate the beam on an isotropic Fourier grid 
        kbeam2d = np.exp(-(fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))    
    
        ## lensing noise curves require CMB power spectra
        ## this could be from camb theory data or actual map
        ## it is from camb at the moment 
    
        # get theory spectrum - this should be the lensed spectrum!
        ells0, dltt = np.loadtxt("data/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt0 = dltt/ells0/(ells0 + 1.)*2.*np.pi

        edges = np.arange(20,9001,20)
        cents, p1d = maps.binned_power(ready, bin_edges=edges, mask=taper) 
        plt.plot(cents, cents*(cents+1)*p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none');    

        ells, clarr = fit_act_ps(cents, p1d)

        ## interpolate ells and cltt 1D power spectrum specification 
        ## isotropically on to the Fourier 2D space grid
    	
        # build interpolated 2D Fourier CMB theory 
        ucltt = maps.interp(ells0, cltt0)(modlmap)
        tclaa = maps.interp(ells, clarr)(modlmap)
        tclaa = tclaa/(kbeam2d**2.)

        ## total TT spectrum includes beam-deconvolved noise
        ## so create a Planck-like total beam-deconvolved spectrum using a Gaussian beam func.
    	
        # total power spectrum for filters 
        #noise_arcmin = 10.
        #cl_noise = (noise_arcmin*np.pi/180./60.)**2.
        #tcltt2d = ucltt2d + np.nan_to_num(cl_noise/kbeam2d**2.)


        ##### Planck stamp #####

        pstamp = maps.get_planck_cutout(pmap, ras[i], decs[i], stamp_width, px=0.5)

        if pstamp is None: continue
                 
	# checking stamps - would 80% be good enough? 
        #true = np.nonzero(pstamp)[0]
        #ntrue = true.size
        #req = 0.8*(2.*stamp_width)**2.
        #if ntrue < req: continue

        pstamp = pstamp*1e6

        plt.imshow(pstamp); plt.colorbar(); plt.show()

        ## if we want to do any sort of harmonic analysis 
        ## we require periodic boundary conditions
        ## can prepare an edge taper map on the same footprint as our map of interest

        # get an edge taper map and apodize
        #ptaper = maps.get_taper(pstamp.shape, taper_percent=12.0, pad_percent=3.0, weight=None)
             
        # applying this to the stamp makes it have a nice zeroed edge!            
        #ptapered_stamp = pstamp*ptaper[0]
        #ptaper = ptaper[0]

        ptapered_stamp = pstamp*taper
        
        ## all outputs are 2D arrays in Fourier space
        ## so we will need some way to bin it in annuli
        ## a map of the absolute wavenumbers is useful for this : enmap.modlmap
    
        # get geometry and Fourier info
        #pshape, pwcs = ptapered_stamp.shape, ptapered_stamp.wcs
        #pmodlmap = enmap.modlmap(pshape, pwcs)

        #print(pstamp.wcs)
        
        # build a Gaussian beam transfer function 
        beam_arcmin_p = 5.
        pfwhm = np.deg2rad(beam_arcmin_p/60.)
        # evaluate the beam on an isotropic Fourier grid 
        pkbeam2d = np.exp(-(pfwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))  

        #(2)measure the binned power spectrum from the input map 
        pedges = np.arange(20,4001,20)
        pcents, pp1d = maps.binned_power(pstamp, bin_edges=pedges, mask=taper)   

        #print(p1d.shape[0]) = 120
        #for j in range(cents.size):
        #    print(j, cents[j], p1d[j])

        plt.plot(pcents, pcents*(pcents+1)*pp1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none')

        ellsp, clarrp = fit_plc_ps(pcents, pp1d)
     
        ## interpolate ells and clarr 1D power spectrum specification 
        ## isotropically on to the Fourier space grid  
 	
        # build interpolated 2D Fourier CMB theory 
        tclpp = maps.interp(ellsp, clarrp)(modlmap)
        tclpp = tclpp/(pkbeam2d**2.)

    	
        ## need to have a Fourier space mask in hand 
        ## that enforces what multipoles in the CMB map are included 

        # build a Fourier space mask     
        xmask = modlmap*0+1
        xmask[modlmap < 100] = 0
        xmask[modlmap > 2000] = 0

        ymask = modlmap*0+1
        ymask[modlmap < 500] = 0
        ymask[modlmap > 6000] = 0

        kmask = modlmap*0+1
        kmask[modlmap < 40] = 0
        kmask[modlmap > 5000] = 0
    	
        ## the noise was specified for a beam deconvolved map 
        ## so we deconvolve the beam from our map

        # get a beam deconvolved Fourier map
        kmap_p = np.nan_to_num(enmap.fft(ptapered_stamp, normalize='phys')/pkbeam2d)
        kmap_a = np.nan_to_num(enmap.fft(tapered_stamp, normalize='phys')/kbeam2d)
    
        # build symlens dictionary 
        feed_dict = {
            'uC_T_T' : ucltt, # goes in the lensing response func = lensed theory 
            'tC_A_T_A_T' : tclaa, # the fit ACT power spectrum with ACT beam deconvolved
            'tC_P_T_P_T' : tclpp, # approximate Planck power spectrum with Planck beam deconvolved 
            'tC_A_T_P_T' : ucltt, # same lensed theory as above, no instrumental noise  
            'X' : kmap_p, # Planck map
            'Y' : kmap_a # ACT map
        }
    
        # ask for reconstruction in Fourier space
        krecon = symlens.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY='TT', xmask=xmask, ymask=ymask, field_names=['P','A'], xname='X_l1', yname='Y_l1', kmask=kmask, physical_units=True)
    
        # transform to real space
        kappa = enmap.ifft(krecon, normalize='phys').real


        #print(maps.minimum_ell(shape, wcs))

        kedges = np.arange(20,8001,400)
        kcents, kp1d = maps.binned_power(kappa, bin_edges=kedges)   

        #print(kp1d)
        plt.plot(kcents, kcents*(kcents+1)*kp1d/(2.*np.pi))
        plt.show()




        #if np.any(np.abs(kappa) > 15): continue

        #print(count+1, 'kappa max =', np.max(np.abs(kappa)))

        # stacking the stamps   
        stack += kappa

        # to  check actually how many stamp are cut
        count += 1

        #plt.imshow(kappa); plt.colorbar(); plt.savefig('plots/why/%d_kappa.png' % count); plt.clf()  #plt.show()
        plt.imshow(kappa); plt.colorbar(); plt.show()

        break 

        # for mean field stacking counts 
        if N_stamp is not None and count == N_stamp: break

    # averging the stack of stamps
    stack /= count

    #plt.imshow(stack); plt.show()
    
    elapsed_stack = time.time() - start_stack
    print("\r ::: stacking took %.1f seconds " % elapsed_stack)

    return(stack, count, stamp_width)

# choose the input map ***    
#input_map = amap


# staking at cluster positions 
field, N_stamp, stamp_width = stacking(N_cluster, ras, decs)
print("\r ::: number of stamps of clusters : %d " % N_stamp)

#N_stamp = 20
#stamp_width = 120

'''
# to find random positions in the map
dec0, dec1, ra0, ra1 = [-62.0, 22.0, -180.0, 180.0]
width = stamp_width/60.


# iterations for mean field
N_iterations = 2
meanf = np.zeros(field.shape)

# in case that a stamp is not created in a given location 
N_bigger = N_stamp + 4000

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
    #print(k+1)
    #plt.imshow(meanf); plt.colorbar(); plt.savefig('plots/why/mean.png') #plt.show()


#img = np.zeros(field.shape)

#k = 0 
#while (k < N_iterations):
#    img += enmap.read_map('test/%d_iter.fits' %(k+1))
#    k += 1

# averging over the number of iterations
#meanf = img/N_iterations


meanf /= N_iterations

# mean field subtraction
final = field - meanf

elapsed_total = int((time.time() - start_total)/60.0)
print("\r ::: entire run took {} minutes ".format(elapsed_total))

plt.imshow(field); plt.title('stacked lensing reconstruction'); plt.colorbar(); plt.show()
plt.imshow(meanf); plt.title('mean field (%d iterations)' % N_iterations); plt.colorbar(); plt.show()
plt.imshow(final); plt.title('after mean field subtraction'); plt.colorbar(); plt.show()
plt.imshow(final[100:140,100:140]); plt.title('zoom'); plt.colorbar(); plt.show()

#plt.savefig('plots/30field_realmap.png'); plt.clf(); 
#plt.savefig('plots/31meanf_realmap.png'); plt.clf();
#plt.savefig('plots/32final_realmap.png'); plt.clf();
#plt.savefig('plots/33zoom_realmap.png'); plt.clf();
'''

'''
plt.imshow(field); plt.title('stacked lensing reconstruction'); plt.colorbar(); plt.savefig('plots/14field_realmap.png'); plt.show()
plt.imshow(meanf); plt.title('mean field (%d iterations)' % N_iterations); plt.colorbar(); plt.savefig('plots/15meanf_realmap.png'); plt.show()
plt.imshow(final); plt.title('after mean field subtraction'); plt.colorbar(); plt.savefig('plots/16final_realmap.png'); plt.show()
plt.imshow(final[100:140,100:140]); plt.title('zoom'); plt.colorbar(); plt.savefig('plots/17zoom_realmap.png'); plt.show()
'''
