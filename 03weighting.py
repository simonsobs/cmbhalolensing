import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pixell import enmap, reproject, enplot
from orphics import maps, mpi, io, stats
from scipy.optimize import curve_fit
from numpy import save
import symlens
import healpy as hp
import os, sys
import time as t

start = t.time() 

# AdvACT catalogue
catalogue_name = "data/AdvACT_S18Clusters_v1.0-beta.fits" #[4024] 
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg']
decs = hdu[1].data['DECDeg']
mass = hdu[1].data['M500']

# selecting clusters with mass estimates for weighting 
ras0 = []
decs0 = []
mass0 = []

for i in range(ras.size):
	if mass[i] != -99:
		ras0.append(ras[i])
		decs0.append(decs[i])
		mass0.append(mass[i])

ras = np.array(ras0)
decs = np.array(decs0)	
mass = np.array(mass0)

N_cluster = len(ras) 
N_stamp = []

nsims = N_cluster
nsims_rd = N_cluster + 6000

# the number of iteration for the mean field
N_iter = 10

# MPI paralellization! 
comm, rank, my_tasks = mpi.distribute(nsims)
comm_rd, rank_rd, my_tasks_rd = mpi.distribute(nsims_rd)
print('cluster stamp', rank)
print('random stamp', rank_rd)

s = stats.Stats(comm)
s_rd = stats.Stats(comm_rd)


# Planck tSZ deprojected map; so sz sources have been projected out
plc_map = "data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
hmap = hp.read_map(plc_map)

# Planck binary mask
plc_mask = "data/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
hmask = hp.read_map(plc_mask)

# now Planck map is masked 
pmap = hmap*hmask


# ACT coadd map
act_map = '/global/project/projectdirs/act/data/coadds/act_s08_s18_cmb_f150_daynight_srcfree_map.fits'
#act_map = 'data/modelSubtracted_updated.fits'
amap = enmap.read_map(act_map)
print(amap.shape)

# corresponding inverse-variance map
ivar_map = '/global/project/projectdirs/act/data/coadds/act_s08_s18_cmb_f150_daynight_srcfree_ivar.fits'
imap = enmap.read_map(ivar_map)



# bin size and range for 1D binned power spectrum 
act_edges = np.arange(100,8001,20)
plc_edges = np.arange(20,4001,20)

# function for fitting 1D power spectrum of given stamp 
def fit_p1d(cents, p1d, which):

    # remove nans in cls (about 10%)        
    mask_nan = ~np.isnan(p1d) 
    ells = cents[mask_nan]
    cltt = p1d[mask_nan]

    clarr = np.zeros(ells.size)  

    '''
    #####################
    # two parameter fit #
    #####################

    logy = np.log10(cltt)
       
    def line(x, a, b):
        return a*0.999**x + b

    popt, pcov = curve_fit(line, ells, logy, maxfev=1000)
    #perr = np.sqrt(np.diag(pcov))
    #print(popt, pcov, perr)

    i = 0
    for i in range(ells.size):          
        clarr[i] = line(ells[i], *popt)

    clarr = 10.**clarr 
    '''

    ######################
    # four parameter fit #
    ######################

    if which == 'act':
        cut1 = np.argmax(ells > 4000)  
        cut2 = np.argmax(ells > 5000) 

    elif which == 'plc':
        cut1 = np.argmax(ells > 3000)  
        cut2 = np.argmax(ells > 2000) 

    logx1 = np.log10(ells[:cut1])
    logy1 = np.log10(cltt[:cut1])
    logx2 = np.log10(ells[cut2:])
    logy2 = np.log10(cltt[cut2:])

    def line(x, a, b):
        return a*x + b

    popt1, pcov1 = curve_fit(line, logx1, logy1)
    popt2, pcov2 = curve_fit(line, logx2, logy2)

    ## logy = a*logx + b
    ## y = 10**b * x**a    

    amp1 = 10.**popt1[1]
    amp2 = 10.**popt2[1]
    ind1 = popt1[0]
    ind2 = popt2[0]    
    
    i = 0
    for i in range(ells.size):          
        clarr[i] = amp1*ells[i]**ind1 + amp2*ells[i]**ind2

    #if which == 'plc':
    #    plt.plot(ells, ells*(ells+1)*clarr/(2*np.pi), 'r-')
    #    plt.yscale('log')
    #    plt.xlabel("$\ell$")
    #    plt.ylabel("$\ell(\ell+1)C_{\ell}/2\pi\,$ [$\mu$K-rad]$^2$")
    #    plt.show()

    return ells, clarr



# stamp size and resolution 
stamp_width = 120.
pixel = 0.5

# beam and FWHM 
act_beam = 1.4 
plc_beam = 5.
act_fwhm = np.deg2rad(act_beam/60.)
plc_fwhm = np.deg2rad(plc_beam/60.)

# Planck stamp mask
xlmin=100
xlmax=2000

# ACT stamp mask
ylmin=500
ylmax=6000
lxcut=20
lycut=20

# kmask
klmin=40
klmax=5000


def stacking(N, ras, decs, which, k_iter=None):

    i = 0
    count = 0

    for i in N:

        # cut out a stamp from the ACT map (CAR -> gnomonic) 
        stamp = reproject.postage_stamp(amap, ras[i], decs[i], stamp_width, pixel)    
        ivar = reproject.postage_stamp(imap, ras[i], decs[i], stamp_width, pixel)
               
        if stamp is None: continue
        ivar = ivar[0]

        ##### temporary 1: avoid weird noisy ACT stamps
        if np.any(ivar <= 1e-4): continue
        astamp = stamp[0]
        if np.any(astamp >= 1e3): continue

        # cut out a stamp from the Planck map (healpix -> gnomonic) 
        pstamp = maps.get_planck_cutout(pmap, ras[i], decs[i], stamp_width, pixel)

        # unit: K -> uK 
        if pstamp is None: continue
        pstamp = pstamp*1e6
                 
        ##### temporary 2: avoid weird noisy Planck stamps - would 80% be good enough? 
        true = np.nonzero(pstamp)[0]
        ntrue = true.size
        req = 0.8*(2.*stamp_width)**2.
        if ntrue < req: continue

        # get an edge taper map and apodize
        taper = maps.get_taper(astamp.shape, astamp.wcs, taper_percent=12.0, pad_percent=3.0, weight=None)
        taper = taper[0]

        # applying this to the stamp makes it have a nice zeroed edge!    
        act_stamp = astamp*taper
        plc_stamp = pstamp*taper

        # get geometry and Fourier info
        shape, wcs = astamp.shape, astamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        
        # evaluate the 2D Gaussian beam on an isotropic Fourier grid 
        act_kbeam2d = np.exp(-(act_fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))
        plc_kbeam2d = np.exp(-(plc_fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))  

        # get theory spectrum - this should be the lensed spectrum!
        ells, dltt = np.loadtxt("data/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt = dltt/ells/(ells + 1.)*2.*np.pi

        # measure the binned power spectrum from given stamp 
        act_cents, act_p1d = maps.binned_power(astamp, bin_edges=act_edges, mask=taper) 
        plc_cents, plc_p1d = maps.binned_power(pstamp, bin_edges=plc_edges, mask=taper)  
        #plt.plot(act_cents, act_cents*(act_cents+1)*act_p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none') 
        #plt.plot(plc_cents, plc_cents*(plc_cents+1)*plc_p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none')

        # fit 1D power spectrum 
        act_ells, act_cltt = fit_p1d(act_cents, act_p1d, 'act')
        plc_ells, plc_cltt = fit_p1d(plc_cents, plc_p1d, 'plc') 
	    
        # build interpolated 2D Fourier CMB from theory and maps 
        ucltt = maps.interp(ells, cltt)(modlmap)
        tclaa = maps.interp(act_ells, act_cltt)(modlmap)
        tclpp = maps.interp(plc_ells, plc_cltt)(modlmap)

        ## total TT spectrum includes beam-deconvolved noise
        ## so create a total beam-deconvolved spectrum using a Gaussian beam func.

        tclaa = tclaa/(act_kbeam2d**2.)
        tclpp = tclpp/(plc_kbeam2d**2.)

        ## the noise was specified for a beam deconvolved map 
        ## so we deconvolve the beam from our map

        # get a beam deconvolved Fourier map
        act_kmap = np.nan_to_num(enmap.fft(act_stamp, normalize='phys')/act_kbeam2d)
        plc_kmap = np.nan_to_num(enmap.fft(plc_stamp, normalize='phys')/plc_kbeam2d)

        # build symlens dictionary 
        feed_dict = {
            'uC_T_T' : ucltt,     # goes in the lensing response func = lensed theory 
            'tC_A_T_A_T' : tclaa, # the fit ACT power spectrum with ACT beam deconvolved
            'tC_P_T_P_T' : tclpp, # approximate Planck power spectrum with Planck beam deconvolved 
            'tC_A_T_P_T' : ucltt, # same lensed theory as above, no instrumental noise  
            'X' : plc_kmap,       # Planck map
            'Y' : act_kmap        # ACT map
        }

        ## need to have a Fourier space mask in hand 
        ## that enforces what multipoles in the CMB map are included 

        # build a Fourier space mask    
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax)
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=lxcut, lycut=lycut)
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax)

        # ask for reconstruction in Fourier space
        krecon = symlens.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY='TT', xmask=xmask, ymask=ymask, field_names=['P','A'], xname='X_l1', yname='Y_l2', kmask=kmask, physical_units=True)

        # transform to real space
        kappa = enmap.ifft(krecon, normalize='phys').real

        ##### temporary 3: to get rid of stamps with tSZ cluster in random locations
        if np.any(np.abs(kappa) > 15): continue 

        # inverse variance noise weighting
        ivmean = np.mean(ivar)

        # stacking the stamps   
        if which == 'cluster':
            weight = ivmean*mass[i] 
            stack = kappa*weight
            s.add_to_stack('kmap', stack)  
            s.add_to_stack('kw', weight)
            
        elif which == 'random':
            stack = kappa*ivmean
            s_rd.add_to_stack('mean%s'%k, stack)
            s_rd.add_to_stack('mw%s'%k, ivmean)  

        # check actually how many stamps are cut out of given map
        count += 1

        # for mean field stacking counts to be same as the ones from cluster loc 
        if which == 'random' and count == N_stamp: break

    return(count)

# stacking at cluster positions 
N_stamp = stacking(my_tasks, ras, decs, 'cluster')

# random positions in the map
dec0, dec1, ra0, ra1 = [-62.0, 22.0, -180.0, 180.0]
deg_width = stamp_width/60.

# iteration for the mean field calculation 
k = 0
for k in range(N_iter):

    # generate random positions 
    decs_rd = np.random.rand(nsims_rd)*(dec1 - dec0 - deg_width) + dec0 + deg_width/2.
    ras_rd = np.random.rand(nsims_rd)*(ra1 - ra0 - deg_width) + ra0 + deg_width/2.
        
    # stacking at random positions 
    N_stamp_rd = stacking(my_tasks_rd, ras_rd, decs_rd, 'random', k)


# collect from all MPI cores and calculate stacks
s.get_stacks()
s_rd.get_stacks()

# and/or ?
if rank==0 and rank_rd==0: 

    kmap = s.stacks['kmap']
    kweights = s.stacks['kw']

    kmap /= kweights

    all_cl = s.stack_count['kmap']
    print("\r ::: number of cluster stamps : %d" %all_cl)

    mean = np.zeros(kmap.shape)
    mweights = 0

    i = 0
    for i in range(N_iter):
        mean += s_rd.stacks['mean%s'%i]
        mweights += s_rd.stacks['mw%s'%i]
        
        all_rd = s_rd.stack_count['mean%s'%i]
        print("\r ::: number of random stamps %d in %d : %d" %(i+1, N_iter, all_rd))      

    mean /= mweights
   
    final = kmap - mean

    io.plot_img(kmap,'plots/00kmap.png',flip=False,ftsize=12,ticksize=10)
    io.plot_img(mean,'plots/01mean.png',flip=False,ftsize=12,ticksize=10)
    io.plot_img(final,'plots/02final.png',flip=False,ftsize=12,ticksize=10)
    io.plot_img(final[100:140,100:140],'plots/03zoom.png',flip=False,ftsize=12,ticksize=10)

    save('plots/04pdata.npy',final)

    elapsed = t.time() - start
    print("\r ::: entire run took %.1f seconds" %elapsed)


