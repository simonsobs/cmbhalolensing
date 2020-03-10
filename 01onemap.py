from pixell import enmap, reproject, enplot
from astropy.io import fits
from orphics import maps
import numpy as np
import matplotlib.pyplot as plt
import symlens
import time
from scipy.optimize import curve_fit


start_total = time.time()

act_map = '/global/project/projectdirs/act/data/coadds/act_planck_s08_s18_cmb_f150_daynight_srcfree_map.fits'
amap = enmap.read_map(act_map)
print(amap.shape)

ivar_map = '/global/project/projectdirs/act/data/coadds/act_planck_s08_s18_cmb_f150_daynight_srcfree_ivar.fits'
imap = enmap.read_map(ivar_map)


#enplot.show(enplot.plot(amap[0], downgrade=4, colorbar=True))
#plot = enplot.plot(amap[0], downgrade=4, colorbar=True)
#enplot.write("coadd_map", plot)

#posmap = imap.posmap()
#dec = np.rad2deg(posmap[0])
#ra = np.rad2deg(posmap[1])
#print(dec, ra)

# new ACT catalogue yeahhhh
catalogue_name = "data/AdvACT_190220_confirmed.fits"
hdu = fits.open(catalogue_name)
ras = hdu[1].data['RADeg']
decs = hdu[1].data['DECDeg']

N_cluster = len(ras) #3200
N_stamp = []



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
        ivar = reproject.postage_stamp(imap, ras[i], decs[i], stamp_width, 0.5)

        ## therefore this makes a stamp of 120 x 120 arcmin (2 x 2 degree)
        ## consisting of 240 x 240 pixels of width 0.5 arcmins
        ## (each pixel represents half arcmin)

        if stamp is None: continue
        ivar = ivar[0]
        if np.any(ivar <= 0): continue
        ready = stamp[0]

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

        # build a Gaussian beam transfer function
        beam_arcmin = 1.4
        fwhm = np.deg2rad(beam_arcmin/60.)

        # evaluate the beam on an isotropic Fourier grid
        kbeam2d = np.exp(-(fwhm**2.)*(modlmap**2.)/(16.*np.log(2.)))

        ## lensing noise curves require CMB power spectra
        ## this could be from camb theory data or actual map

        # get theory spectrum - this should be the lensed spectrum!
        ells0, dltt = np.loadtxt("data/camb_theory.dat", usecols=(0,1), unpack=True)
        cltt0 = dltt/ells0/(ells0 + 1.)*2.*np.pi

        # bin the power spectrum from given stamp
        edges = np.arange(30,8001,20)
        cents, p1d = maps.binned_power(ready, bin_edges=edges, mask=taper)
        #plt.plot(cents, cents*(cents+1)*p1d/(2.*np.pi), 'k.', marker='o', ms=4, mfc='none');    

        # remove nans in cls (about 10%)
        mask_nan = ~np.isnan(p1d)
        ells = cents[mask_nan]
        cltt = p1d[mask_nan]

        # need to choose a rough range for fitting
        cut1 = np.argmax(ells > 4500)
        cut2 = np.argmax(ells > 5000)

        logx1 = np.log10(ells[:cut1])
        logy1 = np.log10(cltt[:cut1])
        logx2 = np.log10(ells[cut2:])
        logy2 = np.log10(cltt[cut2:])

        def line(x, a, b):
            return a*x + b

        # fit the binned power spectrum to the power law
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

        # to fill in the cls
        clarr = np.zeros(ells.size)
        clarr1 = np.zeros(ells.size)
        clarr2 = np.zeros(ells.size)

        i = 0
        while (i < ells.size):
            clarr1[i] = amp1*(ells[i]**ind1)
            clarr2[i] = amp2*(ells[i]**ind2)
            i += 1

        # find a cross point
        j = 0
        while (j < ells.size):
            if clarr1[j] <= clarr2[j]: break
            j += 1
        cross = j


        i = 0
        while (i < ells.size):
            if i < cross:
                clarr[i] = clarr1[i]
            else:
                clarr[i] = clarr2[i]
            i += 1

        ## interpolate ells and cltt 1D power spectrum specification
        ## isotropically on to the Fourier 2D space grid

        # build interpolated 2D Fourier CMB theory
        ucltt2d = maps.interp(ells0, cltt0)(modlmap)
        tcltt2d = maps.interp(ells, clarr)(modlmap)
        tcltt2d = tcltt2d/(kbeam2d**2.)

        ## need to have a Fourier space mask in hand
        ## that enforces what multipoles in the CMB map are included

        # build a Fourier space mask
        kmask = modlmap*0+1
        kmask[modlmap < 300] = 0
        kmask[modlmap > 7000] = 0

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

        # for mean field stacking counts
        if N_stamp is not None and count == N_stamp: break

    # averging the stack of stamps
    stack /= count

    plt.imshow(stack); plt.show()

    elapsed_stack = time.time() - start_stack
    print("\r ::: stacking took %.1f seconds " % elapsed_stack)

    return(stack, count, stamp_width)

# choose the input map ***
input_map = amap


# staking at cluster positions
field, N_stamp, stamp_width = stacking(N_cluster, input_map, ras, decs)
print("\r ::: number of stamps of clusters : %d " % N_stamp)


# to find random positions in the map
dec0, dec1, ra0, ra1 = [-62.0, 22.0, -180.0, 180.0]
width = stamp_width/60.


# iterations for mean field
N_iterations = 10
meanf = np.zeros(field.shape)

# in case that a stamp is not created in a given location
N_bigger = N_stamp + 1000

k = 0
# stacking at random positions
while (k < N_iterations):

    rd_decs = np.random.rand(N_bigger)*(dec1 - dec0 - width) + dec0 + width/2.
    rd_ras = np.random.rand(N_bigger)*(ra1 - ra0 - width) + ra0 + width/2.

    rd_field, N_rd_stamp, _ = stacking(N_bigger, input_map, rd_ras, rd_decs)
    print("\r ::: number of stamps of random positions : %d " % N_rd_stamp)

    #enmap.write_fits('test/%d_iter.fits'% (k+1), rd_field)
    #print("\r ::: iteration complete: %d of %d  " % ((k+1), N_iterations)

    meanf += rd_field

    k += 1


# averging over the number of iterations
meanf /= N_iterations

# mean field subtraction
final = field - meanf

elapsed_total = int((time.time() - start_total)/60.0)
print("\r ::: entire run took {} minutes ".format(elapsed_total))

plt.imshow(field); plt.title('stacked lensing reconstruction'); plt.colorbar(); plt.show()
plt.imshow(meanf); plt.title('mean field (%d iterations)' % N_iterations); plt.colorbar(); plt.show()
plt.imshow(final); plt.title('after mean field subtraction'); plt.colorbar(); plt.show()
plt.imshow(final[100:140,100:140]); plt.title('zoom'); plt.colorbar(); plt.show()
