import time as t
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject, utils, curvedsky, wcsutils
from orphics import mpi, maps, stats, io, cosmology
import utils as cutils
import sys
import healpy as hp
from symlens import qe
from past.utils import old_div
from scipy.interpolate import *

start = t.time() 

sim_path = "/global/project/projectdirs/act/data/msyriac/dr5_cluster_sims"
my_sim_path = "/global/cscratch1/sd/eunseong/data/webskysim"
another_path = "/global/cscratch1/sd/msyriac/shared/websky"
output_path = "/global/homes/e/eunseong/cmbhalolensing/wsky_outs"

#-------------------------------------------------------------------------------

save_name = "cmbszsub_cmb_hlmax"

which_cat = "act_tsz"
#which_cat = "act_tsz_ksz_cib"
#which_cat = "halo"

test = False
mean_field = True
stack_check = False

freq_sz = 150
#freq_sz = 90

#-------------------------------------------------------------------------------

save_dir = f'{output_path}/{save_name}'
io.mkdir(f'{save_dir}')
print(" ::: saving to ", save_dir)

# frequency for sz and cib 
if freq_sz is 150: freq_cib = 145
elif freq_sz is 90: freq_cib = 93

# simulation setting 
tap_per = 12.0
pad_per = 3.0

fwhm = 1.4
nlevel = 20

def load_beam(freq):
    if freq=='f150': fname = sim_path+'beam_gaussian_la145.txt'
    elif freq=='f090': fname = paths.data+'beam_gaussian_la093.txt'
    ls,bls = np.loadtxt(fname,usecols=[0,1],unpack=True)
    assert ls[0]==0
    bls = bls / bls[0]
    return maps.interp(ls,bls)  
    

# gradient leg filter
xlmin = 200 
xlmax = 2000

# high resolution leg filter 
ylmin = 200
ylmax = 6000
#ylmax = 3500
ylcut = 2

# filter for kappa 
klmin = 200
klmax = 5000
#klmax = 3000

# stamp size and resolution 
px = 0.5
width = 120./60.
maxr = width * utils.degree / 2.0

# for binned kappa profile
arcmax = 15.0
arcstep = 1.5
bin_edges = np.arange(0, arcmax, arcstep)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret 

# reading catalogue ------------------------------------------------------------

if which_cat is "halo":

    cat = "halos.pksc"

    omegab = 0.049
    omegac = 0.261
    omegam = omegab + omegac
    h      = 0.68
    ns     = 0.965
    sigma8 = 0.81

    c = 3e5

    H0 = 100*h
    nz = 100000
    z1 = 0.0
    z2 = 6.0
    za = np.linspace(z1,z2,nz)
    dz = za[1]-za[0]

    H      = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)
    dchidz = lambda z: c/H(z)

    chia = np.cumsum(dchidz(za))*dz

    zofchi = interp1d(chia,za)    
    
    rho    = 2.775e11 * omegam * h**2 # Msun/Mpc^3

    # load the entire halo catalogue 
    f = open(f'{my_sim_path}/halos.pksc')
    N = np.fromfile(f, count=3, dtype=np.int32)[0]
    catalog=np.fromfile(f, count=N*10, dtype=np.float32)
    catalog=np.reshape(catalog, (N,10))

    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
    R  = catalog[:,6] # Mpc

    # convert to mass, RA and DEc
    M200m       = 4*np.pi/3.*rho*R**3  # this is M200m (mean density 200 times mean) in Msun 
    theta, phi  = hp.vec2ang(np.column_stack((x,y,z))) # in radians
    chi         = np.sqrt(x**2+y**2+z**2)    # Mpc
    
    mass     = M200m/1e14                 
    ras      = np.rad2deg(phi)
    decs     = np.rad2deg(np.pi/2. - theta)
    redshift = zofchi(chi)

    print(" ::: total number of halos = ", N) # 862923142


elif which_cat is "act_tsz":

    # load Matt's catalogue 
    cat = f'{sim_path}/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits' # 11902
    hdu = fits.open(cat)
    #print(hdu[1].columns)
    ras = hdu[1].data['RADeg']
    decs = hdu[1].data['DECDeg']
    mass = hdu[1].data['M500c'] # 1e14 Msun
    snr = hdu[1].data['SNR']
    zs = hdu[1].data['redshift'] 
    
elif which_cat is "act_tsz_ksz_cib":

    cat = f'{sim_path}/ACTSim_CMB-T_cmb-tsz-ksz-cib_MFMF_pass2_mass.fits' # 7943
    hdu = fits.open(cat)
    #print(hdu[1].columns)
    ras = hdu[1].data['RADeg']
    decs = hdu[1].data['DECDeg']
    mass = hdu[1].data['M500c'] # 1e14 Msun
    snr = hdu[1].data['SNR']
    zs = hdu[1].data['redshift']


if mean_field is True: 
    # load random catalogue 
    if which_cat is "act_tsz": 
        cat = f'{my_sim_path}/webskysim_ACT_cmb_tsz_randoms.txt'
    elif which_cat is "act_tsz_ksz_cib": 
        cat = f'{my_sim_path}/webskysim_ACT_cmb_tsz_ksz_cib_randoms.txt'
        
    ras, decs = np.loadtxt(cat, unpack=True)
    print(" ::: this is a mean-field run")

print(" ::: name of catalogue = ", cat)
print(" ::: total number of clusters = ", len(ras)) 


# reading maps -----------------------------------------------------------------

# read tSZ cluster model image map by Matt Hilton
if which_cat is "act_tsz":
    imap = f'{sim_path}/model_MFMF_pass2_cmb-tsz_f150.fits'
elif which_cat is "act_tsz_ksz_cib":
    imap = f'{sim_path}/model_MFMF_pass2_cmb-tsz-ksz-cib_f150.fits'
    
modelmap = enmap.read_map(imap)    
print(" ::: reading tsz cluster model image map :", imap) 

if mean_field is True:
    fshape, fwcs = enmap.fullsky_geometry(res=px*utils.arcmin, proj='car') #(21601, 43200)
else: 
    fshape = modelmap.shape  #(10320, 43200)
    fwcs = modelmap.wcs 

# reading kappa map 
imap = f'{my_sim_path}/kap_lt4.5.fits'
kmap = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=12000, rot=None)[0]
print(" ::: reading true kappa map :", imap) 

# reading lensed alm 
imap = f'{my_sim_path}/lensed_alm.fits'
alm = np.complex128(hp.read_alm(imap, hdu=(1, 2, 3)))
lmap = curvedsky.alm2map(alm, enmap.empty(fshape, fwcs, dtype=np.float32))
print(" ::: reading lensed alm map and converting to lensed cmb map :", imap) 

# reading tsz map
imap = f'{my_sim_path}/tsz.fits'
ymap = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=12000, rot=None)[0]

# reading high resolution tsz map
#imap = f'{another_path}/tsz_8192_alm.fits'
#alm = np.complex128(hp.read_alm(imap, 1))
#ymap = curvedsky.alm2map(alm, enmap.empty(fshape, fwcs, dtype=np.float32))
#print(" ::: reading high resolution tsz map") 

# convert compton-y to delta-T (in uK) 
tcmb = 2.726
tcmb_uK = tcmb * 1e6 #micro-Kelvin
H_cgs = 6.62608e-27
K_cgs = 1.3806488e-16

def fnu(nu):
    """
    nu in GHz
    tcmb in Kelvin
    """
    mu = H_cgs*(1e9*nu)/(K_cgs*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

tszmap = fnu(freq_sz) * ymap * tcmb_uK
print(" ::: reading ymap and convering to tsz map at %d GHz :" %freq_sz, imap) 

if which_cat is "act_tsz_ksz_cib":
    # reading ksz map
    imap = f'{my_sim_path}/ksz.fits'
    kszmap = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=12000, rot=None)[0]
    print(" ::: reading ksz map :", imap) 
    #print(kszmap.shape) #(10320, 43200)

    if freq_sz is 90:
        # reading cib_93_GHz map
        imap = f'{my_sim_path}/cib_nu0093.fits'
        cibmap_i = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=12000, rot=None)[0]

    elif freq_sz is 150: 
        # reading cib_145_GHz map
        imap = f'{my_sim_path}/cib_nu0145.fits'
        cibmap_i = reproject.enmap_from_healpix(imap, fshape, fwcs, ncomp=1, unit=1, lmax=12000, rot=None)[0]
    
    print(" ::: reading cib map at %d GHz :" %freq_cib, imap) 
    #print(cibmap_i.shape) #(10320, 43200)

    kboltz = 1.3806503e-23 #MKS
    hplanck = 6.626068e-34 #MKS
    clight = 299792458.0 #MKS

    def ItoDeltaT(nu):
        # conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
        #   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
        #   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
        nu *= 1e9
        X = hplanck*nu/(kboltz*tcmb)
        dBnudT = (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/tcmb_uK * 1e26
        return 1./dBnudT

    cibmap = ItoDeltaT(freq_cib) * cibmap_i 

    tsz_ksz_cib_map = lmap + tszmap + kszmap + cibmap 
    ksz_cib_map = lmap + kszmap + cibmap 
 
    # FIXME : for some reason model image subtraction at map level doesn't work for the mean-field
    #         model image subtraction for the mean-field is done at stamp level for now   
    if mean_field is not True:
        tsz_ksz_cib_msub_map = tsz_ksz_cib_map - modelmap

tsz_map = lmap + tszmap

# FIXME : for some reason model image subtraction at map level doesn't work for the mean-field
#         model image subtraction for the mean-field is done at stamp level for now 
if mean_field is not True:
    tsz_msub_map = tsz_map - modelmap

#-------------------------------------------------------------------------------

if test is True: nsims = 10 # just for a quick check 
else: nsims = len(ras)

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

j = 0  # local counter for this MPI task
for task in my_tasks:
    i = task
    cper = int((j + 1) / len(my_tasks) * 100.0)
    if rank==0: print(f"Rank {rank} performing task {task} as index {j} ({cper}% complete.).")
      
    coords = np.array([decs[i], ras[i]]) * utils.degree  
   
    # cut out a stamp from the simulated map (CAR -> plain)
    kstamp = reproject.thumbnails(
        kmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False # only needed for maps made natively in CAR
    ) # true kappa stamp for the reference
 
    lstamp = reproject.thumbnails(
        lmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed alm
 
    tsz_stamp = reproject.thumbnails(
        tsz_ksz_cib_map if which_cat is "act_tsz_ksz_cib" else tsz_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed alm + tsz (or + ksz + cib) 
 
    if which_cat is "act_tsz_ksz_cib":    
        ksz_cib_stamp = reproject.thumbnails(
            ksz_cib_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed alm + ksz + cib
 
    model_stamp = reproject.thumbnails(
        modelmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # tsz cluster model image 90 or 150 GHz 

    if mean_field is not True:
        tsz_msub_stamp = reproject.thumbnails(
            tsz_ksz_cib_msub_map if which_cat is "act_tsz_ksz_cib" else tsz_msub_map,
            coords,
            r=maxr,
            res=px * utils.arcmin,
            proj="tan",
            oversample=2,
            pixwin=False
        ) # lensed alm + tsz (or + ksz + cib) and model image subtraction   
    else:
        tsz_msub_stamp = tsz_stamp - model_stamp        

    hres = tsz_msub_stamp    
    grad = lstamp    

    # initialise calculations based on geometry 
    if j == 0:     
        # get geometry and Fourier info   
        shape = lstamp.shape
        wcs = lstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        modrmap = enmap.modrmap(shape, wcs)
        
        #assert wcsutils.equal(lstamp.wcs, kstamp.wcs)

        # get an edge taper map and apodize
        taper = maps.get_taper(
            lstamp.shape,
            lstamp.wcs,
            taper_percent=tap_per,
            pad_percent=pad_per,
            weight=None,
        )
        taper = taper[0]  

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid 
        bfunc150 = cutils.load_beam("f150") # not used actually
        bfunc90 = cutils.load_beam("f090")  
        
        beam2d_150 = bfunc150(modlmap)
        beam2d_90 = bfunc90(modlmap)
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

    # same filter as the post-reconstuction for true kappa
    k_stamp = maps.filter_map(kstamp, kmask)   
    s.add_to_stack('kstamp', k_stamp)
      
    binned_true = bin(k_stamp, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("tk1d", binned_true)     

    # same filter as hres leg for cmb stack 
    hres_stamp = maps.filter_map(hres, ymask)   
    grad_stamp = maps.filter_map(grad, ymask)  

    s.add_to_stack('hres_stamp_st', hres_stamp)
    s.add_to_stack('grad_stamp_st', grad_stamp)   

    if stack_check is True:
        j = j + 1
        continue

    # taper stamp
    tapered_hres = hres * taper 
    tapered_grad = grad * taper 
       
    # apply beam     
    #l_stamp = maps.filter_map(tapered_lstamp, beam2d)
    
    # get a beam deconvolved Fourier stamp
    #k_map = enmap.fft(l_stamp, normalize="phys")/beam2d
    
    # get a Fourier transformed stamp
    k_hres_map = enmap.fft(tapered_hres, normalize="phys") 
    k_grad_map = enmap.fft(tapered_grad, normalize="phys")

    assert np.all(np.isfinite(k_hres_map))            
    assert np.all(np.isfinite(k_grad_map)) 
    
    # build symlens dictionary 
    feed_dict = {
        'uC_T_T' : ucltt2d, 
        'tC_T_T' : tcltt2d, 
        'X' : k_grad_map, # grad leg
        'Y' : k_hres_map, # hres leg
    }  

    # do lensing reconstruction in Fourier space    
    rkmap = qe.reconstruct(shape, wcs, feed_dict, estimator='hdv', XY="TT", xmask=xmask, ymask=ymask, kmask=kmask, physical_units=True)
      
    assert np.all(np.isfinite(rkmap))
        
    # transform to real space
    kappa = enmap.ifft(rkmap, normalize='phys').real    

    # stack reconstructed kappa     
    s.add_to_stack("lstamp", kappa)
    
    binned_kappa = bin(kappa, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("k1d", binned_kappa) 
    
    j = j + 1

#-------------------------------------------------------------------------------
            
s.get_stacks()
s.get_stats()
    
if rank==0:

    if mean_field is True:
        save_name = save_name + "_mf" 

    hres_st = s.stacks['hres_stamp_st']
    grad_st = s.stacks['grad_stamp_st']     
 
    hres_zoom = hres_st[100:140,100:140]  
    grad_zoom = grad_st[100:140,100:140] 
         
    modrmap = hres_zoom.modrmap()
    modrmap = np.rad2deg(modrmap)*60. 
    
    hbinned = bin(hres_zoom, modrmap, bin_edges)
    gbinned = bin(grad_zoom, modrmap, bin_edges)

    io.plot_img(hres_st, f'{save_dir}/{save_name}_00hres.png')  
    io.plot_img(grad_st, f'{save_dir}/{save_name}_01grad.png')             
    io.plot_img(hres_zoom, f'{save_dir}/{save_name}_00hres_zoom.png')   
    io.plot_img(grad_zoom, f'{save_dir}/{save_name}_01grad_zoom.png')  
    np.save(f'{save_dir}/{save_name}_00hres.npy', hres_st) 
    np.save(f'{save_dir}/{save_name}_01grad.npy', grad_st)  
    io.save_cols(f'{save_dir}/{save_name}_00binned_hres.txt', (centers, hbinned))   
    io.save_cols(f'{save_dir}/{save_name}_01binned_grad.txt', (centers, gbinned))      
    
    if stack_check is not True:
  
        kmap = s.stacks['kstamp']
        lmap = s.stacks['lstamp']
        
        kmap_zoom = kmap[100:140,100:140] 
        lmap_zoom = lmap[100:140,100:140] 
                
        modrmap = kmap_zoom.modrmap()
        modrmap = np.rad2deg(modrmap)*60. 
        
        kbinned = bin(kmap_zoom, modrmap, bin_edges)
        lbinned = bin(lmap_zoom, modrmap, bin_edges)
 
        io.plot_img(kmap, f'{save_dir}/{save_name}_10tkappa.png')   
        io.plot_img(lmap, f'{save_dir}/{save_name}_11rkappa.png')                 
        io.plot_img(kmap[100:140,100:140], f'{save_dir}/{save_name}_10tkappa_zoom.png')   
        io.plot_img(lmap[100:140,100:140], f'{save_dir}/{save_name}_11rkappa_zoom.png')  
        np.save(f'{save_dir}/{save_name}_10tkappa.npy', kmap)     
        np.save(f'{save_dir}/{save_name}_11rkappa.npy', lmap)                 
        io.save_cols(f'{save_dir}/{save_name}_10binned_tkappa_from2D.txt', (centers, kbinned))
        io.save_cols(f'{save_dir}/{save_name}_11binned_rkappa_from2D.txt', (centers, lbinned))
 
        tbinned = s.stats['tk1d']['mean']
        tcovm = s.stats['tk1d']['covmean']
        tcorr = stats.cov2corr(s.stats['tk1d']['covmean'])
        terrs = s.stats['tk1d']['errmean']
        
        binned = s.stats['k1d']['mean']
        covm = s.stats['k1d']['covmean']
        corr = stats.cov2corr(s.stats['k1d']['covmean'])
        errs = s.stats['k1d']['errmean']
                     
        np.savetxt(f'{save_dir}/{save_name}_10binned_tkappa.txt', tbinned)       
        np.savetxt(f'{save_dir}/{save_name}_11binned_rkappa.txt', binned)
        np.savetxt(f'{save_dir}/{save_name}_10tkappa_errs.txt', terrs)               
        np.savetxt(f'{save_dir}/{save_name}_11rkappa_errs.txt', errs)    
        np.savetxt(f'{save_dir}/{save_name}_10tkappa_covm.txt', tcovm)
        np.savetxt(f'{save_dir}/{save_name}_11rkappa_covm.txt', covm)
        np.save(f'{save_dir}/{save_name}_10tkappa_corr.npy', tcorr)  
        np.save(f'{save_dir}/{save_name}_11rkappa_corr.npy', corr) 


    if test is True:
 
        plt.plot(centers, hbinned, 'ko-', label='hres')
        plt.plot(centers, gbinned, 'bo-', label='grad')       
        plt.title('binned profile before reconstruction')
        plt.xlabel('$\\theta$ [arcmin]')
        plt.ylabel('cmb temperature $[uK]$')
        plt.axhline(y=0,linewidth=2.,color='grey',alpha=0.2)
        plt.legend(loc="best")
        plt.show()
        plt.clf() 
    
        if not stack_check is True:
        
            plt.plot(centers, kbinned, 'ko-', label='true kappa')
            plt.plot(centers, lbinned, 'ro-', label='reconstructed kappa')
            plt.title('binned kappa profile')
            plt.xlabel('$\\theta$ [arcmin]')
            plt.ylabel('$\kappa$')
            plt.axhline(y=0,linewidth=2.,color='grey',alpha=0.2)
            plt.legend(loc="best")
            plt.show()
            plt.clf()  

elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)
