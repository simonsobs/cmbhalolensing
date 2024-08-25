import time as t
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell import enmap, reproject, utils, wcsutils, bunch
from orphics import mpi, maps, stats, io, cosmology
from symlens import qe
import argparse

start = t.time()

paths = bunch.Bunch(io.config_from_yaml("input/sim_data.yml"))
print("Paths: ", paths)


# RUN SETTING ------------------------------------------------------------------

parser = argparse.ArgumentParser() 
parser.add_argument(
    "save_name", type=str, help="Name you want for your output."
)
parser.add_argument(
    "which_sim", type=str, help="Choose the sim e.g. websky or sehgal or agora."
)
parser.add_argument(
    "which_cat", type=str, help="Choose the catalogue type e.g. halo or tsz."
)
parser.add_argument(
    "hres_choice", type=str, help="Choose the map for high resolution leg e.g. cmb, cmb_tsz, etc."
)
parser.add_argument(
    "grad_choice", type=str, help="Choose the map for gradient leg e.g. cmb, cmb_tsz, etc."
)
parser.add_argument(
    "--is-observed", action="store_true", help="Beam and noise have been applied to the map."
)
parser.add_argument(
    "--is-meanfield", action="store_true", help="This is a mean-field run."
)
parser.add_argument(
    "--is-test", action="store_true", help="This is a test run for first 10 entries."
)
parser.add_argument(
    "--full-sample", action="store_true", help="Entire sample will be used (SNR > 4)."
)
parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
)
args = parser.parse_args() 

if args.which_sim == "websky": output_path = paths.websky_output_path
elif args.which_sim == "sehgal": output_path = paths.sehgal_output_path
elif args.which_sim == "agora": output_path = paths.agora_output_path
# output_path = paths.{args.which_sim}_output_path

save_name = args.save_name
save_dir = f"{output_path}/{save_name}"
io.mkdir(f"{save_dir}")
print(" ::: saving to", save_dir)

simsuite_path = f"{paths.simsuite_path}/{args.which_sim}_{paths.simsuite_version}/"
# cat_path = f"{paths.cat_path}/{paths.nemosim_version}/"
cat = paths.cmass_cat
print(" ::: catalogue:", args.which_sim, args.which_cat, "[ simsuite ver:", paths.simsuite_version, "]")
# print(" ::: catalogue:", args.which_sim, args.which_cat, "[ simsuite ver:", paths.simsuite_version, "/ nemosim ver:", paths.nemosim_version, "]")
print(" ::: hres:", args.hres_choice, "/ grad:", args.grad_choice)

if args.is_meanfield: print(" ::: this is a mean-field run")


# SIM SETTING ------------------------------------------------------------------

xlmin = 200 
xlmax = 2000

ylmin = 200
ylmax = 3500  # low lmax cut (high lmax cut is 6000)
ylcut = 2

klmin = 200
klmax = 3000  # low lmax cut (high lmax cut is 5000)
print(" ::: hlmax =", ylmax, "and klmax =", klmax)

tap_per = 12.0
pad_per = 3.0

fwhm = 1.5
nlevel = 15.0   

px = 0.5
width_deg = 120./60.
maxr = width_deg * utils.degree / 2.0

# for radially binned kappa profile 
arcmax = 15.0
arcstep = 1.5
bin_edges = np.arange(0, arcmax, arcstep)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret 


# READING CATALOGUE ------------------------------------------------------------

if args.which_cat == "halo":
    if not args.full_sample:
        cat = cat_path + f"{args.which_sim}_halo_snr5p5.txt"
    else:
        cat = cat_path + f"{args.which_sim}_halo.txt"
    ras, decs, zs, masses = np.loadtxt(cat, unpack=True)

elif args.which_cat == "tsz":
    if args.which_sim == "websky":
        cat = paths.websky_tsz_cat
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        masses = hdu[1].data["M200m"] # 1e14 Msun: websky halo cat only provides M200m
        snr = hdu[1].data["SNR"]
        zs = hdu[1].data["redshift"] 
    elif args.which_sim == "sehgal" or args.which_sim == "agora":    
        cat = paths.sehgal_tsz_cat
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        masses = hdu[1].data["M200c"] # 1e14 Msun
        snr = hdu[1].data["SNR"]
        zs = hdu[1].data["redshift"] 

    # SNR cut corresponding to cosmo sample is 5.5
    if not args.full_sample:
        snr_cut = 5.5
        keep = snr > snr_cut
        ras = ras[keep]
        decs = decs[keep]
        masses = masses[keep]
        zs = zs[keep]
        snr = snr[keep]

    print(" ::: min and max SNR = %.2f and %.2f" %(snr.min(), snr.max()))

elif args.which_cat == "cmass":
    if args.which_sim == "websky":
        cat = paths.websky_cmass_cat
        ras, decs, zs = np.loadtxt(cat, unpack=True)
        dec_cut = np.where(np.logical_and(decs<10, decs>-10))
        ras = ras[dec_cut]
        decs = decs[dec_cut]
        zs = zs[dec_cut]

        mf_cat = paths.scratch + "websky_cmass_10x_randoms.txt"

    elif args.which_sim == "agora":
        cat = paths.agora_cmass_cat
        data = np.load(cat).item()
        dec_cut = np.where(np.logical_and(data['dec']<10, data['dec']>-10))[0]
        ras = data['ra'][dec_cut]
        decs = data['dec'][dec_cut]
        zs = data['z'][dec_cut]
        masses = data['M200c'][dec_cut]

        mf_cat = paths.scratch + "agora_cmass_10x_randoms.txt"
        
        print(" ::: min and max M200 = %.2f and %.2f" %(masses.min(), masses.max()), "(mean = %.2f)" %masses.mean())

print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 

if args.is_meanfield:

    Nx = 10 * len(ras)

    # load random catalogue - created by mapcat.py + randcat.py
    # cat = cat_path + f"{args.which_sim}_{args.which_cat}_randoms.txt"
    cat = mf_cat
    ras, decs,_,_ = np.loadtxt(cat)    

    if not args.full_sample:
        ras = ras[:Nx]
        decs = decs[:Nx]

    print(" ::: reading random catalogue for mean-field run")

print(" ::: name of catalogue =", cat)
print(" ::: total number of clusters for stacking =", len(ras)) 




# READING MAPS -----------------------------------------------------------------

if args.which_sim == "websky":
    if args.high_accuracy:
        true = paths.websky_sim_path + paths.websky_kappa4p5_reproj
    else:
        true = paths.websky_sim_path + paths.websky_kappa_reproj
elif args.which_sim == "sehgal":
    true = paths.sehgal_sim_path + paths.sehgal_kappa_reproj
elif args.which_sim == "agora":
    true = paths.agora_sim_path + paths.agora_kappa_reproj

if not args.is_observed: 
    print(" ::: preparing for SIGNAL maps")

    cmb = f"{simsuite_path}scmb.fits"
    cmb_cib = f"{simsuite_path}scmb_cib.fits"
    cmb_ksz_cib = f"{simsuite_path}scmb_ksz_cib.fits"

    cmb_tsz = f"{simsuite_path}scmb_tsz.fits"
    cmb_tsz_cib = f"{simsuite_path}scmb_tsz_cib.fits"
    cmb_tsz_ksz_cib = f"{simsuite_path}scmb_tsz_ksz_cib.fits"
    
else: 
    print(" ::: preparing for OBSERVED maps")

    cmb = f"{simsuite_path}ocmb.fits"
    cmb_cib = f"{simsuite_path}ocmb_cib.fits"
    cmb_ksz_cib = f"{simsuite_path}ocmb_ksz_cib.fits"

    cmb_tsz = f"{simsuite_path}ocmb_tsz.fits"
    cmb_tsz_cib = f"{simsuite_path}ocmb_tsz_cib.fits" 
    cmb_tsz_ksz_cib = f"{simsuite_path}ocmb_tsz_ksz_cib.fits"    

print(" ::: reading true kappa map:", true)
print(" ::: reading lensed cmb map:", cmb)
print(" ::: reading lensed cmb + cib map:", cmb_cib)
print(" ::: reading lensed cmb + ksz + cib map:", cmb_ksz_cib)
print(" ::: reading lensed cmb + tsz map:", cmb_tsz)
print(" ::: reading lensed cmb + tsz + cib map:", cmb_tsz_cib)
print(" ::: reading lensed cmb + tsz + ksz + cib map:", cmb_tsz_ksz_cib)

true_map = enmap.read_map(true, delayed=False)
cmb_map = enmap.read_map(cmb, delayed=False)
cmb_cib_map = enmap.read_map(cmb_cib, delayed=False)
cmb_ksz_cib_map = enmap.read_map(cmb_ksz_cib, delayed=False)
cmb_tsz_map = enmap.read_map(cmb_tsz, delayed=False)
cmb_tsz_cib_map = enmap.read_map(cmb_tsz_cib, delayed=False)
cmb_tsz_ksz_cib_map = enmap.read_map(cmb_tsz_ksz_cib, delayed=False)

print(" ::: maps are ready!")



# LOOP OVER ASSIGNED TASKS -----------------------------------------------------

if args.is_test: 
    print(" ::: this is a test run!")
    nsims = 10
else: nsims = len(ras)

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

j = 0  # local counter for this MPI task
for task in my_tasks:
    i = task
    cper = int((j + 1) / len(my_tasks) * 100.0)
    if rank==0: print(f"Rank {rank} performing task {task} as index {j} ({cper}% complete.).")
      
    coords = np.array([decs[i], ras[i]]) * utils.degree  

    # cut out a stamp from the simulated map
    kstamp = reproject.thumbnails(
        true_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # true kappa

    cmb = reproject.thumbnails(
        cmb_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True if args.high_accuracy else False # only True for maps made natively in CAR
    ) # lensed cmb

    cmb_cib = reproject.thumbnails(
        cmb_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + cib

    cmb_ksz_cib = reproject.thumbnails(
        cmb_ksz_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + ksz + cib

    cmb_tsz = reproject.thumbnails(
        cmb_tsz_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz

    cmb_tsz_cib = reproject.thumbnails(
        cmb_tsz_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz + cib

    cmb_tsz_ksz_cib = reproject.thumbnails(
        cmb_tsz_ksz_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz + ksz + cib

    # initialise calculations based on geometry 
    if j == 0:     

        # get geometry and Fourier info   
        shape = kstamp.shape
        wcs = kstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        modrmap = enmap.modrmap(shape, wcs)
        
        assert wcsutils.equal(kstamp.wcs, cmb.wcs)
        assert wcsutils.equal(kstamp.wcs, cmb_tsz.wcs)

        # get an edge taper map and apodize
        taper = maps.get_taper(
            kstamp.shape,
            kstamp.wcs,
            taper_percent=tap_per,
            pad_percent=pad_per,
            weight=None,
        )
        taper = taper[0]  

        # evaluate the 2D Gaussian beam on an isotropic Fourier grid
        beam2d = maps.gauss_beam(modlmap, fwhm)    

        # build Fourier space masks for lensing reconstruction
        xmask = maps.mask_kspace(shape, wcs, lmin=xlmin, lmax=xlmax) # grad
        ymask = maps.mask_kspace(shape, wcs, lmin=ylmin, lmax=ylmax, lxcut=ylcut, lycut=ylcut) # hres
        kmask = maps.mask_kspace(shape, wcs, lmin=klmin, lmax=klmax) # kappa

        # get theory spectrum and build interpolated 2D Fourier CMB from theory and maps      
        theory = cosmology.default_theory()
        ucltt2d = theory.lCl("TT", modlmap)

        # total spectrum includes beam-deconvolved noise       
        npower = (nlevel * np.pi/180./60.)**2.
        tcltt2d = ucltt2d + npower/beam2d**2.
        tcltt2d[~np.isfinite(tcltt2d)] = 0

    # same filter as the post-reconstuction for true kappa
    k_stamp = maps.filter_map(kstamp, kmask)   
    s.add_to_stack("kstamp", k_stamp)
    binned_true = bin(k_stamp, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("tk1d", binned_true)     

    # choose the map for each leg 
    hres = globals()[args.hres_choice]
    grad = globals()[args.grad_choice]

    # taper stamp
    tapered_hres = hres * taper
    tapered_grad = grad * taper  

    if not args.is_observed:
        # get a Fourier transformed stamp 
        k_hres = enmap.fft(tapered_hres, normalize="phys") 
        k_grad = enmap.fft(tapered_grad, normalize="phys")
    else:
        # get a beam-deconvolved Fourier transformed stamp
        k_hres = enmap.fft(tapered_hres, normalize="phys")/beam2d
        k_grad = enmap.fft(tapered_grad, normalize="phys")/beam2d

    assert np.all(np.isfinite(k_hres))            
    assert np.all(np.isfinite(k_grad)) 

    # build symlens dictionary 
    feed_dict = {
        "uC_T_T" : ucltt2d, 
        "tC_T_T" : tcltt2d, 
        "X" : k_grad, # grad leg
        "Y" : k_hres, # hres leg
    }  

    # do lensing reconstruction in Fourier space    
    rkmap = qe.reconstruct(shape, wcs, feed_dict, estimator="hdv", XY="TT", xmask=xmask, ymask=ymask, kmask=kmask, physical_units=True)
      
    assert np.all(np.isfinite(rkmap))
        
    # transform to real space
    kappa = enmap.ifft(rkmap, normalize="phys").real    

    # stack reconstructed kappa     
    s.add_to_stack("lstamp", kappa)    
    binned_kappa = bin(kappa, modrmap * (180 * 60 / np.pi), bin_edges)   
    s.add_to_stats("k1d", binned_kappa) 

    # stack stamps pre-reconstruction as well (same filter as hres leg)
    hres_stamp = maps.filter_map(hres, ymask) 
    grad_stamp = maps.filter_map(grad, ymask)  
    s.add_to_stack("hres_stamp_st", hres_stamp)
    s.add_to_stack("grad_stamp_st", grad_stamp) 

    # save the list of masses and redshifts for matched stack mass fitting
    if not args.is_meanfield:     
        s.add_to_stats("redshift", (zs[i],))
        # s.add_to_stats("masses", (masses[i],))

    j = j + 1


            
# COLLECT FROM ALL MPI CORES AND CALCULATE STACKS ------------------------------

s.get_stacks()
s.get_stats()
   
if rank==0:

    if args.is_meanfield:
        save_name = save_name + "_mf" 

    # stacks before lensing reconstruction   
    hres_st = s.stacks["hres_stamp_st"]
    grad_st = s.stacks["grad_stamp_st"]     
 
    hres_zoom = hres_st[100:140,100:140]  
    grad_zoom = grad_st[100:140,100:140] 
         
    modrmap = hres_zoom.modrmap()
    modrmap = np.rad2deg(modrmap)*60. 
    
    hbinned = bin(hres_zoom, modrmap, bin_edges)
    gbinned = bin(grad_zoom, modrmap, bin_edges)

    io.plot_img(hres_st, f"{save_dir}/{save_name}_0hres.png")  
    io.plot_img(grad_st, f"{save_dir}/{save_name}_0grad.png")             
    io.plot_img(hres_zoom, f"{save_dir}/{save_name}_0hres_zoom.png")   
    io.plot_img(grad_zoom, f"{save_dir}/{save_name}_0grad_zoom.png")  
    np.save(f"{save_dir}/{save_name}_0hres.npy", hres_st) 
    np.save(f"{save_dir}/{save_name}_0grad.npy", grad_st)  
    io.save_cols(f"{save_dir}/{save_name}_0binned_hres.txt", (centers, hbinned))   
    io.save_cols(f"{save_dir}/{save_name}_0binned_grad.txt", (centers, gbinned))      

    # reconstructed lensing field     
    kmap = s.stacks["kstamp"]
    lmap = s.stacks["lstamp"]
    
    kmap_zoom = kmap[100:140,100:140] 
    lmap_zoom = lmap[100:140,100:140] 
            
    modrmap = kmap_zoom.modrmap()
    modrmap = np.rad2deg(modrmap)*60. 

    kbinned = bin(kmap_zoom, modrmap, bin_edges)
    lbinned = bin(lmap_zoom, modrmap, bin_edges)

    io.plot_img(kmap, f"{save_dir}/{save_name}_1tkappa.png")   
    io.plot_img(lmap, f"{save_dir}/{save_name}_1rkappa.png")                 
    io.plot_img(kmap[100:140,100:140], f"{save_dir}/{save_name}_1tkappa_zoom.png")   
    io.plot_img(lmap[100:140,100:140], f"{save_dir}/{save_name}_1rkappa_zoom.png")  
    np.save(f"{save_dir}/{save_name}_1tkappa.npy", kmap)     
    np.save(f"{save_dir}/{save_name}_1rkappa.npy", lmap)                 
    io.save_cols(f"{save_dir}/{save_name}_1binned_tkappa_from2D.txt", (centers, kbinned))
    io.save_cols(f"{save_dir}/{save_name}_1binned_rkappa_from2D.txt", (centers, lbinned))

    tbinned = s.stats["tk1d"]["mean"]
    tcovm = s.stats["tk1d"]["covmean"]
    tcorr = stats.cov2corr(s.stats["tk1d"]["covmean"])
    terrs = s.stats["tk1d"]["errmean"]
    
    binned = s.stats["k1d"]["mean"]
    covm = s.stats["k1d"]["covmean"]
    corr = stats.cov2corr(s.stats["k1d"]["covmean"])
    errs = s.stats["k1d"]["errmean"]

    np.savetxt(f"{save_dir}/{save_name}_1tkappa_errs.txt", terrs)               
    np.savetxt(f"{save_dir}/{save_name}_1rkappa_errs.txt", errs)    
    np.savetxt(f"{save_dir}/{save_name}_1tkappa_covm.txt", tcovm)
    np.savetxt(f"{save_dir}/{save_name}_1rkappa_covm.txt", covm)
    np.save(f"{save_dir}/{save_name}_1tkappa_corr.npy", tcorr)  
    np.save(f"{save_dir}/{save_name}_1rkappa_corr.npy", corr) 
    io.save_cols(f"{save_dir}/{save_name}_1binned_tkappa.txt", (centers, tbinned))
    io.save_cols(f"{save_dir}/{save_name}_1binned_rkappa.txt", (centers, binned)) 

    enmap.write_map(f"{save_dir}/{save_name}_kmask.fits", kmask)   
    np.savetxt(f"{save_dir}/{save_name}_bin_edges.txt", bin_edges)

    # if not args.is_meanfield:
    #     np.savetxt(f"{save_dir}/{save_name}_z_mass1e14.txt", np.c_[s.vectors["redshift"], s.vectors["masses"]])

elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)
