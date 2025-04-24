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
import sys
import os
sys.path.append(os.path.abspath("/home3/nehajo/scripts/"))
import profiles

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
    "sim_version", type=str, help="Version name of simsuite."
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
    "--is-test", action="store_true", help="This is a test run for first 100 entries."
)
parser.add_argument(
    "--full-sample", action="store_true", help="Entire sample will be used (SNR > 4)."
)
parser.add_argument(
    "--high-accuracy", action="store_true", help="If using a high accuracy websky map, choose this option."
)
parser.add_argument(
    "--debug", action="store_true", help="No kappa reconstruction, only save plots of hres and grad before and after filtering."
)
parser.add_argument(
    "--inpaint", action="store_true", help="Perform gradient inpainting."
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

simsuite_path = f"{paths.simsuite_path}/{args.which_sim}_{args.sim_version}/"
# cat_path = f"{paths.cat_path}/{paths.nemosim_version}/"
# cat = paths.cmass_cat
# print(" ::: catalogue:", args.which_sim, args.which_cat, "[ simsuite ver:", args.sim_version, "]")
# print(" ::: catalogue:", args.which_sim, args.which_cat, "[ simsuite ver:", args.sim_version, "/ nemosim ver:", paths.nemosim_version, "]")
print(" ::: catalogue:", args.which_sim, args.which_cat)
print(" ::: hres:", args.hres_choice, "/ grad:", args.grad_choice)

if args.is_meanfield: print(" ::: this is a mean-field run")


# SIM SETTING ------------------------------------------------------------------

xlmin = 200 
xlmax = 2000

ylmin = 200
ylmax = 6000 #3500  # low lmax cut (high lmax cut is 6000)
ylcut = 2

klmin = 200
klmax = 5000 #3000  # low lmax cut (high lmax cut is 5000)
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
        data = np.load(cat, allow_pickle=True).item()
        ras = data['ra']
        decs = data['dec']
        zs = data['z']
        masses = None

        mf_cat = paths.websky_cmass_randoms

    elif args.which_sim == "agora":
        cat = paths.agora_cmass_cat
        data = np.load(cat, allow_pickle=True).item()
        ras = data['ra']
        decs = data['dec']
        zs = data['z']
        masses = data['M200c']

        mf_cat = paths.agora_cmass_randoms
        
elif args.which_cat == "highmass":
    if args.which_sim == "agora":
        cat = paths.agora_highmass_cat
        data = np.load(cat, allow_pickle=True).item()
        ras = data['ra']
        decs = data['dec']
        zs = data['z']
        masses = data['M200c']
        
if masses is not None: print(" ::: min and max M200 = %.2f and %.2f" %(masses.min(), masses.max()), "(mean = %.2f)" %masses.mean())

print(" ::: min and max redshift = %.2f and %.2f" %(zs.min(), zs.max()), "(mean = %.2f)" %zs.mean()) 

if args.is_meanfield:

    Nx = 10 * len(ras)

    # load random catalogue - created by mapcat.py + randcat.py
    # cat = cat_path + f"{args.which_sim}_{args.which_cat}_randoms.txt"
    cat = mf_cat
    ras, decs = np.loadtxt(cat, unpack=True)    

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

    hres = f"{simsuite_path}s{args.hres_choice}.fits"
    grad = f"{simsuite_path}s{args.grad_choice}.fits"

    
else: 
    print(" ::: preparing for OBSERVED maps")

    hres = f"{simsuite_path}o{args.hres_choice}.fits"
    grad = f"{simsuite_path}o{args.grad_choice}.fits"


print(" ::: reading true kappa map:", true)
print(" ::: reading lensed hres map:", hres)
print(" ::: reading lensed grad map:", grad)

true_map = enmap.read_map(true, delayed=False)
hres_map = enmap.read_map(hres, delayed=False)
grad_map = enmap.read_map(grad, delayed=False)

print(" ::: maps are ready!")



# LOOP OVER ASSIGNED TASKS -----------------------------------------------------

if args.is_test: 
    print(" ::: this is a test run!")
    nsims = 100
else: nsims = len(ras)

comm, rank, my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

tk1ds = []
k1ds = []
zvals = []
if masses is not None: Ms = []
Nobj = 0

j = 0  # local counter for this MPI task
for task in my_tasks:
    task_start = t.time()
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

    hres = reproject.thumbnails(
        hres_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True if args.high_accuracy else False # only True for maps made natively in CAR
    ) # lensed hres

    grad = reproject.thumbnails(
        grad_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True if args.high_accuracy else False # only True for maps made natively in CAR
    ) # lensed grad

    # initialise calculations based on geometry 

    if j == 0:     

        # get geometry and Fourier info   
        shape = kstamp.shape
        wcs = kstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        modrmap = enmap.modrmap(shape, wcs)
        
        assert wcsutils.equal(kstamp.wcs, hres.wcs)
        assert wcsutils.equal(kstamp.wcs, grad.wcs)

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
    binned_true = bin(k_stamp, modrmap * (180 * 60 / np.pi), bin_edges)   
    tk1ds.append(binned_true.copy())

    # choose the map for each leg 
    # hres = globals()[args.hres_choice]
    # grad = globals()[args.grad_choice]

    # taper stamp
    tapered_hres = hres * taper
    tapered_grad = grad * taper  

    if args.inpaint:
        """ 
        If inpainting, we 
        (1) resample the stamp to 64x64 (2 arcmin pixels)
        (2) Inpaint a hole of radius 7 arcmin 
        """

        rmin = 7*utils.arcmin
        crop_pix = int(16./px)
        cutout = maps.crop_center(tapered_grad, cropy=crop_pix, cropx=crop_pix, sel=False)
        cutout_sel = maps.crop_center(tapered_grad, cropy=crop_pix, cropx=crop_pix, sel=True)
        Ndown, Ndown2 = cutout.shape[-2:]
        if Ndown != Ndown2: raise Exception
        hres_fiducial_rms=20

        if j==0:
            from orphics import pixcov
            pshape = cutout.shape
            pwcs = cutout.wcs
            ipsizemap = enmap.pixsizemap(pshape, pwcs)
            pivar = maps.ivar(pshape, pwcs, hres_fiducial_rms, ipsizemap=ipsizemap)
            beam = lambda x: maps.gauss_beam(x, fwhm)
            pcov = pixcov.tpcov_from_ivar(Ndown, pivar, theory.lCl, beam)
            geo = pixcov.make_geometry(pshape, pwcs, rmin, n=Ndown, deproject=True, iau=False, res=None, pcov=pcov)

        cutout = pixcov.inpaint_stamp(cutout,geo)
        tapered_grad[cutout_sel] = cutout.copy()
        inp_stamp = tapered_grad / taper

        if j==0:
            inpaint_st = inp_stamp.copy()
        else:
            inpaint_st = inpaint_st + inp_stamp

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
    binned_kappa = bin(kappa, modrmap * (180 * 60 / np.pi), bin_edges)   
    k1ds.append(binned_kappa.copy()) 

    # stack stamps pre-reconstruction as well (same filter as hres leg)
    hres_stamp = maps.filter_map(hres, ymask) 
    grad_stamp = maps.filter_map(grad, ymask)  

    # save the list of masses and redshifts for matched stack mass fitting
    if not args.is_meanfield:
        zvals.append(zs[i].copy(),)
        if masses is not None: Ms.append(masses[i].copy(),)
    
    if j == 0:

        hstamps = hres_stamp.copy()
        gstamps = grad_stamp.copy()
        if args.debug:
            hres_pre = hres.copy()
            grad_pre = grad.copy()
        else:
            kstamps = k_stamp.copy()
            lstamps = kappa.copy()

    else:
        hstamps = hstamps + hres_stamp
        gstamps = gstamps + grad_stamp
        if args.debug:
            hres_pre = hres_pre + hres
            grad_pre = grad_pre + grad
        else:
            kstamps = kstamps + k_stamp
            lstamps = lstamps + kappa
    Nobj +=1
    j = j + 1
    task_end = t.time()
    if args.is_test and (rank == 0):
        print(f"Task {task} took {task_end-task_start} s")

print(f"Rank {rank} has {Nobj} total")

            
# COLLECT FROM ALL MPI CORES AND CALCULATE STACKS ------------------------------

hstamps = np.asarray(hstamps)
hres_st = utils.reduce(hstamps, comm, root=0, op=mpi.MPI.SUM)

gstamps = np.asarray(gstamps)
grad_st = utils.reduce(gstamps, comm, root=0, op=mpi.MPI.SUM)

Nobj = utils.reduce(np.asarray([Nobj]), comm, root=0, op=mpi.MPI.SUM)

if args.inpaint:
    istamps = np.asarray(inpaint_st)
    inpaint_st = utils.reduce(inpaint_st, comm, root=0, op=mpi.MPI.SUM)


if not args.debug:
    kstamps = np.asarray(kstamps)
    kmap = utils.reduce(kstamps, comm, root=0, op=mpi.MPI.SUM)

    lstamps = np.asarray(lstamps)
    lmap = utils.reduce(lstamps, comm, root=0, op=mpi.MPI.SUM)
    
    tk1ds = utils.allgatherv(tk1ds, comm)
    k1ds = utils.allgatherv(k1ds, comm)
    zvals = utils.allgatherv(zvals, comm)

else:
    hres_pre = np.asarray(hres_pre)
    hres_pre_st = utils.reduce(hres_pre, comm, root=0, op=mpi.MPI.SUM)

    grad_pre = np.asarray(grad_pre)
    grad_pre_st = utils.reduce(grad_pre, comm, root=0, op=mpi.MPI.SUM)

# Ms = utils.allgatherv(Ms, comm)

if rank==0: #TODO: make stamps enmaps again

    if args.is_meanfield:
        save_name = save_name + "_mf" 
    # s.dump(stats_dir)

    # stacks before lensing reconstruction   

    if not args.debug:
        print(Nobj)
        print(k1ds.shape[0])
        print(tk1ds.shape[0])
        assert k1ds.shape[0] == Nobj[0]
        assert tk1ds.shape[0] == Nobj[0]

    hres_st = enmap.enmap(hres_st, wcs)
    grad_st = enmap.enmap(grad_st, wcs)
    
    hres_st /= Nobj
    grad_st /= Nobj
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

    if args.inpaint:
        inp_st = enmap.enmap(inpaint_st, wcs)/Nobj
        inp_zoom = inp_st[100:140,100:140]
        ibinned = bin(inp_zoom, modrmap, bin_edges)
        io.plot_img(inp_st, f'{save_dir}/{save_name}_02inp.png')              
        io.plot_img(inp_zoom, f'{save_dir}/{save_name}_02inp_zoom.png')  
        np.save(f'{save_dir}/{save_name}_02inp.npy', inp_st)    
        io.save_cols(f'{save_dir}/{save_name}_02binned_inp.txt', (centers, ibinned))

    if not args.debug:
        # reconstructed lensing field     
        kmap = enmap.enmap(kmap, wcs)
        lmap = enmap.enmap(lmap, wcs)
        kmap /= Nobj
        lmap /= Nobj

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

        tbinned = tk1ds.mean(axis=0)
        print("true:", tk1ds.shape, tbinned.shape)
        terrs, tcovm = profiles.errors(tk1ds, Nobj=tk1ds.shape[0])
        tcorr = profiles.correlation_matrix(tcovm)
        
        binned = k1ds.mean(axis=0)
        errs, covm = profiles.errors(k1ds, Nobj=k1ds.shape[0])
        corr = profiles.correlation_matrix(covm)

        np.savetxt(f"{save_dir}/{save_name}_1tkappa_errs.txt", terrs)               
        np.savetxt(f"{save_dir}/{save_name}_1rkappa_errs.txt", errs)    
        np.savetxt(f"{save_dir}/{save_name}_1tkappa_covm.txt", tcovm)
        np.savetxt(f"{save_dir}/{save_name}_1rkappa_covm.txt", covm)
        np.save(f"{save_dir}/{save_name}_1tkappa_corr.npy", tcorr)  
        np.save(f"{save_dir}/{save_name}_1rkappa_corr.npy", corr) 
        io.save_cols(f"{save_dir}/{save_name}_1binned_tkappa.txt", (centers, tbinned))
        io.save_cols(f"{save_dir}/{save_name}_1binned_rkappa.txt", (centers, binned))
        np.savetxt(f"{save_dir}/{save_name}_1binned_tkappa_vectors.txt", tk1ds)
        np.savetxt(f"{save_dir}/{save_name}_1binned_rkappa_vectors.txt", k1ds)

        enmap.write_map(f"{save_dir}/{save_name}_kmask.fits", kmask)   
        np.savetxt(f"{save_dir}/{save_name}_bin_edges.txt", bin_edges)
        enmap.write_map_geometry(f"{save_dir}/{save_name}_map_geometry.fits", shape, wcs)

    else:
        hres_pre_st = enmap.enmap(hres_pre_st, wcs)
        grad_pre_st = enmap.enmap(grad_pre_st, wcs)
        
        hres_pre_st /= Nobj
        grad_pre_st /= Nobj
        hres_pre_zoom = hres_pre_st[100:140,100:140]  
        grad_pre_zoom = grad_pre_st[100:140,100:140] 
            
        modrmap = hres_pre_zoom.modrmap()
        modrmap = np.rad2deg(modrmap)*60. 
        
        hbinned_pre = bin(hres_pre_zoom, modrmap, bin_edges)
        gbinned_pre = bin(grad_pre_zoom, modrmap, bin_edges)

        io.plot_img(hres_pre_st, f"{save_dir}/{save_name}_0hres_unfiltered.png")  
        io.plot_img(grad_pre_st, f"{save_dir}/{save_name}_0grad_unfiltered.png")             
        io.plot_img(hres_pre_zoom, f"{save_dir}/{save_name}_0hres_unfiltered_zoom.png")   
        io.plot_img(grad_pre_zoom, f"{save_dir}/{save_name}_0grad_unfiltered_zoom.png")  
        np.save(f"{save_dir}/{save_name}_0hres_unfiltered.npy", hres_pre_st) 
        np.save(f"{save_dir}/{save_name}_0grad_unfiltered.npy", grad_pre_st)  
        io.save_cols(f"{save_dir}/{save_name}_0binned_hres_unfiltered.txt", (centers, hbinned_pre))   
        io.save_cols(f"{save_dir}/{save_name}_0binned_grad_unfiltered.txt", (centers, gbinned_pre)) 

    # if not args.is_meanfield:
    #     np.savetxt(f"{save_dir}/{save_name}_z_mass1e14.txt", np.c_[s.vectors["redshift"], s.vectors["masses"]])

elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)
