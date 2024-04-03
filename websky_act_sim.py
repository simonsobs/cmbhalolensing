import time as t
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
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

my_path = "/home3/eunseong/cmbhalolensing/data"
sim_path = "/data5/sims/websky/data"
act_sim_path = "/data5/sims/websky/dr5_clusters"
output_path = "/home3/eunseong/cmbhalolensing/wsky_outs"
mat_path = "/home1/mathm/repos/cmbhalolensing"

#-------------------------------------------------------------------------------

save_name = "7act_cmb_tsz_ksz_cib_msub_inp"
save_dir = f"{output_path}/{save_name}"
io.mkdir(f"{save_dir}")
print(" ::: saving to ", save_dir)

# options: halo, act_tsz, act_tsz_ksz_cib
# which_cat = "act_tsz"
which_cat = "act_tsz_ksz_cib"

# options: l_only, l_tsz, l_tsz_msub, l_ksz_cib, l_tsz_ksz_cib, l_tsz_ksz_cib_msub
hres_choice = "l_tsz_ksz_cib_msub"
grad_choice = "l_ksz_cib"
print(" ::: hres =", hres_choice, "and grad =", grad_choice)

test = False
mean_field = True
inpainting = True
print(" ::: test:", test, "/ mean_field:", mean_field, "/ inpainting:", inpainting)

freq_sz = 150

# simulation setting -----------------------------------------------------------

xlmin = 200
xlmax = 2000

ylmin = 200
# ylmax = 6000 # high lmax cut
ylmax = 3500  # low lmax cut
ylcut = 2

klmin = 200
# klmax = 5000 # high lmax cut
klmax = 3000  # low lmax cut

print(" ::: hlmax =", ylmax, "and klmax =", klmax)

tap_per = 12.0
pad_per = 3.0

fwhm = 1.4
nlevel = 20

px = 0.5
width = 120./60.
maxr = width * utils.degree / 2.0

#-------------------------------------------------------------------------------

# for binned kappa profile
arcmax = 15.0
arcstep = 1.5
bin_edges = np.arange(0, arcmax, arcstep)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret

def load_beam(freq):
    if freq=="f150": fname = f"{act_sim_path}/beam_gaussian_la145.txt"
    elif freq=="f090": fname = f"{act_sim_path}/beam_gaussian_la093.txt"
    ls,bls = np.loadtxt(fname,usecols=[0,1],unpack=True)
    assert ls[0]==0
    bls = bls / bls[0]
    return maps.interp(ls,bls)

def apply_beam(imap):
    # map2alm of the maps, almxfl(alm, beam_1d) to convolve with beam, alm2map to convert back to map
    alm_lmax = 8192 * 3
    if freq_sz == 150: bfunc = cutils.load_beam("f150")
    elif freq_sz == 90: bfunc = cutils.load_beam("f090")
    imap_alm = curvedsky.map2alm(imap, lmax=alm_lmax)
    beam_convoloved_alm = curvedsky.almxfl(imap_alm, bfunc)
    return curvedsky.alm2map(beam_convoloved_alm, enmap.empty(imap.shape, imap.wcs))


# reading catalogue ------------------------------------------------------------

if mean_field == True:

    # load random catalogue - created by mapcat.py + randcat.py
    if which_cat == "halo":
        cat = f"{my_path}/websky_halo_randoms_actlike.txt"
    elif which_cat == "act_tsz":
        cat = f"{my_path}/websky_ACT_cmb_tsz_randoms.txt"
    elif which_cat == "act_tsz_ksz_cib":
        cat = f"{my_path}/websky_ACT_cmb_tsz_ksz_cib_randoms.txt"
        #cat = f"{my_path}/websky_ACT_cmb_tsz_ksz_cib_randoms_snr5p5.txt"

    ras, decs = np.loadtxt(cat, unpack=True)
    print(" ::: this is a mean-field run")

else:

    if which_cat == "halo":

        cat = "halos.pksc"

        omegab = 0.049
        omegac = 0.261
        omegam = omegab + omegac
        h      = 0.68
        ns     = 0.965
        sigma8 = 0.81

        rho = 2.775e11 * omegam * h**2 # Msun/Mpc^3
        c = 3e5

        H0 = 100*h
        nz = 100000
        z1 = 0.0
        z2 = 6.0
        za = np.linspace(z1,z2,nz)
        dz = za[1]-za[0]

        H      = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)
        dchidz = lambda z: c/H(z)
        chia   = np.cumsum(dchidz(za))*dz
        zofchi = interp1d(chia,za)

        # load the entire halo catalogue
        f = open(f"{sim_path}/halos.pksc")
        N = np.fromfile(f, count=3, dtype=np.int32)[0]
        catalog = np.fromfile(f, count=N*10, dtype=np.float32)
        catalog = np.reshape(catalog, (N,10))

        x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
        R  = catalog[:,6] # Mpc

        # convert to mass, redshift, RA and DEC
        M200m      = 4*np.pi/3.*rho*R**3  # this is M200m (mean density 200 times mean) in Msun
        chi        = np.sqrt(x**2+y**2+z**2)  # Mpc
        theta, phi = hp.vec2ang(np.column_stack((x,y,z))) # in radians

        mass     = M200m/1e14
        zs       = zofchi(chi)
        ras      = np.rad2deg(phi)
        decs     = np.rad2deg(np.pi/2. - theta)

        print(" ::: total number of halos = ", N) # 862923142
        print(" ::: min and max mass: ", mass.min(), mass.max()) # 0.012929761 34.642567
        print(" ::: min and max redshift: ", zs.min(), zs.max())

        print(" ::: selecting the sample that's similar to ACT DR5 sim catalogue")
        temp_cat = f"{act_sim_path}/ACTSim_CMB-T_cmb-tsz-ksz-cib_MFMF_pass2_mass.fits" # 7943
        hdu = fits.open(temp_cat)
        act_mass = hdu[1].data["M200m"] # 1e14 Msun
        num_mbin = 30 # chosen to return a similar mass distribution of the sample

        mbins = np.linspace(act_mass.min(), act_mass.max(), num_mbin)
        mhist, mbin_edges = np.histogram(act_mass, mbins)

        new_mass = []
        new_zs = []
        new_ras = []
        new_decs = []

        for i in range(len(mhist)):
            min_mass = mbin_edges[i]
            max_mass = mbin_edges[i+1]

            ind0 = np.where((mass > min_mass) & (mass < max_mass))[0]
            temp_mass = mass[ind0]
            temp_zs = zs[ind0]
            temp_ras = ras[ind0]
            temp_decs = decs[ind0]

            np.random.seed(100)
            ind1 = np.random.choice(len(temp_mass), size=mhist[i])
            temp_mass = temp_mass[ind1]
            temp_zs = temp_zs[ind1]
            temp_ras = temp_ras[ind1]
            temp_decs = temp_decs[ind1]

            new_mass.append(temp_mass)
            new_zs.append(temp_zs)
            new_ras.append(temp_ras)
            new_decs.append(temp_decs)

        mass = np.concatenate(new_mass, axis=None)
        zs = np.concatenate(new_zs, axis=None)
        ras = np.concatenate(new_ras, axis=None)
        decs = np.concatenate(new_decs, axis=None)

        print(" ::: sample selection is done!")
        print(" ::: min and max mass: ", mass.min(), mass.max())
        print(" ::: min and max redshift: ", zs.min(), zs.max())

    elif which_cat == "act_tsz":

        cat = f"{act_sim_path}/ACTSim_CMB-T_cmb-tsz_MFMF_pass2_mass.fits" # 11902
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        mass = hdu[1].data["M500c"] # 1e14 Msun
        snr = hdu[1].data["SNR"]
        zs = hdu[1].data["redshift"]

    elif which_cat == "act_tsz_ksz_cib":

        cat = f"{act_sim_path}/ACTSim_CMB-T_cmb-tsz-ksz-cib_MFMF_pass2_mass.fits" # 7943
        hdu = fits.open(cat)
        ras = hdu[1].data["RADeg"]
        decs = hdu[1].data["DECDeg"]
        mass = hdu[1].data["M500c"] # 1e14 Msun
        snr = hdu[1].data["SNR"]
        zs = hdu[1].data["redshift"]

        # # SNR cut corresponds to data cosmological sample
        # snr_cut = 5.5 # 2956
        # ras = ras[snr > snr_cut]
        # decs = decs[snr > snr_cut]
        # mass = mass[snr > snr_cut]
        # zs = zs[snr > snr_cut]

print(" ::: name of catalogue = ", cat)
print(" ::: total number of clusters for stacking = ", len(ras))

# reading maps -----------------------------------------------------------------

# read tSZ cluster model image map by Matt Hilton
if which_cat == "act_tsz":
    if freq_sz == 150:
        imap = f"{act_sim_path}/model_MFMF_pass2_cmb-tsz_f150.fits"
    elif freq_sz == 90:
        imap = f"{act_sim_path}/model_MFMF_pass2_cmb-tsz_f090.fits"
elif which_cat == "act_tsz_ksz_cib":
    if freq_sz == 150:
        imap = f"{act_sim_path}/model_MFMF_pass2_cmb-tsz-ksz-cib_f150.fits"
    elif freq_sz == 90:
        imap = f"{act_sim_path}/model_MFMF_pass2_cmb-tsz-ksz-cib_f090.fits"

modelmap = enmap.read_map(imap)
shape, wcs = modelmap.shape, modelmap.wcs # (10320, 43200)
print(" ::: reading tsz cluster model image map at %d GHz:" %freq_sz, imap)
print("modelmap", np.shape(modelmap))


# # ------------------------------------------------------------------------------

# # # test switching lensed cmb map back to the original one

# r_lmap = f"{my_path}/reproj_actfoot_dlensed.fits"
# r_tszmap = f"{my_path}/reproj_actfoot_tsz_8192.fits"
# r_kszmap = f"{my_path}/reproj_actfoot_ksz.fits"
# r_cibmap = f"{my_path}/reproj_actfoot_cib_nu0145.fits"

# b_lmap = f"{my_path}/beam_convolved_actfoot_lmap.fits"
# b_l_tsz_map = f"{my_path}/beam_convolved_actfoot_l_tsz_map.fits"
# b_l_ksz_cib_map = f"{my_path}/beam_convolved_actfoot_l_ksz_cib_map.fits"
# b_l_tsz_ksz_cib_map = f"{my_path}/beam_convolved_actfoot_l_tsz_ksz_cib_map.fits"

# # reading lensed alm map (switching to the map that Mat generated 2024.03.26, this matches the model map shape)
# ifile = f"{mat_path}/dlensed_actfoot.fits"
# lmap = enmap.read_map(ifile)
# print(" ::: reading lensed cmb map:", ifile)
# print("lmap", np.shape(lmap)) # (10320, 43200)

# enmap.write_map(r_lmap, lmap)
# print(" ::: saving reprojected lmap")

# tszmap = enmap.read_map(r_tszmap, delayed=False)
# kszmap = enmap.read_map(r_kszmap, delayed=False)
# cibmap = enmap.read_map(r_cibmap, delayed=False)

# print(" ::: applying the act beam to the maps")
# lmap = apply_beam(lmap)
# l_tsz_map = apply_beam(lmap + tszmap)
# l_ksz_cib_map = apply_beam(lmap + kszmap + cibmap)
# l_tsz_ksz_cib_map = apply_beam(lmap + tszmap + kszmap + cibmap)

# enmap.write_map(b_lmap, lmap)
# enmap.write_map(b_l_tsz_map, l_tsz_map)
# enmap.write_map(b_l_ksz_cib_map, l_ksz_cib_map)
# enmap.write_map(b_l_tsz_ksz_cib_map, l_tsz_ksz_cib_map)
# print(" ::: saving beam convoloved lmap")

# sys.exit()

# ------------------------------------------------------------------------------


# reprojected maps
r_kmap = f"{my_path}/reproj_actfoot_kap_lt4.5.fits"
r_lmap = f"{my_path}/reproj_actfoot_dlensed.fits" # this is from Mat
# r_lmap = f"{my_path}/reproj_actfoot_lensed_alm.fits"
r_tszmap = f"{my_path}/reproj_actfoot_tsz_8192.fits"
r_kszmap = f"{my_path}/reproj_actfoot_ksz.fits"
r_cibmap = f"{my_path}/reproj_actfoot_cib_nu0145.fits"

# reprojected and beam-convolved maps
b_lmap = f"{my_path}/beam_convolved_actfoot_lmap.fits" # this is from Mat
# b_lmap = f"{my_path}/beam_convolved_actfoot_lmap_org.fits"
b_l_tsz_map = f"{my_path}/beam_convolved_actfoot_l_tsz_map.fits"
b_l_ksz_cib_map = f"{my_path}/beam_convolved_actfoot_l_ksz_cib_map.fits"
b_l_tsz_ksz_cib_map = f"{my_path}/beam_convolved_actfoot_l_tsz_ksz_cib_map.fits"


try:
    # it takes too long to reproject the maps and convolve with the beam
    print(" ::: loading the maps that are already saved (reprojected and beam-convolved)")
    kmap = enmap.read_map(r_kmap, delayed=False)
    lmap = enmap.read_map(b_lmap, delayed=False)
    l_tsz_map = enmap.read_map(b_l_tsz_map, delayed=False)
    l_ksz_cib_map = enmap.read_map(b_l_ksz_cib_map, delayed=False)
    l_tsz_ksz_cib_map = enmap.read_map(b_l_tsz_ksz_cib_map, delayed=False)

    print(" ::: reading true kappa map:", r_kmap)
    print(" ::: reading lensed cmb map:", b_lmap)
    print(" ::: reading lensed cmb + tsz map:", b_l_tsz_map)
    print(" ::: reading lensed cmb + ksz + cib map:", b_l_ksz_cib_map)
    print(" ::: reading lensed cmb + tsz+ ksz + cib map:", b_l_tsz_ksz_cib_map)

except:

    try:
        print(" ::: loading the reprojected maps that are already saved")
        kmap = enmap.read_map(r_kmap, delayed=False)
        lmap = enmap.read_map(r_lmap, delayed=False)
        tszmap = enmap.read_map(r_tszmap, delayed=False)
        kszmap = enmap.read_map(r_kszmap, delayed=False)
        cibmap = enmap.read_map(r_cibmap, delayed=False)

    except:
        # reading lensed alm map (switching to the map that Mat generated 2024.03.27, this matches the model map shape)
        ifile = f"{mat_path}/dlensed_actfoot.fits"
        lmap = enmap.read_map(ifile)
        print(" ::: reading lensed cmb map:", ifile)
        print("lmap", np.shape(lmap)) # (10320, 43200)

        # # reading websky lensed alm map
        # ifile = f"{sim_path}/lensed_alm.fits"
        # alm = np.complex128(hp.read_alm(ifile, hdu=(1, 2, 3)))
        # lmap = curvedsky.alm2map(alm[0,:], enmap.empty(shape, wcs, dtype=np.float64))
        # print(" ::: reading lensed alm map and converting to lensed cmb map:", ifile)
        # print("lmap", np.shape(lmap))

        # reading true kappa map
        # ifile = f"{sim_path}/kap.fits" # CMB lensing convergence from 0<z<1100
        ifile = f"{sim_path}/kap_lt4.5.fits" # CMB lensing convergence from z<4.5 from halo+field websky - Mat's map corresonds to this
        imap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        kmap = reproject.healpix2map(imap, shape, wcs)[0,:,:]
        print(" ::: reading true kappa map:", ifile)
        print("kmap", np.shape(kmap))

        # reading high resolution tSZ map
        ifile = f"{sim_path}/tsz_8192.fits"
        imap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        ymap = reproject.healpix2map(imap, shape, wcs)[0,:,:]
        print(" ::: reading high resolution tsz map:", ifile)
        print("ymap", np.shape(ymap))

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
        print(" ::: convering ymap to tsz map at %d GHz" %freq_sz)

        # reading kSZ map
        ifile = f"{sim_path}/ksz.fits"
        imap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
        kszmap = reproject.healpix2map(imap, shape, wcs)[0,:,:]
        print(" ::: reading ksz map:", ifile)
        print("kszmap", np.shape(kszmap))

        # frequency for CIB
        if freq_sz == 150: freq_cib = 145
        elif freq_sz == 90: freq_cib = 93

        # reading CIB map
        if freq_sz == 90:
            # reading CIB 93GHz map
            ifile = f"{sim_path}/cib_nu0093.fits"
            imap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
            cibmap_i = reproject.healpix2map(imap, shape, wcs)[0,:,:]

        elif freq_sz == 150:
            # reading CIB 145GHz map
            ifile = f"{sim_path}/cib_nu0145.fits"
            imap = np.atleast_2d(hp.read_map(ifile, field=tuple(range(0,1)))).astype(np.float64)
            cibmap_i = reproject.healpix2map(imap, shape, wcs)[0,:,:]

        print(" ::: reading cib map at %d GHz:" %freq_cib, ifile)
        print("cibmap", np.shape(cibmap_i))

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

        # enmap.write_map(r_kmap, kmap)
        # enmap.write_map(r_lmap, lmap)
        # enmap.write_map(r_tszmap, tszmap)
        # enmap.write_map(r_kszmap, kszmap)
        # enmap.write_map(r_cibmap, cibmap)

    print(" ::: applying the act beam to the maps")
    lmap = apply_beam(lmap)
    l_tsz_map = apply_beam(lmap + tszmap)
    l_ksz_cib_map = apply_beam(lmap + kszmap + cibmap)
    l_tsz_ksz_cib_map = apply_beam(lmap + tszmap + kszmap + cibmap)

    # enmap.write_map(b_lmap, lmap)
    # enmap.write_map(b_l_tsz_map, l_tsz_map)
    # enmap.write_map(b_l_ksz_cib_map, l_ksz_cib_map)
    # enmap.write_map(b_l_tsz_ksz_cib_map, l_tsz_ksz_cib_map)

print(" ::: model image subtraction")
l_tsz_msub_map = l_tsz_map - modelmap
l_tsz_ksz_cib_msub_map = l_tsz_ksz_cib_map - modelmap

print(" ::: maps are ready!")




#-------------------------------------------------------------------------------

if test == True: nsims = 10 # just for a quick check
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
        kmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # true kappa

    l_only = reproject.thumbnails(
        lmap,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=True # only needed for maps made natively in CAR
    ) # lensed cmb

    l_tsz = reproject.thumbnails(
        l_tsz_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz

    l_ksz_cib = reproject.thumbnails(
        l_ksz_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + ksz + cib

    l_tsz_ksz_cib = reproject.thumbnails(
        l_tsz_ksz_cib_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz + ksz + cib

    l_tsz_msub = reproject.thumbnails(
        l_tsz_msub_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz - tszmodel

    l_tsz_ksz_cib_msub = reproject.thumbnails(
        l_tsz_ksz_cib_msub_map,
        coords,
        r=maxr,
        res=px * utils.arcmin,
        proj="tan",
        oversample=2,
        pixwin=False
    ) # lensed cmb + tsz + ksz + cib - tszmodel

    # initialise calculations based on geometry
    if j == 0:

        # get geometry and Fourier info
        shape = kstamp.shape
        wcs = kstamp.wcs
        modlmap = enmap.modlmap(shape, wcs)
        modrmap = enmap.modrmap(shape, wcs)

        assert wcsutils.equal(kstamp.wcs, l_only.wcs)

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
        beam2d_gauss = maps.gauss_beam(modlmap, fwhm)
        bfunc150 = cutils.load_beam("f150")
        bfunc90 = cutils.load_beam("f090")
        beam2d_150 = bfunc150(modlmap)
        beam2d_90 = bfunc90(modlmap)

        # select the beam
        if which_cat == "halo":
            beam2d = beam2d_gauss
        else: # for the ACT sim
            if freq_sz == 150:
                beam2d = beam2d_150
            elif freq_sz == 90:
                beam2d = beam2d_90

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
    hres = globals()[hres_choice]
    grad = globals()[grad_choice]
    # io.plot_img(grad, f"{save_dir}/{save_name}_0grad_%d.png" %j)

    # # inpainting the gradient leg
    # if inpainting == True:
    #     mask = np.zeros(grad.shape, dtype=bool)
    #     mask[grad.modrmap()<10.0 * utils.arcmin] = True
    #     mask[mask==0] = False
    #     grad = maps.gapfill_edge_conv_flat(grad, mask)
    #     # io.plot_img(grad, f"{save_dir}/{save_name}_0grad_inp_%d.png" %j)

    # taper stamp
    tapered_hres = hres * taper
    tapered_grad = grad * taper

    # inpainting the gradient leg - old inpainting method
    if inpainting == True:
        """
        If inpainting, we
        (1) resample the stamp to 64x64 (2 arcmin pixels)
        (2) Inpaint a hole of radius 4 arcmin
        """
        rmin = 4 * utils.arcmin
        crop_pixels = int(16. / px) # 16 arcminutes wide
        cutout = maps.crop_center(tapered_grad, cropy=crop_pixels, cropx=crop_pixels, sel=False)
        cutout_sel = maps.crop_center(tapered_grad, cropy=crop_pixels, cropx=crop_pixels, sel=True)
        Ndown, Ndown2 = cutout.shape[-2:]
        if Ndown != Ndown2: raise Exception
        hres_fiducial_rms = 20

        if j==0:
            from orphics import pixcov
            pshape = cutout.shape
            pwcs = cutout.wcs
            # beam_fn = lambda x: maps.gauss_beam(fwhm, x)
            ipsizemap = enmap.pixsizemap(pshape, pwcs)
            pivar = maps.ivar(pshape, pwcs, hres_fiducial_rms, ipsizemap=ipsizemap)
            pcov = pixcov.tpcov_from_ivar(Ndown, pivar, theory.lCl, bfunc150)
            geo = pixcov.make_geometry(pshape, pwcs, rmin, n=Ndown, deproject=True, iau=False, res=None, pcov=pcov)

        cutout = pixcov.inpaint_stamp(cutout, geo)
        tapered_grad[cutout_sel] = cutout.copy()

        inp_stamp = tapered_grad / taper
        s.add_to_stack("inp_stamp_st", inp_stamp)


    # get a beam deconvolved Fourier transformed stamp
    k_hres = enmap.fft(tapered_hres, normalize="phys")/beam2d
    k_grad = enmap.fft(tapered_grad, normalize="phys")/beam2d

    k_hres[~np.isfinite(k_hres)] = 0
    k_grad[~np.isfinite(k_grad)] = 0

    # assert np.all(np.isfinite(k_hres))
    # assert np.all(np.isfinite(k_grad))

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
    if mean_field == False:
        s.add_to_stats("redshift", (zs[i],))
        s.add_to_stats("mass", (mass[i],))

    j = j + 1

#-------------------------------------------------------------------------------

s.get_stacks()
s.get_stats()

if rank==0:

    if mean_field == True:
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

    if inpainting == True:
        inp_st = s.stacks["inp_stamp_st"]
        inp_zoom = inp_st[100:140,100:140]
        ibinned = bin(inp_zoom, modrmap, bin_edges)
        io.plot_img(inp_st, f"{save_dir}/{save_name}_0inp.png")
        io.plot_img(inp_zoom, f"{save_dir}/{save_name}_0inp_zoom.png")
        np.save(f"{save_dir}/{save_name}_0inp.npy", inp_st)
        io.save_cols(f"{save_dir}/{save_name}_0binned_inp.txt", (centers, ibinned))

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

    if mean_field == False:
        np.savetxt(f"{save_dir}/{save_name}_z_mass1e14.txt", np.c_[s.vectors["redshift"], s.vectors["mass"]])

elapsed = t.time() - start
print("\r ::: entire run took %.1f seconds" %elapsed)
