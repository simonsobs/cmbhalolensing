
import camb
from colossus.cosmology import cosmology as colcosmo
from colossus.halo import concentration as conc
from profiley.helpers.lss import power2xi
import pyccl as ccl
from scipy.interpolate import interp1d

import matplotlib.colors as mcolors
import numpy as np
from orphics.io import plot_img, Plotter
from orphics import stats, maps
from pixell import enmap, utils as u
import sys
from sklearn.covariance import LedoitWolf
import os
sys.path.append(os.path.abspath("/home3/nehajo/projects/CMASS-cmbhalolensing/"))
from halo_funcs import TunableNFW as TNFW
sys.path.append(os.path.abspath("/home3/nehajo/scripts/"))
from profiles import errors, chi_square_pte

Lmin=200
Lmax=5000
test=False

pathname=f"/home3/nehajo/projects/CMASS-cmbhalolensing/sims/agora_Lmin200_Lmax5000/agora_cmb_90/agora_cmb_90"
if test: savename = pathname + "_TEST"
else: savename = pathname

def bin(data, modrmap, bin_edges):
    binner = stats.bin2D(modrmap, bin_edges)
    cents, ret = binner.bin(data)
    return ret 

temp = enmap.read_map(f"{pathname}_template.fits")
shape, wcs = temp.shape, temp.wcs
modlmap = enmap.modlmap(shape, wcs)
modrmap = enmap.modrmap(shape, wcs)
kmask = enmap.read_map(f"{pathname}_kmask.fits")
taper = enmap.read_map(f"{pathname}_taper.fits")
mf = enmap.enmap(np.load(f"{pathname}_mf_1rkappa.npy"), wcs)
mf_err = np.loadtxt(f"{pathname}_mf_1rkappa_errs.txt")
cents, Ckt1d = np.loadtxt(f"{pathname}_1binned_tlkappa.txt", unpack=True)
Ckt_errs = np.loadtxt(f"{pathname}_1tlkappa_errs.txt")
_, Clt1d = np.loadtxt(f"{pathname}_1binned_rlkappa.txt", unpack=True)
Clt_errs = np.loadtxt(f"{pathname}_1rlkappa_errs.txt")

ell_bin_edges = np.loadtxt(f"{pathname}_ellbin_edges.txt")
bin_cents = (ell_bin_edges[1:] + ell_bin_edges[:-1]) / 2.0
assert cents.all() == bin_cents.all()

if test:
    #recalc template
    cat = "/data7/nehajo/CMASS/sim_cats/agora_lensed_cmasslike_deccut.npy"
    data = np.load(cat, allow_pickle=True).item()
    ras = data['ra']
    decs = data['dec']
    zs = data['z']
    masses = data['M200c']

    H0 = 67.7 # agora fiducial
    Om = 0.307
    Ob = 0.048
    h = H0/100
    zmean = 0.58
    temp_mass = 2.e13

    ombh2 = Ob*h**2
    omch2 = (Om - Ob)*h**2
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=0.06,  
                    As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
    camb_results = camb.get_results(pars)
    d = camb_results.angular_diameter_distance(zmean)

    cosmo = ccl.Cosmology(Omega_c=Om-Ob, Omega_b=Ob, h=h, A_s=2e-9, n_s=0.96)
    k = np.logspace(-15, 15, 10000)
    mdef = ccl.halos.MassDef(200, 'critical')
    bias = ccl.halos.HaloBiasTinker10(mass_def=mdef)
    rho_m = ccl.background.rho_x(cosmo, 1, 'matter')
    r_xi = np.logspace(-10, 7, 1000)

    
    colcosmo.setCosmology('planck15')

    def get_temp_ellstamp(M, z):
        temp_thetas = np.linspace(0.01, 50, 1000)
        temp_rad = temp_thetas*u.arcmin*d*(1+z)
        Pk = ccl.linear_matter_power(cosmo, k, 1/(1+z))
        c = conc.concentration(temp_mass, '200c', z, model = "diemer15")
        bh = bias(cosmo, temp_mass, 1/(1+z))
        Pgm = bh * Pk
        lnPgm_lnk = interp1d(np.log(k), np.log(Pgm))
        xi = power2xi(lnPgm_lnk, r_xi)
        xi_smooth = interp1d(r_xi, xi)
        nfw = TNFW(temp_mass, c, z, f_2h=xi_smooth, frame="comoving")

        temp_profile = nfw.convergence(temp_rad, 1100).flatten()
        temp_stamp = enmap.enmap(maps.interp(temp_thetas*u.arcmin, temp_profile)(modrmap), wcs)
        temp = temp_stamp*taper
        temp_ellstamp = enmap.fft(temp, normalize="phys")*kmask

        Cltemp = (temp_ellstamp*np.conj(temp_ellstamp)).real

        return temp, Cltemp

    
    print("template:", temp.shape)

    #one temp
    temp_stamp_test, Cltemp_test = get_temp_ellstamp(masses.mean(), zs.mean())
    Cltemp1d_test = bin(Cltemp_test, modlmap, ell_bin_edges)

    # #mean temp
    # for mass, z in zip(masses[sample], zs[sample]):
    #     Cltemp_mean += get_temp_ellstamp(mass, z)
        
    # Cltemp_mean /= len(sample)
    # Cltemp1d_mean = bin(Cltemp_mean, modlmap, ell_bin_edges)

    pl = Plotter(xyscale='linlin', xlabel='$L$', ylabel='$L^2 C_L^{\kappa \kappa}$')
    pl.add(cents, (cents**2)*Cltemp1d_test, color="tab:purple", label="recalc temp x temp")
    # pl.add(cents, (cents**2)*Cltemp1d, color="black", label="temp x temp")
    pl._ax.axhline(y=0, ls="dotted", color="gray")
    pl.done(f"{savename}_template_comparison.png")

temp_ellstamp = enmap.fft(temp, normalize="phys")*kmask
plot_img(enmap.fftshift(temp_ellstamp).real, f"{savename}_template_fft.png")

Cltemp = (temp_ellstamp*np.conj(temp_ellstamp)).real
Cltemp1d = bin(Cltemp, modlmap, ell_bin_edges)


mf_ellstamp = enmap.fft(mf, normalize="phys")*kmask

Cmt_stamp = (mf_ellstamp*np.conj(temp_ellstamp)).real
Cmt1d = bin(Cmt_stamp, modlmap, ell_bin_edges)

Clt1d_mfsub = Clt1d-Cmt1d

# Cmt1d_stacked = bin(mf_ellstamp, modlmap, ell_bin_edges)
np.savetxt(f"{savename}_1binned_rlkappa_mfsub.txt", Clt1d_mfsub) 
cut = np.where(np.logical_and(cents > Lmin, cents < Lmax))[0]

pl = Plotter(xyscale='linlin', xlabel='$L$', ylabel='$L^2 C_L^{\kappa \kappa}$')
pl.add_err(cents[cut], (cents[cut]**2)*Clt1d_mfsub[cut], (cents[cut]**2)*Clt_errs[cut], ls="-", label="cmb recon x template \nmfsub", color="tab:blue")
pl.add_err(cents[cut], (cents[cut]**2)*Ckt1d[cut], (cents[cut]**2)*Ckt_errs[cut], ls="-", color="black", label="true x template")
if test: pl.add(cents[cut], (cents[cut]**2)*Cltemp1d, color="tab:purple", label="temp x temp")
pl._ax.axhline(y=0, ls="dotted", color="gray")
pl.done(f"{savename}_1binned_rlkappa_mfsub.png")

pl = Plotter(xyscale='linlin', xlabel='$L$', ylabel='$L^2 C_L^{\kappa \kappa}$')
pl.add_err(cents[cut], (cents[cut]**2)*Clt1d[cut], (cents[cut]**2)*Clt_errs[cut], ls="-", label="cmb recon x template \nmfsub", color="tab:blue")
pl.add_err(cents[cut], (cents[cut]**2)*Ckt1d[cut], (cents[cut]**2)*Ckt_errs[cut], ls="-", color="black", label="true x template")
pl._ax.axhline(y=0, ls="dotted", color="gray")
pl.add(cents[cut], (cents[cut]**2)*Cmt1d[cut], ls="--", color="gray", label="mf x template")
pl.done(f"{savename}_1binned_rlkappa_mf.png")