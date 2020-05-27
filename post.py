import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import utils as cutils
from pixell import enmap, reproject, enplot, utils, wcsutils
from orphics import maps, mpi, io, stats,cosmology,lensing
from scipy.optimize import curve_fit
from numpy import save
import symlens
import healpy as hp
import os, sys
import time as t
from enlib import bench
import warnings
import re
from szar import counts

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Stacked CMB lensing.')
parser.add_argument("stack_path", type=str,help='Stack relative path.')
parser.add_argument("mf_path", type=str,help='Meanfield relative path.')
parser.add_argument("--theory",     type=str,  default=None,help="Lensed theory location for comparison of sim.")
parser.add_argument("--stamp-width-arcmin",     type=float,  default=128.0,help="Stamp width arcmin.")
parser.add_argument("--pix-width-arcmin",     type=float,  default=0.5,help="Stamp width arcmin.")
parser.add_argument("--z",     type=float,  default=0.55,help="Redshift for profile fit.")
parser.add_argument("--conc",     type=float,  default=3.0,help="Concentration for profile fit.")
parser.add_argument("--sigma_mis",     type=float,  default=None,help="Miscentering Rayleigh width in arcmin.")
parser.add_argument("--mass-guess",     type=float,  default=2e14,help="Mass guess in solar masses.")
parser.add_argument("--snr-guess",     type=float,  default=10,help="SNR guess.")
parser.add_argument("--nsigma",     type=float,  default=10,help="Number of sigma away from mass-guess to evaulate likelihood at.")
parser.add_argument("--num-ms",     type=int,  default=20,help="Number of mass points to evaluate likelihood at.")
parser.add_argument("--arcmax",     type=float,  default=10.,help="Maximum arcminute radius distance for fit.")
parser.add_argument("--overdensity",     type=float,  default=200.,help="NFW mass definition overdensity.")
parser.add_argument("--critical", action='store_true',help='Whether NFW mass definition is wrt critical density (default: mean matter density).')
parser.add_argument("--at-z0", action='store_true',help='Whether NFW mass definition is at z=0 (default: at cluster redshift).')


args = parser.parse_args()

mf_path = args.mf_path

if mf_path is not "":
    mf_paramstr = re.search(rf'plmin_(.*?)_meanfield', args.mf_path).group(1)
st_paramstr = re.search(rf'plmin_(.*)', args.stack_path).group(1)
if mf_path is not "":
    assert mf_paramstr==st_paramstr, "ERROR: The parameters for the stack and mean-field do not match."

tap_per = float(re.search(rf'tapper_(.*?)_padper', args.stack_path).group(1))
pad_per = float(re.search(rf'padper_(.*?)_', args.stack_path).group(1))
stamp_width_arcmin = float(re.search(rf'swidth_(.*?)_tapper', args.stack_path).group(1))
klmin = int(re.search(rf'klmin_(.*?)_klmax', args.stack_path).group(1))
klmax = int(re.search(rf'klmax_(.*?)_lxcut', args.stack_path).group(1))


s_stack, shape_stack, wcs_stack, kmask, modrmap, bin_edges = cutils.load_dumped_stats(args.stack_path,get_extra=True)
if mf_path is not "":
    s_mf, shape_mf, wcs_mf = cutils.load_dumped_stats(args.mf_path)

if mf_path is not "":
    assert np.all(shape_stack==shape_mf)
    assert wcsutils.equal(wcs_stack,wcs_mf)
assert np.all(shape_stack==kmask.shape)

shape = shape_stack
wcs = wcs_stack
cents = (bin_edges[:-1]+bin_edges[1:])/2.
crop = 40

unweighted_stack,nmean_weighted_kappa_stack,opt_weighted_kappa_stack,opt_binned,opt_covm,opt_corr,opt_errs,binned,covm,corr,errs = cutils.analyze(s_stack,wcs)
if mf_path is not "":
    mf_unweighted_stack,mf_nmean_weighted_kappa_stack,mf_opt_weighted_kappa_stack,mf_opt_binned,mf_opt_covm,mf_opt_corr,mf_opt_errs,mf_binned,mf_covm,mf_corr,mf_errs = cutils.analyze(s_mf,wcs)

cutils.plot("unweighted_nomfsub.png",unweighted_stack,stamp_width_arcmin,tap_per,pad_per,crop=None)
cutils.plot("unweighted_nomfsub_zoom.png",unweighted_stack,stamp_width_arcmin,tap_per,pad_per,crop=crop)

if mf_path is not "":
    stamp = opt_weighted_kappa_stack - mf_opt_weighted_kappa_stack
    cutils.plot("opt_weighted_mfsub.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=None)
    cutils.plot("opt_weighted_mfsub_zoom.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=crop)

    fwhm = 3.0
    modlmap = opt_weighted_kappa_stack.modlmap()
    stamp = maps.filter_map(opt_weighted_kappa_stack - mf_opt_weighted_kappa_stack,maps.gauss_beam(modlmap,fwhm))
    cutils.plot("sm_opt_weighted_mfsub.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=None)
    cutils.plot("sm_opt_weighted_mfsub_zoom.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=crop)
else:
    stamp = opt_weighted_kappa_stack 
    cutils.plot("opt_weighted_nomfsub.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=None)
    cutils.plot("opt_weighted_nomfsub_zoom.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=crop)

    fwhm = 3.0
    modlmap = opt_weighted_kappa_stack.modlmap()
    stamp = maps.filter_map(opt_weighted_kappa_stack ,maps.gauss_beam(modlmap,fwhm))
    cutils.plot("sm_opt_weighted_nomfsub.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=None)
    cutils.plot("sm_opt_weighted_nomfsub_zoom.png",stamp,stamp_width_arcmin,tap_per,pad_per,crop=crop)


io.plot_img(corr,'corr.png')

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
if mf_path is not "":
    pl.add_err(cents, opt_binned - mf_opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa, mean-field subtracted (optimal)")
    pl.add_err(cents+0.2, binned - mf_binned, yerr=errs,ls="-",label="Filtered kappa, mean-field subtracted")
    pl.add_err(cents, mf_opt_binned, yerr=mf_opt_errs,label="Mean-field (optimal)",ls="-")
    pl.add_err(cents+0.2, mf_binned, yerr=mf_opt_errs,label="Mean-field",ls="-")
else:
    pl.add_err(cents, opt_binned, yerr=opt_errs,ls="-",label="Filtered kappa (optimal)")
    pl.add_err(cents+0.2, binned, yerr=errs,ls="-",label="Filtered kappa")

pl.hline(y=0)
pl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
pl._ax.set_ylim(-0.02,0.15)
pl.done(f'profile.png')

arcmax = 5.
nbins = bin_edges[bin_edges<arcmax].size - 1
if mf_path is "":
    mf_opt_binned = opt_binned*0
    mf_binned = opt_binned*0
diff = (opt_binned - mf_opt_binned)[:nbins]
cinv = np.linalg.inv(opt_covm[:nbins,:nbins])
chisquare = np.dot(np.dot(diff,cinv),diff)
snr = np.sqrt(chisquare)
print("Naive SNR wrt null (optimal) : ", snr)

diff = (binned - mf_binned)[:nbins]
cinv = np.linalg.inv(covm[:nbins,:nbins])
chisquare = np.dot(np.dot(diff,cinv),diff)
snr = np.sqrt(chisquare)
print("Naive SNR wrt null : ", snr)

z = args.z

print("Mean redshift : ",z)

conc = args.conc
cc = None
sigma_mis = args.sigma_mis
mguess = args.mass_guess
merr_guess = (1/args.snr_guess) * mguess
masses = np.linspace(mguess-args.nsigma*merr_guess,mguess+args.nsigma*merr_guess,args.num_ms)
masses = masses[masses>0]
arcmax = args.arcmax
nbins = bin_edges[bin_edges<arcmax].size - 1
profile = (opt_binned - mf_opt_binned)[:nbins]
cov = opt_covm[:nbins,:nbins]
fbin_edges = bin_edges[:nbins+1]
fcents = cents[:nbins]
lnlikes,like_fit,fit_mass,mass_err,fprofiles,fit_profile = lensing.fit_nfw_profile(profile,cov,masses,z,conc,cc,shape,wcs,fbin_edges,lmax=0,lmin=0,
                                                                                   overdensity=args.overdensity,
                                                                                   critical=args.critical,at_cluster_z=args.at_z0,
                                                                                   mass_guess=mguess,sigma_guess=merr_guess,kmask=kmask,sigma_mis=sigma_mis)
    
print("Fit mass : " , fit_mass/1e14,mass_err/1e14)
snr  = fit_mass / mass_err
print("Fit mass SNR : ", snr)

pl = io.Plotter(xlabel='$M$',ylabel='$L$')
likes = np.exp(lnlikes)
pl.add(masses,likes/likes.max())
pl.add(masses,like_fit/like_fit.max())
pl.done('likes.png')

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
pl.add_err(fcents, profile, yerr=np.sqrt(np.diagonal(cov)),ls="-",color='k')
for fp in fprofiles:
    pl.add(fcents, fp,alpha=0.2)
pl.add(fcents, fit_profile,color='k',ls='--')
pl.done('fprofiles.png')


if args.theory is not None:
    savedir = cutils.p['scratch'] + f"/{args.theory}/"
    lensed_version = args.theory
    bfact = float(re.search(rf'bfact_(.*?)_pfact', args.theory).group(1))
    stamp_width_arcmin = float(re.search(rf'swidth_(.*?)_pwidth', lensed_version).group(1))
    pix_width_arcmin = float(re.search(rf'pwidth_(.*?)_bfact', lensed_version).group(1))
    dNpix = int(stamp_width_arcmin * bfact / (pix_width_arcmin))
    ddNpix = int(stamp_width_arcmin / (pix_width_arcmin))
    kappa = enmap.read_map(f'{savedir}kappa.fits')
    dkappa = kappa.resample((dNpix,dNpix))
    tkappa = maps.crop_center(dkappa,ddNpix)
    fkappa = maps.filter_map(tkappa,kmask)
    binner = stats.bin2D(modrmap*180*60/np.pi, bin_edges)
    tcents,t1d = binner.bin(fkappa)
    assert np.all(np.isclose(cents,tcents))
    
    diff = opt_binned - mf_opt_binned

    pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    pl.add_err(cents,diff,yerr=opt_errs,ls='-')
    pl.add(tcents,t1d,ls='--')
    pl.done('theory_comp.png')

    pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
    pl.add_err(cents,diff/t1d,yerr=opt_errs/t1d,ls='-')
    pl._ax.set_ylim(0.6,1.4)
    pl.hline(y=1)
    pl.done('theory_comp_ratio.png')
