from __future__ import print_function
from orphics import maps,io,cosmology,stats
from pixell import enmap
import numpy as np
import os,sys
from enlib import bench
from scipy.special import erfinv,erf

#isave_name = "v01daynight" #args.sys[1] #"mpz_lam20_night_v3"

arcmax = 8.

pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')


for save_name,label in zip([isave_name,f'{isave_name}_curl'],['lensing','curl']):
    no_off = False

    try:
        #cents,profile = np.loadtxt(f'/scratch/r/rbond/msyriac/data/depot/cmbh/postprocess/{save_name}/{save_name}_profile.txt',unpack=True)
        #cov = np.loadtxt(f'/scratch/r/rbond/msyriac/data/depot/cmbh/postprocess/{save_name}/{save_name}_covmat.txt')  
   
        cents = np.loadtxt(f'/global/cscratch1/sd/eunseong/results/v01daynullsz_hilton_beta_plmin_200_plmax_2000_almin_200_almax_6000_klmin_200_klmax_5000_lxcut_2_lycut_2_swidth_128.00_tapper_12.00_padper_3.00_daynight_act_planck_s18/bin_edges.txt',unpack=True)

        opt_p = np.loadtxt(f'/global/cscratch1/sd/eunseong/results/v01daynullsz_hilton_beta_plmin_200_plmax_2000_almin_200_almax_6000_klmin_200_klmax_5000_lxcut_2_lycut_2_swidth_128.00_tapper_12.00_padper_3.00_daynight_act_planck_s18/profile.txt',unpack=True)

        opt_p_mf = np.loadtxt(f'/global/cscratch1/sd/eunseong/results/v01daynullsz_hilton_beta_plmin_200_plmax_2000_almin_200_almax_6000_klmin_200_klmax_5000_lxcut_2_lycut_2_swidth_128.00_tapper_12.00_padper_3.00_daynight_act_planck_s18/profile_mf.txt',unpack=True)

        cov = np.loadtxt(f'/global/cscratch1/sd/eunseong/results/v01daynullsz_hilton_beta_plmin_200_plmax_2000_almin_200_almax_6000_klmin_200_klmax_5000_lxcut_2_lycut_2_swidth_128.00_tapper_12.00_padper_3.00_daynight_act_planck_s18/covm.txt',unpack=True)

    except:
        continue



    profile = opt_p - opt_p_mf


    if no_off: cov = np.diag(np.diagonal(cov))

    scov = cov
    sprofile = profile
    nbins = None
    if arcmax is not None:
        nbins = cents[cents<arcmax].size
        print(nbins)
        sprofile = profile[:nbins]
        scov = cov[:nbins,:nbins]

    cinv = np.linalg.inv(scov)
    diff = sprofile
    chisquare = np.dot(np.dot(diff,cinv),diff)

    print("Chisquare , dof : ", chisquare, diff.size)

    Nsims = 10000000
    samples = np.random.multivariate_normal(diff*0,scov,size=Nsims)
    if label=='curl':
        Nexs = 20
        for i in range(Nexs):
            pl2 = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$')
            pl2.add_err(cents[:nbins], samples[i], yerr=np.sqrt(np.diagonal(scov)))
            pl2.hline(y=0)
            pl2.done(f'{isave_name}_curl_sample_{i}.png')

    chisquares = np.einsum('ik,ik->i', np.einsum('ij,jk->ik',samples,cinv),samples)

    pte = chisquares[chisquares>chisquare].size / Nsims

    ptetext = f"PTE: {pte:.1e}"
    print(ptetext)

    nsigma = np.sqrt(2.) * erfinv(1.-pte)
    snrtext = f"SNR: {nsigma:.2f}"
    print(snrtext)

    names = {'lensing':'Lensing','curl':'Curl'}
    flabel = f'{names[label]}, {ptetext}, {snrtext}'

    pl.add_err(cents, profile, yerr=np.sqrt(np.diagonal(cov)),
               color='k' if label!='curl' else None,
               alpha=0.8 if label=='curl' else 1,
               marker='o',
               label=flabel,
               addx=0.2 if label=='curl' else 0.)


pl.hline(y=0)
pl._ax.set_ylim(-0.15,0.15)
pl.done(f'{isave_name}_curlcomp.png')
