import matplotlib.pyplot as plt
from pixell import enmap,utils as u,lensing as plensing,bench
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys
import analysis as chutils


nsims = int(sys.argv[1])
comm,rank,my_tasks = mpi.distribute(nsims)
nlen = len(my_tasks)
s = stats.Stats(comm)

# Initialize
analyzer = chutils.Analysis('choices.yaml','flatsky_sim', debug=(rank==0))

for i,task in enumerate(my_tasks):

    # Get lensed sim; (no beam, no noise)
    omap = analyzer.get_stamp(seed=task)

    # Get reconstruction
    recon = analyzer.get_recon(omap,do_real=True,debug = (rank==0 and i==0))

    # Add to stats
    s.add_to_stats('cp1d',recon['fourier']['profile'])
    s.add_to_stats('r1d',recon['real']['profile'])
    s.add_to_stack('stack',recon['real']['map'])
    if rank==0 and (i+1)%10==0: print(f"Rank {rank} done with {i+1} / {nlen}..." )

s.get_stats()
s.get_stacks()


if rank==0:
    rcents = analyzer.rcents
    lcents = analyzer.lcents
    Lmin = analyzer.Lmin
    Lmax = analyzer.Lmax
    p1d = analyzer.template_auto_p1d
    cmean = s.stats['cp1d']['mean']
    cerr = s.stats['cp1d']['errmean']
    ccov = s.stats['cp1d']['covmean']

    rsel = rcents<10*u.arcmin
    rmean = s.stats['r1d']['mean'][rsel]
    rcov = s.stats['r1d']['covmean'][rsel,:][:,rsel]
    rerr = np.sqrt(np.diagonal(rcov))

    
    stack = s.stacks['stack']
    io.plot_img(stack,'stack.png')
    
    sel = np.logical_and(lcents>Lmin,lcents<Lmax)
    signal = p1d[sel]
    cov = ccov[sel,:][:,sel]
    err = cerr[sel]
    
    io.plot_img(stats.cov2corr(ccov),'ccov.png')
    cinv = np.linalg.inv(cov)

    snr1 = np.sqrt(np.dot(np.dot(signal,cinv),signal)  * (1000/nsims))
    snr2 = np.sqrt(np.sum(signal**2./err**2.)  * (1000/nsims))
    snr3 = np.sqrt(np.dot(np.dot(rmean,np.linalg.inv(rcov)),rmean)  * (1000/nsims))

    print(
        f"SNR estimates (scaled to 1000 sims): "
        f"\n  Method 1: Fourier w/ covmat :     {snr1:.2f}"
        f"\n  Method 2: Fourier diagonal:       {snr2:.2f}"
        f"\n  Method 3: Real w/covmat:       {snr3:.2f}"
        f"\n The second is quickest to converge. All three should agree in the limit of infinite sims."
    )
    
    
    pl = io.Plotter(xyscale='linlin',xlabel='L',ylabel='L^2 C_L')
    pl.add(lcents,lcents**2 * p1d,marker='o',label='full template')
    pl.add_err(lcents,lcents**2 * cmean,yerr=cerr * lcents**2.,marker='o')
    pl.add(lcents[sel],lcents[sel]**2 * p1d[sel],marker='d',color='r',label='selected template')
    pl.vline(x=500)
    pl.hline(y=0)
    pl._ax.set_ylim(-3e-5,5.1e-5)
    pl.done('recon_template_cross_p1d.png')

    pl = io.Plotter()
    pl.add_err(rcents[rsel]/u.arcmin,rmean,yerr=rerr,marker='o')
    pl.add(analyzer.thetas/u.arcmin, analyzer.template_kappa_1d)
    pl._ax.set_xlim(0,10)
    pl._ax.set_ylim(-0.01,rmean.max()*1.2)
    pl.hline(y=0)
    pl.done('recon_profile.png')
