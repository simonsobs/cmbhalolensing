import matplotlib.pyplot as plt
from pixell import enmap,utils as u,lensing as plensing,bench
from orphics import io, lensing as olensing, maps, cosmology,stats, mpi
import numpy as np
import symlens
import sys


"""


"""

periodic = False
nsims = int(sys.argv[1])
Lmin = 500 # below this the effects of windowing become substantial
Lmax = 5000 # above 6000, points become correlated by beam
theta_max = 20 # arcmin






#kapfile = 'kappa_profile.txt'

analyzer = Analysis('choices.yaml','flatsky_sim')


comm,rank,my_tasks = mpi.distribute(nsims)



s = stats.Stats(comm)

nlen = len(my_tasks)
for i,task in enumerate(my_tasks):

    omap = analyzer.get_stamp(seed=task)
    if i==0:
        r = Recon(omap.shape,omap.wcs,apodize=True if not(periodic) else False,
                  klmax=Lmax)
        dmodrmap = omap.modrmap()

        template = maps.interp(csim.thetas,csim.kappa_1d)(dmodrmap)

        ctaper = maps.radial_mask(omap.shape,omap.wcs,
                                 roll_start=theta_max*u.arcmin,
                                 roll_width=20*u.arcmin,
                                 window="cosine")
        if rank==0: io.plot_img(ctaper,'ctaper.png')
        w2 = np.mean(ctaper**2.)
        tkmap = enmap.fft(enmap.enmap(template,omap.wcs) * ctaper,normalize='phys')
        
        cents,p1d,p2d = power(tkmap,tkmap,r.minell,Lmax)

        Kmask = maps.mask_kspace(omap.shape, omap.wcs, lmin=Lmin, lmax=Lmax)

        rbin_edges = np.arange(0.,theta_max*u.arcmin,2.0*u.arcmin)
        rbinner = stats.bin2D(omap.modrmap(),rbin_edges)
        
        if rank==0:
            io.plot_img((np.fft.fftshift(omap.modlmap()**2 * p2d)),'p2d.png')
            
            pl = io.Plotter(xyscale='linlin',xlabel='L',ylabel='L^2 C_L')
            pl.add(cents,cents**2 * p1d,marker='o')
        


        
    krecon = r.recon(omap, p2d_plot = 'data_p2d.png' if (rank==0 and i==0) else None)
    okmap = krecon.copy()
    okmap[~Kmask] = 0
    okreal = enmap.ifft(okmap,normalize='phys').real
    
    kreal = enmap.ifft(krecon,normalize='phys').real
    kmap = enmap.fft(kreal * ctaper,normalize='phys')
    
    

    cents,cp1d,cp2d = power(tkmap,kmap,r.minell,Lmax)
    if not(np.all(np.isfinite(cp1d))): raise ValueError
    if i==0 and rank==0:
        io.plot_img((np.fft.fftshift(omap.modlmap()**2 * cp2d)),'cp2d.png')

    s.add_to_stats('cp1d',cp1d.copy())
    rcents,r1d = rbinner.bin(okreal)
    s.add_to_stats('r1d',r1d.copy())
    s.add_to_stack('stack',okreal.copy())
    # s.add_to_stack('kstack_i',kmap.imag.copy())
    if rank==0 and (i+1)%10==0: print(f"Rank {rank} done with {i+1} / {nlen}..." )

s.get_stats()
s.get_stacks()


if rank==0:
    cmean = s.stats['cp1d']['mean']
    cerr = s.stats['cp1d']['errmean']
    ccov = s.stats['cp1d']['covmean']

    rsel = rcents<10*u.arcmin
    rmean = s.stats['r1d']['mean'][rsel]
    rcov = s.stats['r1d']['covmean'][rsel,:][:,rsel]
    rerr = np.sqrt(np.diagonal(rcov))

    
    stack = s.stacks['stack']
    io.plot_img(stack,'stack.png')
    
    sel = np.logical_and(cents>Lmin,cents<Lmax)
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
    

    pl.add_err(cents,cents**2 * cmean,yerr=cerr * cents**2.,marker='o')
    pl.add(cents[sel],cents[sel]**2 * p1d[sel],marker='d',color='r')
    pl.vline(x=500)
    pl.hline(y=0)
    pl._ax.set_ylim(-3e-5,5.1e-5)
    pl.done('reconprofile.png')

    pl = io.Plotter()
    pl.add_err(rcents[rsel]/u.arcmin,rmean,yerr=rerr,marker='o')
    pl.add(thetas/u.arcmin, kappa)
    pl._ax.set_xlim(0,10)
    pl._ax.set_ylim(-0.01,rmean.max()*1.2)
    pl.hline(y=0)
    pl.done('realprofile.png')
