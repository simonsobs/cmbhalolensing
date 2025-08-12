from orphics import io, lensing as olensing, maps, cosmology
import numpy as np
from pixell import enmap,utils as u,lensing as plensing,bench

m500c = 2e14
z=1.0
theta_max = 3.*120. * u.arcmin

kapfile = 'kappa_profile.txt'
thetas,kappa_1h,kappa_2h,_,_,_,_,_ = olensing.kappa_nfw_profiley(mass=m500c,conc=None,
                                                                 z=z,z_s=1100.,background='critical',delta=500, R_off_Mpc = None,
                                                                 apply_filter=False)

io.save_cols(kapfile,(thetas,kappa_1h,kappa_2h))

tot_kappa = kappa_1h + kappa_2h    
pl = io.Plotter(xyscale='loglog',xlabel='$\\theta$ (arcmin)',ylabel='$\\kappa(\\theta)$')
pl.add(thetas/u.arcmin,tot_kappa)
pl.add(thetas/u.arcmin,kappa_2h,ls='--')
pl.add(thetas/u.arcmin,kappa_1h,ls=':')
pl.done('profile.png')
