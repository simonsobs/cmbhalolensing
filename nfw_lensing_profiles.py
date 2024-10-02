import camb
import halo_funcs as hf
import matplotlib.pyplot as plt
import numpy as np
from orphics import io, lensing
from pixell import utils as u
from profiley.nfw import TNFW
from halo_funcs import TunableNFW as NFW



G = u.G*u.M_sun / (1e6*u.pc)**3  # G in Mpc and Msun units
c_light = u.c/(1e6*u.pc)       # c in Mpc and s units
H0 = 67.5
h=H0/100
z_cmb = np.asarray([1100.])

pars = camb.set_params(H0=H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                       As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
camb_results = camb.get_results(pars)

c = np.asarray([5.])
M = np.asarray([2.e13])
zs = np.asarray([0.6])

comL = camb_results.angular_diameter_distance(zs[0])
Rs = np.geomspace(0.01, 10, 500)
thetas = Rs/comL
print("comL:",comL)

bins = np.geomspace(Rs.min(), Rs.max(), 11)
print(bins)
amps = np.asarray([np.ones(10), np.full(10, 2), np.arange(10)+1., 
                   (np.arange(10)+1)*0.1])

rho_pl = io.Plotter(xyscale='loglog', xlabel='R [Mpc]', ylabel='$\\rho$ (R)')
sigma_pl = io.Plotter(xyscale='loglin', xlabel='R [Mpc]', ylabel='$\\ R x \Delta \Sigma$ [Mpc M$_\odot$ / pc$^2$]')
k_pl = io.Plotter(xyscale='linlin', xlabel='$\\theta$ [arcmin]', ylabel='$\\kappa$ (R)')

for amp in amps:
    print(amp)
    nfw = NFW(M, c, zs, bins, amp)
    rho = nfw.profile(Rs)
    # rho = nfw.tuned_profile(Rs, comL*bin_edges, alphas)
    sigma = nfw.projected_cumulative(Rs) #rho shape: (10082, 200, 1)
    print("sigma-->",sigma.shape)
    dsigma = nfw.projected_excess(Rs)
    print("dsigma-->",dsigma.shape)
    kappa = nfw.convergence(Rs, z_cmb) #rho shape: (500, 200, 1)
    print("kappa-->", kappa.shape)


    rho_pl.add(Rs, rho)
    sigma_pl.add(Rs, Rs*dsigma.flatten()/1.e12) #same units as weak lensing gg
    k_pl.add(Rs, kappa)

rho_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/rho_test.png")
sigma_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/sigma_test.png")
k_pl.done("/home3/nehajo/projects/cmbhalolensing/tests/kappa_test.png")
print("plots done")