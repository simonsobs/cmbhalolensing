import camb
from cobaya.likelihood import Likelihood
from halo_funcs import TunableNFW as NFW
import numpy as np
from pixell import utils as u

class MyLikelihood(Likelihood):
    H0 = 67.5
    h=H0/100
    z_cmb = np.asarray([1100.])

    c = np.asarray([5.])
    M = np.asarray([2.e13])
    zs = np.asarray([0.55])

    def initialize(self):
        pars = camb.set_params(H0=self.H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                        As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
        self.camb_results = camb.get_results(pars)
        self.comL = self.camb_results.angular_diameter_distance(zs[0])
        self.Rs = np.geomspace(0.01, 10, 1000)
        self.thetas = self.Rs/self.comL/u.arcmin

        ###TODO: read in data and covmat
                # calculate C_inv
                # define self.bins

    def get_requirements(self):
        pass
        ###TODO: alphas as params (fixed number? depend on binning?)

    def logp(self, **params_values):
        ###TODO: read in alphas from params_values
        nfw = NFW(self.M, self.c, self.zs, self.bins, amps)
        model = nfw.convergence(self.Rs, self.z_cmb)

        chi2 = np.dot(self.data-model, np.dot(C_inv, self.data-model))