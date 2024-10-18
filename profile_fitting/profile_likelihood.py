import camb
from cobaya.likelihood import Likelihood
from halo_funcs import TunableNFW as NFW
import matplotlib.pyplot as plt
import numpy as np
from pixell import utils as u

class MyLikelihood(Likelihood):
    data_path = ""
    covmat_path = ""
    H0 = 67.5
    h=H0/100
    z_cmb = np.asarray([1100.])

    c = 5
    logM = 13
    zmean = 0

    min_R = 0.01
    max_R = 15
    num_R = 1000

    def initialize(self):
        pars = camb.set_params(H0=self.H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                        As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
        self.camb_results = camb.get_results(pars)
        self.comL = self.camb_results.angular_diameter_distance(self.zmean)
        self.Rs = np.geomspace(self.min_R, self.max_R, self.num_R)
        self.thetas = self.Rs/self.comL/u.arcmin

        self.theta_x, self.data = np.loadtxt(self.data_path, unpack=True)
        self.Rs_x = self.theta_x * u.arcmin * self.comL

        self.covmat = np.loadtxt(self.covmat_path)
        
        data_bins = (self.theta_x[1:] + self.theta_x[:-1])/2 # bins in Mpc
        bins = np.zeros(len(data_bins)+2)
        bins[0] = self.min_R
        bins[-1] = self.max_R
        bins[1:-1] = data_bins
        # self.amp_bins = np.asarray([self.min_R, 5., self.max_R])
        self.bins = bins

        # TESTING
        self.data = self.data[1:]
        self.bins = self.bins[1:]
        self.covmat = self.covmat[1:, 1:]

        self.amp_bins = bins
        self.C_inv = np.linalg.inv(self.covmat)

    def get_requirements(self):
        return {"amp1": None,
                "amp2": None,
                "amp3": None,
                "amp4": None,
                "amp5": None,
                "amp6": None,
                "amp7": None,
                "amp8": None,
                "amp9": None,}
    
    def binner(self, profile):
        binned = [np.mean(profile[np.where(np.logical_and(self.Rs > self.bins[i], self.Rs < self.bins[i+1]))]) for i in range(len(self.bins)-1)]
        return np.asarray(binned)
    
    def logp(self, **params_values):
        amps = np.asarray(list(params_values.values())[1:])
        # print("amps", amps)
        M = 10**self.logM
        nfw = NFW(M, self.c, self.zmean, self.amp_bins, amps)
        kappa = nfw.convergence(self.Rs, self.z_cmb)
        model = self.binner(kappa.flatten())

        # print("model", model)
        # print("data", self.data)
        # print("-->", model.shape, self.data.shape)
        # print("-->", self.Rs_x.shape, self.theta_x.shape)
        # errs = np.sqrt(np.diag(self.covmat))
        # print("-->", errs.shape)

        # plt.plot(self.theta_x, model)
        # plt.errorbar(self.theta_x, self.data, yerr=np.sqrt(np.diag(self.covmat)))
        # plt.savefig("/home3/nehajo/projects/cmbhalolensing/profile_fitting/test_plot.png")
        # plt.close()

        chi2 = np.dot(self.data-model, np.dot(self.C_inv, self.data-model))
        # print("CHI2", chi2)
        loglk = -1/2 * chi2
        return loglk