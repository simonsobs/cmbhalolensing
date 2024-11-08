import camb
from cobaya.likelihood import Likelihood
from halo_funcs import TunableNFW as NFW, filter_profile
import matplotlib.pyplot as plt
import numpy as np
from orphics import maps, stats
from pixell import utils as u, enmap

class MyLikelihood(Likelihood):
    data_path = ""
    covmat_path = ""
    map_geom = ""
    bin_edges = ""

    H0 = 67.5
    h=H0/100
    z_cmb = np.asarray([1100.])

    c = 5
    logM = 13
    zmean = 0
    log2h = 14

    min_R = 0.01
    max_R = 15
    num_R = 1000
    inner_edges = []

    Lmin = 200
    Lmax = 5000

    def initialize(self):
        pars = camb.set_params(H0=self.H0, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,  
                        As=2e-9, ns=0.965, halofit_version='mead', lmax=3000)
        self.camb_results = camb.get_results(pars)
        self.comL = self.camb_results.angular_diameter_distance(self.zmean)

        self.theta_x, self.data = np.loadtxt(self.data_path, unpack=True)

        self.covmat = np.loadtxt(self.covmat_path)
        self.C_inv = np.linalg.inv(self.covmat)

        self.shape,self.wcs = enmap.read_map_geometry(self.map_geom)
        self.modrmap = enmap.modrmap(self.shape,self.wcs)
        R_edge = np.max(self.modrmap)*self.comL
        if R_edge > self.max_R:
            self.max_R = R_edge
        self.Rs = np.geomspace(self.min_R, self.max_R, self.num_R)
        self.thetas = self.Rs/self.comL/u.arcmin
        
        self.model_bins = np.loadtxt(self.bin_edges)

        self.Rs_bins = np.zeros(len(self.inner_edges) + 2)
        self.Rs_bins[1:-1] = np.asarray(self.inner_edges)
        self.Rs_bins[0] = self.min_R
        self.Rs_bins[-1] = self.max_R

        self.theta_bins = self.Rs_bins*self.comL*u.arcmin

        

    def get_requirements(self):
        return {"amp1": None,
                "amp2": None,
                "amp3": None}
    
    def filter_profile(self, kappa, theta, bin_edges):
        r = theta*u.arcmin
        bins = bin_edges*u.arcmin

        kappa2d = maps.interp(r, kappa)(self.modrmap)
        kappa2d = enmap.enmap(kappa2d)

        fk = enmap.fft(kappa2d, normalize='phys')
        kfilter = fk*0 + 1
        modlmap = enmap.modlmap(self.shape,self.wcs)
        kfilter[modlmap<self.Lmin]=0
        kfilter[modlmap>self.Lmax]=0
        filtered_k = enmap.ifft(fk * kfilter, normalize='phys').real

        binner = stats.bin2D(self.modrmap, bins)
        cents, kf1d = binner.bin(filtered_k)

        return cents/u.arcmin, kf1d
    
    def logp(self, **params_values):
        amps = np.asarray(list(params_values.values())[1:])
        # print("amps", amps)
        M = 10**self.logM
        two_halo = 10**self.log2h
        nfw = NFW(M, self.c, self.zmean, self.Rs_bins, amps, two_halo)
        kappa1d = nfw.convergence(self.Rs, self.z_cmb).flatten()
        fthetas, fkappa = filter_profile(kappa1d, self.thetas, self.model_bins, self.Lmin, self.Lmax, self.shape, self.wcs)
        assert fthetas.all() == self.theta_x.all()
        model = fkappa
        
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