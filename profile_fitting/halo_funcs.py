import matplotlib.pyplot as plt
import numpy as np
from orphics import maps, stats
from pixell import enmap, utils as u
from profiley.nfw import BaseNFW

class TunableNFW(BaseNFW):

    """
    NFW profile based on profiley that allows rho to be scaled by R bins

    """
    def __init__(self, 
        mass,
        c,
        z,
        bins, 
        amps, 
        two_halo=0,
        alpha=1,
        beta=3,
        gamma=1,
        overdensity=500,
        *args, **kwargs):

        self._set_shape(mass * c * z * alpha * beta * gamma)
        super().__init__(mass, c, z, *args, **kwargs)

        self.bins = bins
        self.amps = amps
        self.two_halo = two_halo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def profile(self, r):
        exp = (self.beta - self.gamma) / self.alpha
        rho = self.delta_c * self.rho_bg / ((r / self.rs) ** self.gamma * (1 + (r / self.rs) ** self.alpha) ** exp) + self.two_halo
        for i in range(len(self.bins)-1):
            rho = np.where(np.logical_and(r >= self.bins[i], r <= self.bins[i+1]), rho*self.amps[i], rho)
        return rho
    
def filter_profile(kappa, theta, data_bins, Lmin, Lmax, shape, wcs, plot=False, savedir=None):
    r = theta*u.arcmin
    bins = data_bins*u.arcmin

    modrmap = enmap.modrmap(shape,wcs)

    kappa2d = maps.interp(r, kappa)(modrmap)
    kappa2d = enmap.enmap(kappa2d)
    
    if plot and (savedir is not None):
        fig, ax = plt.subplots()
        im = ax.imshow(kappa2d)
        fig.colorbar(im)
        fig.savefig(savedir+"test_kappa2d.png")
        plt.close(fig)

    fk = enmap.fft(kappa2d, normalize='phys')
    kfilter = fk*0 + 1
    modlmap = enmap.modlmap(shape,wcs)
    kfilter[modlmap<Lmin]=0
    kfilter[modlmap>Lmax]=0
    filtered_k = enmap.ifft(fk * kfilter, normalize='phys').real

    if plot and (savedir is not None):
        fig, ax = plt.subplots()
        im = ax.imshow(filtered_k)
        fig.colorbar(im)
        fig.savefig(savedir+"test_kappa2d_filtered.png")
        plt.close(fig)

    binner = stats.bin2D(modrmap, bins)
    cents, kf1d = binner.bin(filtered_k)

    return cents/u.arcmin, kf1d