import numpy as np
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
        alpha=1,
        beta=3,
        gamma=1,
        overdensity=500,
        *args, **kwargs):

        self._set_shape(mass * c * z * alpha * beta * gamma)
        super().__init__(mass, c, z, *args, **kwargs)

        self.bins = bins
        self.amps = amps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def profile(self, r):
        exp = (self.beta - self.gamma) / self.alpha
        rho = self.delta_c * self.rho_bg / ((r / self.rs) ** self.gamma * (1 + (r / self.rs) ** self.alpha) ** exp)
        for i in range(len(self.bins)-1):
            rho = np.where(np.logical_and(r >= self.bins[i], r <= self.bins[i+1]), rho*self.amps[i], rho)
        return rho