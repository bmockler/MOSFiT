"""Definitions for the `TdePhotosphere` class."""
from math import pi

# import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import C_CGS, DAY_CGS, KM_CGS, M_SUN_CGS  # FOUR_PI
from mosfit.modules.photospheres.photosphere import Photosphere


# from scipy.interpolate import interp1d


class TdePhotosphere(Photosphere):
    """Photosphere for a tidal disruption event.

    Photosphere that expands/recedes as a power law of Mdot
    (or equivalently L (proportional to Mdot) ).
    """

    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS

    def process(self, **kwargs):
        """Process module."""
        kwargs = self.prepare_input('luminosities', **kwargs)
        self._times = np.array(kwargs['rest_times'])
        self._Mh = kwargs['bhmass']
        self._Mstar = kwargs['starmass']
        self._l = kwargs['lphoto']
        self._Rph_0 = kwargs['Rph0']
        self._luminosities = np.array(kwargs['luminosities'])
        self._rest_t_explosion = kwargs['resttexplosion']
        self._beta = kwargs['beta']  # for now linearly interp between
        # beta43 and beta53 for a given 'b' if Mstar is in transition region
        self._rphotmaxwind = kwargs['rphotmaxwind']
        self._vphotmaxwind = kwargs['vphotmaxwind']

        Rsolar = c.R_sun.cgs.value
        self._Rstar = kwargs['Rstar'] * Rsolar

        tpeak = kwargs['tpeak']

        Ledd = kwargs['Ledd']
        self._Leddlim = kwargs['Leddlim']
        Llim = self._Leddlim*Ledd  # user defined multiple of Ledd

        rt = (self._Mh / self._Mstar)**(1. / 3.) * self._Rstar
        self._rp = rt / self._beta

        r_isco = 6 * c.G.cgs.value * self._Mh * M_SUN_CGS / (C_CGS * C_CGS)
        rphotmin = r_isco

        a_p = (c.G.cgs.value * self._Mh * M_SUN_CGS * ((
            tpeak - self._rest_t_explosion) * DAY_CGS / np.pi)**2)**(1. / 3.)


        if False: # just for saving for testing
            # semi-major axis of material that accretes at self._times,
            # only calculate for times after first mass accretion
            a_t = (c.G.cgs.value * self._Mh * M_SUN_CGS * ((
                self._times - self._rest_t_explosion) * DAY_CGS / np.pi)**2)**(
                    1. / 3.)
            a_t[self._times < self._rest_t_explosion] = 0.0
            
            if False:
                rphotmax = self._rp + 2 * a_t  # rphotmax set to apocenter of debris stream

            if False: # just for saving for testing
                rphotmax = 2*self._rp + self._vphotmaxwind * C_CGS * (
                    self._times - self._rest_t_explosion) * DAY_CGS + self._rp + 2 * a_t 
                rphotmax[self._times < self._rest_t_explosion] = 0.0
        
        
        rphot = self._Rph_0 * a_p * (self._luminosities / Llim)**self._l

        # soft max -- inverse( 1/rphot + 1/rphotmax) -- implementing max photosphere in constraints instead
        #rphot = (rphot * rphotmax) / (rphot + rphotmax) + rphotmin
        
        # adding rphotmin on to rphot for soft min
        rphot = rphot + rphotmin


        Tphot = (self._luminosities / (rphot**2 * self.STEF_CONST))**0.25

        return {self.key('radiusphot'): rphot, self.key('temperaturephot'): Tphot,
                'rp': self._rp, 'rphotmin': rphotmin}
