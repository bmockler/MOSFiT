from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from mosfit.constants import DAY_CGS, FOUR_PI, KM_CGS, M_SUN_CGS, C_CGS
from mosfit.modules.photospheres.photosphere import Photosphere
from scipy.interpolate import interp1d

class tde_photosphere(Photosphere):
    """Photosphere that expands/recedes as a power law of Mdot (or equivalently L (proportional to Mdot) ).
    """

    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value
    RAD_CONST = KM_CGS * DAY_CGS
    TESTING = False
    testnum = 0

    def process(self, **kwargs):
        
        kwargs = self.prepare_input('luminosities', **kwargs)
        self._times = np.array(kwargs['rest_times']) #np.array(kwargs['dense_times']) #kwargs['rest_times']
        self._Mh = kwargs['bhmass']
        self._Mstar = kwargs['starmass']
        self._l = kwargs['lphoto']
        self._Rph_0 = 10.0**(kwargs['Rph0']) # parameter is varied in logspace, kwargs['Rph_0'] = log10(Rph0)
        self._luminosities = np.array(kwargs['luminosities'])
        self._rest_t_explosion = kwargs['resttexplosion']
        #self._beta = kwargs['beta'] # getting beta at this point in process is more complicated than expected bc
        # it can be a beta for a 4/3 - 5/3 combination. Can easily get 'b' -- scaled constant that is linearly related to beta
        # but beta itself is not well defined. -- what does this mean exactly? beta = rt/rp
        Rsolar = c.R_sun.cgs.value
        self._Rstar = kwargs['Rstar'] * Rsolar 

        # Assume solar metallicity for now
        kappa_t = 0.2*(1 + 0.74) # thomson opacity using solar metallicity ( 0.2*(1 + X) = mean Thomson opacity)
        tpeak = kwargs['tpeak'] 

        Ledd = (4 * np.pi * c.G.cgs.value * self._Mh * M_SUN_CGS *
                C_CGS / kappa_t)
        self._beta = 1 # set to this for now, eventually need to somehow get from scaled beta 'b'
        # this should still help with general size of rphotmax
        rt = (self._Mh / self._Mstar)**(1./3.) * self._Rstar # self._Rstar already in cgs units
        rp = rt/self._beta
        
        r_isco = 6 * c.G.cgs.value * self._Mh * M_SUN_CGS / (C_CGS * C_CGS) # Risco in cgs
        rphotmin = r_isco #2*rp #r_isco    
        
        a_p =(c.G.cgs.value * self._Mh * M_SUN_CGS * ((tpeak -
             self._rest_t_explosion) * DAY_CGS / np.pi)**2)**(1. / 3.)


        # semi-major axis of material that accretes at self._times, only calculate for times after first mass accretion
        a_t = (c.G.cgs.value * self._Mh * M_SUN_CGS * ((self._times -
             self._rest_t_explosion) * DAY_CGS / np.pi)**2)**(1. / 3.)
        a_t[self._times < self._rest_t_explosion] = 0.0 
        
        
        rphotmax = rp + 2 * a_t

        # adding rphotmin on to rphot so that there's a soft minimum
        # also creating soft max by doing inverse( 1/rphot + 1/rphotmax)
        # this means the new max is rphotmax/2
        rphot =  self._Rph_0 * a_p * (self._luminosities/ Ledd)**self._l 

        if self.TESTING == True:
            np.savetxt('test_dir/test_photosphere/precut_photosphere/time+rphot'+'{:08d}'.format(self.testnum)+'.txt',
                            (self._times, rphot))

        rphot = (rphot * rphotmax)/(rphot + rphotmax) + rphotmin      

        nan = rphot[np.isnan(rphot)]  
        if len(nan) > 0:
            print (rphot)

        Tphot = (self._luminosities / (rphot**2 * self.STEF_CONST))**0.25

        # ----------------TESTING ----------------
        if self.TESTING == True:
            np.savetxt('test_dir/test_photosphere/end_photosphere/time+Tphot+rphot'+'{:08d}'.format(self.testnum)+'.txt',
                            (self._times, Tphot, rphot, rphotmax)) #, header = 'M_h = '+str(self._Mh)+ '; ilumzero = '+str(ilumzero)) # set time = 0 when explosion goes off
            #np.savetxt('test_dir/test_photosphere/end_photosphere/postilumzerotime+Tphot+rphot'+'{:08d}'.format(self.testnum)+'postilumzero.txt',
            #                (self._times[ilumzero:], Tphot[ilumzero:], rphot[ilumzero:]), header = 'M_h ='+str(self._Mh))
           
            self.testnum += 1
        
        # ----------------------------------------

        return {'radiusphot': rphot, 'temperaturephot': Tphot} 