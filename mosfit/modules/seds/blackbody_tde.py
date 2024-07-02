"""Definitions for the `Blackbody` class."""
from math import pi

import numexpr as ne
import numpy as np
from astropy import constants as c
from astropy import units as u
from mosfit.constants import FOUR_PI
from mosfit.modules.seds.sed import SED
from time import perf_counter


# Important: Only define one ``Module`` class per file.


class Blackbody(SED):
    """Blackbody spectral energy dist. for given temperature and radius."""

    C_CONST = c.c.cgs.value
    FLUX_CONST = FOUR_PI * (
        2.0 * c.h * c.c ** 2 * pi).cgs.value * u.Angstrom.cgs.scale
    X_CONST = (c.h * c.c / c.k_B).cgs.value
    STEF_CONST = (4.0 * pi * c.sigma_sb).cgs.value

    def process(self, **kwargs):
        """Process module."""
        lum_key = self.key('luminosities')
        kwargs = self.prepare_input(lum_key, **kwargs)
        self._luminosities = kwargs[lum_key]
        self._bands = kwargs['all_bands']
        self._band_indices = kwargs['all_band_indices']
        self._frequencies = kwargs['all_frequencies']
        self._radius_phot = kwargs[self.key('radiusphot')]
        self._temperature_phot = kwargs[self.key('temperaturephot')]
        self._include_latetime_luminosity = kwargs[self.key('latetimlum')]
        if self._include_latetime_luminosity:
            self._latetime_luminosity_fraction = kwargs[self.key('latetime_luminosityfraction')]  # this is a constant value over the lc
            self._latetime_temperature = kwargs[self.key('latetime_temperature')]  # this is a constant value over the lc
        xc = self.X_CONST  # noqa: F841
        fc = self.FLUX_CONST  # noqa: F841
        cc = self.C_CONST

        # Some temp vars for speed.
        zp1 = 1.0 + kwargs[self.key('redshift')]
        Azp1 = u.Angstrom.cgs.scale / zp1
        czp1 = cc / zp1

        if self._include_latetime_luminosity:
            imax = np.argmax(self._luminosities)
            radius_phot = self._radius_phot[imax]
            temperature_phot = self._temperature_phot[imax]
            
            latetime_temperature = self._latetime_temperature
            wien_log_const = 3.67e-1 #cm*K, wien's law for lbda*Llbda (instead of just Llbda)
            lbda_max = wien_log_const/latetime_temperature # wavelength of max lbda*Llbda for late-time temp
            
            lbda_Llbda_max = lbda_max * (fc/u.Angstrom.cgs.scale * radius_phot**2 / lbda_max**5 /  #luminosity at lbda_max at tpeak, don't want scaled to angstrom
                            np.expm1(xc / lbda_max / temperature_phot) )

            # luminosity cannot exceed luminosity at peak in any band
            # note that different bands can peak at different times, so this might be smaller fraction of true peak lum in band centered on lbda 
            latetime_luminosity = self._latetime_luminosity_fraction * lbda_Llbda_max  #np.max(self._luminosities)
            
            # R = ( L/(T^4 * stefconst) )^(1/2)

            latetime_radiusphot = np.sqrt(latetime_luminosity/ latetime_temperature**4 /self.STEF_CONST)

        seds = []
        rest_wavs_dict = {}
        evaled = False

        for li, lum in enumerate(self._luminosities):
            bi = self._band_indices[li]
            if lum == 0.0:
                seds.append(np.zeros(len(
                    self._sample_wavelengths[bi]) if bi >= 0 else 1))
                continue

            if bi >= 0:
                rest_wavs = rest_wavs_dict.setdefault(
                    bi, self._sample_wavelengths[bi] * Azp1)
            else:
                rest_wavs = np.array(  # noqa: F841
                    [czp1 / self._frequencies[li]])

            radius_phot = self._radius_phot[li]  # noqa: F841
            temperature_phot = self._temperature_phot[li]  # noqa: F841

            if not evaled:
                if self._include_latetime_luminosity:
                    seds.append( ne.evaluate(
                        'fc * radius_phot**2 / rest_wavs**5 / '
                        'expm1(xc / rest_wavs / temperature_phot) + '
                        'fc * latetime_radiusphot**2 / rest_wavs**5 / '
                        'expm1(xc / rest_wavs / latetime_temperature)') )
                else:
                    seds.append( ne.evaluate(
                        'fc * radius_phot**2 / rest_wavs**5 / '
                        'expm1(xc / rest_wavs / temperature_phot)') )

                evaled = True
            else:
                try:
                    seds.append(ne.re_evaluate())

                except:
                    if self._include_latetime_luminosity:
                        seds.append(
                            (fc * radius_phot**2. / rest_wavs**5. / np.expm1(xc / rest_wavs / temperature_phot))+
                            (fc * latetime_radiusphot**2. / rest_wavs**5. / np.expm1(xc / rest_wavs / latetime_temperature))
                            )
                    else:
                        seds.append(
                            (fc * radius_phot**2. / rest_wavs**5. / np.expm1(xc / rest_wavs / temperature_phot)))
                    

            seds[-1][np.isnan(seds[-1])] = 0.0

        seds = self.add_to_existing_seds(seds, **kwargs)

        # Units of `seds` is ergs / s / Angstrom.
        return {'sample_wavelengths': self._sample_wavelengths, 'seds': seds}
