"""Definitions for the `Shankar` class."""
import numpy as np
from mosfit.modules.parameters.parameter import Parameter
from scipy.interpolate import interp1d
import os

# Important: Only define one ``Module`` class per file.


class Shankar(Parameter):
    """Shankar black hole mass prior (https://arxiv.org/pdf/astro-ph/0405585.pdf).

    Requires that redshift is constant. Only works for 0.02 <= z <= 5.99
    If z is outside the range, prior for min or max z in range is used.
    """

    def __init__(self, **kwargs):
        """Initialize module."""
        super(Shankar, self).__init__(**kwargs)

        self._z = 0.1696  # kwargs['redshift']
        datadir = os.path.dirname(__file__) + '/'
        (z, logM, logP) = np.genfromtxt(datadir + 'Shankar09.txt',
                                        usecols=(0, 1, 2), skip_header=27,
                                        unpack=True)

        logM = logM[z == z[0]]  # logM arrays are the same for all z

        # Find insertion point. Could also use bisect
        zarr = np.sort(list(set(z)))
        # for now if z is below or above z range, set to min or max z value
        if self._z < zarr[0]:
            self._z = zarr[0]
        if self._z > zarr[-1]:
            self._z = zarr[-1]
        if self._z not in zarr:  # need to interpolate
            zhi = zarr[zarr > self._z][0]
            zlo = zarr[zarr < self._z][-1]
            Pzhi = 10**logP[z == zhi]
            Pzlo = 10**logP[z == zlo]
            P = Pzlo + (self._z - zlo)/(zhi - zlo) * (Pzhi - Pzlo)

        else:
            P = 10**logP[z == self._z]

        # Need to change so logM includes self.min_value & self.max_value
        interpP = interp1d(logM, P)

        logMrun = np.linspace(np.log10(self._min_value),
                              np.log10(self._max_value),
                              num=25)  # number in file is 24
        Mrun = 10**logMrun
        # avoid numerical errors on bounds
        Mrun[0] = self._min_value
        Mrun[-1] = self._max_value
        Prun = interpP(logMrun)

        # need to integrate in logspace bc that is how PDF is defined
        cdfarr = ([0] +
                  [np.trapz(Prun[:i+2],
                   x=logMrun[:i+2]) for i in range(len(logMrun)-1)])

        cdfarr = np.array(cdfarr)
        # self._norm = 1. / np.trapz(P, x=logM)
        self._norm = 1. / cdfarr[-1]  # integral of PDF in mass range

        # create interp fn.s w/ M not logM bc stored that way in MOSFiT
        self._pdf = interp1d(Mrun, Prun/cdfarr[-1])  # normalized PDF
        self._cdf = interp1d(Mrun, cdfarr/cdfarr[-1])
        self._icdf = interp1d(cdfarr/cdfarr[-1], Mrun)

    def lnprior_pdf(self, x):
        """Evaluate natural log of probability density function."""
        value = self.value(x)
        # self._pdf takes log10(Mh)
        #if value == self._max_value:
        #    logvalue = 
        return(np.log(self._pdf(value)))

    def prior_icdf(self, u):
        """Evaluate inverse cumulative density function.

        output mass scaled to 0-1 interval
        Before scaling 10**5 <= Mh <= 10**8
        """
        value = self._icdf(u)

        value = (value - self._min_value) / (self._max_value - self._min_value)
        # np.clip in case of python errors in line above
        return np.clip(value, 0.0, 1.0)
