import invertpy.brain.centralcomplex.fanshapedbody as fb

import loguru as lg
import numpy as np

AVOGADRO_CONSTANT = 6.02214076e+23  # /mol
PLANK_CONSTANT = 6.62597915e-34  # J s
SPEED_OF_LIGHT = 299792458  # m/s


class PathIntegrationDyeLayer(fb.PathIntegratorLayer):
    """
    Ceberg and Nilsson model.
    """

    epsilon: np.ndarray[float]
    """The molar absorption coefficient."""

    length: np.ndarray[float]
    """The optical path length through the sample."""

    k: np.ndarray[float]
    """The rate coefficient (related to half-life as k = log(2) / T_half)."""

    phi: np.ndarray[float]
    """The proportion of the absorbed light that leads to switching (quantum yield)."""

    c_tot: np.ndarray[float]
    """The total concentration of dye molecules per unit."""

    model_transmittance: bool
    """
    If True, the transmittance is computed as T = 10^(-A), A = epsilon * length * (c_tot - c).
    If False, this is simplified to T = c / c_tot.
    """

    def __init__(self, *args, epsilon=1.0, length=10e-04, T_half=1.0, k=None, beta=0, phi=0, c_tot=0.3,
                 volume=None, wavelength=None, w_max=None,
                 parameter_noise=0.0, model_transmittance=True, mem_initial=None, **kwargs):
        """

        Parameters
        ----------
        epsilon: float
            the molar absorption coefficient.
        length: float
            the optical path length through the sample.
        T_half: float
            the half-life of the molecules in their OFF state (the metastable photostationary state)
        k: np.ndarray[float], float, None
            the rate coefficient (related to half-life as k = log(2) / T_half).
        beta: float
            the background activity.
        phi: float
            the proportion of the absorbed light that leads to switching (quantum yield).
        c_tot: float
            the total concentration of dye molecules per unit.
        volume
        wavelength
        w_max
        parameter_noise: float
            the noise to add to the parameters.
        model_transmittance: bool
            If True, the transmittance is computed as T = 10^(-A), A = epsilon * length * (c_tot - c).
            If False, this is simplified to T = c / c_tot.
        mem_initial: np.ndarray[float], float, None
            the initial memory. Default is 0.
        start_at_stable: bool
            If True and the mem_initial is None, it calculates a stable initial memory.
        """

        kwargs.setdefault('gain', 1.0)
        kwargs.setdefault('noise', 0.0)  # 0.1

        fb.PathIntegratorLayer.__init__(self, *args, **kwargs)

        self.epsilon = noisify_column_parameter(epsilon, parameter_noise, self.nb_fbn)
        self.length = noisify_column_parameter(length, parameter_noise, self.nb_fbn)
        self.k = noisify_column_parameter(np.log(2) / T_half if k is None else k, parameter_noise, self.nb_fbn)

        if wavelength is None or volume is None or w_max is None:
            self.k_phi = 1.
        else:
            E = PLANK_CONSTANT * SPEED_OF_LIGHT / wavelength  # (J) energy of the photon
            self.k_phi = w_max / (E * volume * AVOGADRO_CONSTANT)  # M/s
        self.phi = noisify_column_parameter(phi, parameter_noise, self.nb_fbn)
        self.c_tot = noisify_column_parameter(c_tot, parameter_noise, self.nb_fbn)

        self.model_transmittance = model_transmittance
        self.last_c = np.zeros_like(self.r_fbn)

        if mem_initial:
            self.reset_integrator(mem_initial)

        self.beta = beta
        self._f_fbn_inter = super().f_cpu4
        self.f_fbn = lambda x: x
        self.update = True
        self.m_cpu4 = 0.

    def transmittance(self, c):
        """
        The transmittance corresponds to the weight of the synapse

        Parameters
        ----------
        c: np.ndarray[float]
            the OFF-state concentration (c_OFF)

        Returns
        -------
        np.ndarray[float]
            the transmittance
        """

        return transmittance(c, self.epsilon, self.length, self.c_tot)

    def dcdt(self, u):
        """

        Parameters
        ----------
        u: np.ndarray[float]
            the PFN output, i.e., its normalised activity

        Returns
        -------
        Callable
            the dc/dt function.
        """

        return dcdt(u, self.transmittance, k=self.k, phi=self.phi)

    def reset_integrator(self, c0=None):
        fb.PathIntegratorLayer.reset_integrator(self)
        if c0 is None:
            self.last_c[:] = np.zeros_like(self.last_c)
        else:
            self.last_c[:] = np.ones_like(self.last_c) * c0

    def mem_update(self, mem, dt=1.):

        self.m_cpu4 = mem_int = self._f_fbn_inter(mem * 500)  # cpu4 activity
        mem = np.clip(mem_int * self.gain + self.beta, 0, 1)

        self.last_c = np.clip(self.last_c + self.dcdt(mem)(0, self.last_c) * dt, 0, 1)

        return self.cpu4_mem

    @property
    def cpu4_mem(self):
        return self.transmittance(self.last_c)


class PathIntegrationDyeLayer2(fb.PathIntegratorLayer):

    epsilon: np.ndarray[float]
    """The molar absorption coefficient."""

    length: np.ndarray[float]
    """The optical path length through the sample."""

    k: np.ndarray[float]
    """The rate coefficient (related to half-life as k = log(2) / T_half)."""

    phi: np.ndarray[float]
    """The proportion of the absorbed light that leads to switching (quantum yield)."""

    c_tot: np.ndarray[float]
    """The total concentration of dye molecules per unit."""

    def __init__(self, *args, epsilon=1.0, length=10e-04, T_half=1.0, k=None, beta=0, phi=0, c_tot=0.3,
                 # volume=1e-18, wavelength=750, W_max=1e-15,  # unused parameters
                 pfn_weight_factor=1, mem_initial=None, **kwargs):
        """

        Parameters
        ----------
        epsilon: float
            the molar absorption coefficient.
        length: float
            the optical path length through the sample.
        T_half: float
            the half-life of the molecules in their OFF state (the metastable photostationary state)
        k: np.ndarray[float], float, None
            the rate coefficient (related to half-life as k = log(2) / T_half).
        beta: float
            the background activity.
        phi: float
            the proportion of the absorbed light that leads to switching (quantum yield).
        c_tot: float
            the total concentration of dye molecules per unit.
        volume
        wavelength
        W_max
        pfn_weight_factor: float
            the memory gain.
        mem_initial: np.ndarray[float], float, None
            the initial memory. Default is 0.
        """

        kwargs.setdefault('gain', pfn_weight_factor)
        kwargs.setdefault('noise', 0.0)  # 0.1

        fb.PathIntegratorLayer.__init__(self, *args, **kwargs)

        self.epsilon = noisify_column_parameter(epsilon, 0.0, self.nb_fbn)
        self.length = noisify_column_parameter(length, 0.0, self.nb_fbn)
        self.k = noisify_column_parameter(np.log(2) / T_half if k is None else k, 0.0, self.nb_fbn)

        # E = 6.62697915 * 1e-34 * 299792458 / (wavelength * 1e-9)
        # self.k_phi = W_max / (E * volume * 6.02214076*1e23)
        self.phi = noisify_column_parameter(phi, 0.0, self.nb_fbn)
        self.c_tot = noisify_column_parameter(c_tot, 0.0, self.nb_fbn)

        self.last_c = np.zeros_like(self.r_fbn)

        if mem_initial:
            self.reset_integrator(mem_initial)

        self.beta = beta
        self.update = False

    def _fprop(self, delta7=None, tb1=None, tn1=None, tn2=None):
        if delta7 is not None:
            tb1 = delta7
        if tb1 is None:
            tb1 = self.r_tb1
        if tn1 is None:
            tn1 = self.r_tn1
        if tn2 is None:
            tn2 = self.r_tn2

        # Idealised setup, where we can negate the TB1 sinusoid for memorising backwards motion
        mem_tn1 = (.5 - tn1).dot(self.w_tn12cpu4)
        mem_tb1 = (tb1 - 1).dot(self.w_p2f)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_tn2 = 0.25 * tn2.dot(self.w_tn22cpu4)

        mem = mem_tn1 * mem_tb1 - mem_tn2

        self.r_tb1 = tb1
        self.r_tn1 = tn1
        self.r_tn2 = tn2
        self.r_cpu4 = a_cpu4 = self.f_cpu4(mem * 500)

        mem = np.clip(a_cpu4 * self.gain + self.beta, 0, 1)

        self.last_c = np.clip(self.last_c + self.dcdt(mem)(0, self.last_c), 0, 1)

        self.w_tn22cpu4[0, :8] = 1 - self.last_c[:8]
        self.w_tn22cpu4[1, 8:] = 1 - self.last_c[8:]
        self.w_tn12cpu4[0, :8] = 1 - self.last_c[:8]
        self.w_tn12cpu4[1, 8:] = 1 - self.last_c[8:]

        return a_cpu4

    def transmittance(self, c):
        """
        The transmittance corresponds to the weight of the synapse

        Parameters
        ----------
        c: np.ndarray[float]
            the OFF-state concentration (c_OFF)

        Returns
        -------
        np.ndarray[float]
            the transmittance
        """

        return transmittance(c, self.epsilon, self.length, self.c_tot)

    def dcdt(self, u):
        """

        Parameters
        ----------
        u: np.ndarray[float]
            the PFN output, i.e., its normalised activity

        Returns
        -------
        Callable
            the dc/dt function.
        """

        return dcdt(u, self.transmittance, k=self.k, phi=self.phi)

    def reset_integrator(self, c0=None):
        fb.PathIntegratorLayer.reset(self)
        if c0 is not None:
            self.last_c[:] = np.ones_like(self.last_c) * c0
        else:
            self.last_c[:] = np.zeros_like(self.last_c)

    @property
    def cpu4_mem(self):
        return self.transmittance(self.last_c)


def dcdt(u, transmittance_func, k=0.0, phi=0.00045, k_phi=1.0):
    """

    Parameters
    ----------
    u: np.ndarray[float]
        the PFN output, i.e., its normalised activity
    transmittance_func
    k: np.ndarray[float], float
    phi: np.ndarray[float], float
    k_phi: np.ndarray[float], float

    Returns
    -------
    Callable
        the dc/dt function.
    """

    def f(t, c):
        """

        Parameters
        ----------
        t: float
            time
        c: np.ndarray[float]
            the OFF-state concentration (c_OFF)

        Returns
        -------
        np.ndarray[float]
            the concentration change (dc/dt)
        """
        T = transmittance_func(c)
        # -k * c: the first-order back-reaction

        return -k * c + u * (1 - T) * phi * k_phi

    return f


def transmittance(c, epsilon=1e+04, length=1e-03, c_tot=0.3):
    """
    The transmittance corresponds to the weight of the synapse

    Parameters
    ----------
    c: np.ndarray[float]
        the OFF-state concentration (c_OFF)
    epsilon: np.ndarray[float], float
    length: np.ndarray[float], float
    c_tot: np.ndarray[float], float

    Returns
    -------
    np.ndarray[float]
        the transmittance
    """

    return 10 ** -absorbance(c, epsilon, length, c_tot)


def absorbance(c, epsilon=1e+04, length=1e-03, c_tot=0.3):
    return epsilon * length * (c_tot - c)


def noisify_column_parameter(param, noise, shape=None):
    # the noise is proportional to the parameter value (!?)
    return param + np.random.normal(0, noise * param, shape) if noise > 0.0 else param

