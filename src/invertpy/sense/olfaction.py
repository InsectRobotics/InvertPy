from .sensor import Sensor

from scipy.spatial.transform.rotation import Rotation as R

import numpy as np


class Antennas(Sensor):
    def __init__(self, nb_tactile=1, nb_chemical=2, nb_chemical_dimensions=21, length=.01, tol=4e-01, *args, **kwargs):
        """
        The Antennas class is a representation of the pair of insect antennas as a simple sensor.

        It can have multiple chemical and tactile sensors across their length providing rich information of the
        environment.

        Parameters
        ----------
        nb_tactile : int, optional
            the number of tactile sensors in each antenna. Default is 1
        nb_chemical : int, optional
            the number of chemical sensors in each antenna. Default is 2
        nb_chemical_dimensions : int, optional
            the number of dimensions for each chemical sensors in each antenna. The more dimensions the more odours it
            can discriminate. Default is 21
        length : float, optional
            the length of each antenna (in meters). Default is 1cm
        tol : float
            the tolerance of accepted intensities. Default is 0.2
        """

        kwargs.setdefault('nb_output', (2, nb_chemical * nb_chemical_dimensions + nb_tactile))
        kwargs.setdefault('nb_input', kwargs['nb_output'])
        kwargs.setdefault('name', 'antennas')

        super().__init__(*args, **kwargs)

        self._ant_length = length
        self._ant_ori = None
        self._ant_base = None

        self._tactile_x = np.linspace(1, 0, nb_tactile, endpoint=False)
        self._chemical_x = np.linspace(1, 0, nb_chemical, endpoint=False)
        self._nb_dimensions = nb_chemical_dimensions
        self._tol = tol

        self._r = None
        self._r_tactile = None
        self._r_chemical = None

        self.reset()

    def reset(self):
        self._ant_ori = R.from_euler('ZY', [[-45, -30], [45, -30]], degrees=True)
        self._ant_base = np.array([[0.0005, -0.0005, 0],
                                   [0.0005, 0.0005, 0]])

        self._r = np.zeros(self._nb_output, dtype=self.dtype)  # initialise the responses
        self._r_tactile = self._r[:, :self.nb_tactile]
        self._r_chemical = self._r[:, self.nb_tactile:]

    def _sense(self, odours=None, scene=None):
        # initialise the responses
        r_chemical = np.zeros((2, self.nb_chemical * self._nb_dimensions), dtype=self.dtype)
        r_tactile = np.zeros((2, self.nb_tactile), dtype=self.dtype)

        # if odours are given extract the chemical intensities from them
        if odours is not None:
            if not isinstance(odours, list):
                odours = [odours]

            # initialise the odour intensities
            odour_intensities = np.zeros((2 * self.nb_chemical, self._nb_dimensions), self.dtype)
            # transform the relative position of each chemical sensor on the antenna to a local vector
            chemical_xyz = np.outer(self._chemical_x, [self._ant_length, 0, 0])

            pois_loc = np.zeros((2 * chemical_xyz.shape[0], chemical_xyz.shape[1]), dtype=self.dtype)
            # rotate the antenna-centric vectors to the head-centric system and add the base disposition
            pois_loc[:chemical_xyz.shape[0]] = self._ant_base[0] + self._ant_ori[0].apply(chemical_xyz)
            pois_loc[chemical_xyz.shape[0]:] = self._ant_base[1] + self._ant_ori[1].apply(chemical_xyz)
            # rotate the head-centric vectors to the global coordinate system and add the global position of the sensor
            pois_glob = self.xyz.reshape((1, -1)) + self.ori.apply(pois_loc)
            for i, odour in enumerate(odours):
                odour_intensities[:, i] = odour(pois_glob)
            odour_intensities[odour_intensities < self._tol] = 0.

            r_chemical[0] = odour_intensities[:chemical_xyz.shape[0]].flatten()  # left antenna
            r_chemical[1] = odour_intensities[chemical_xyz.shape[0]:].flatten()  # right antenna

        # tactile sensing is not supported yet
        if scene is not None:
            # here it should implement collision detection for the vegetation
            pass

        self._r[:] = np.hstack([r_tactile, r_chemical])

        return self._r

    def __repr__(self):
        return ("Antennas(units=%d, tactile=%.0f, chemical=%d, dimensions=%d, "
                "pos=(%.2f, %.2f, %.2f), ori=(%.2f, %.2f, %.2f), name='%s')") % (
            self.nb_antennas, self.nb_tactile, self.nb_chemical, self._nb_dimensions,
            self.x, self.y, self.z, self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    @property
    def antennas_tip(self):
        """
        The end effector of each of the antennas.

        Returns
        -------
        np.ndarray[float]
        """
        return self._ant_base + self._ant_ori.apply([self._ant_length, 0, 0])

    @property
    def antenna_ori(self):
        """
        The orientation of each of the antennas.

        Returns
        -------
        R
        """
        return self._ant_ori

    @property
    def nb_antennas(self):
        """
        The number of antennas.

        Returns
        -------
        int
        """
        return 2

    @property
    def nb_tactile(self):
        """
        The number of tactile sensors in each antenna.

        Returns
        -------
        int
        """
        return len(self._tactile_x)

    @property
    def nb_chemical(self):
        """
        The number of chemical sensors in each antenna.

        Returns
        -------
        int
        """
        return len(self._chemical_x)

    @property
    def responses(self):
        """
        The latest responses generated by the antennas.

        Returns
        -------
        np.ndarray[float]
        """
        return self._r

    @property
    def responses_t(self):
        """
        The latest tactile responses generated by the antennas.

        Returns
        -------
        np.ndarray[float]
        """
        return self._r_tactile

    @property
    def responses_c(self):
        """
        The latest chemical responses generated by the antennas.

        Returns
        -------
        np.ndarray[float]
        """
        return self._r_chemical

    @property
    def tolerance(self):
        """
        The lowest detectable odour intensity.

        Returns
        -------
        float
        """
        return self._tol
