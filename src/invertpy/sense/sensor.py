"""
Abstract class for sensors. Contains the basic attributes of the sensor, such as the position and orientation of the
sensor, the type of values, the random generator, the name and noise. It also implements some basic physics such as
translation and rotation of the sensor.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from ._helpers import RNG

from scipy.spatial.transform import Rotation as R
from copy import copy

import numpy as np


class Sensor(object):
    def __init__(self, nb_input, nb_output, xyz=None, ori=None, noise=0., rng=RNG, dtype='float32', name="sensor"):
        """
        Abstract class of a sensor that implements its basic functions and sets the abstract methods that need
        to be implemented by every sensor (i.e. the 'reset' and 'sense').

        Parameters
        ----------
        nb_input: int
            the number of input elements that observe with the world.
        nb_output: int, tuple
            the number of output elements that represent the observed values. It could also be the shape of output in
            the form of a tuple.
        xyz: list, np.ndarray
            the 3D position of the sensor.
        ori: R, optional
            the 3D orientation of the sensor.
        noise: float, optional
            the noise introduced in the responses of the component.
        rng: np.random.RandomState, optional
            an instance of the numpy.random.RandomState that will be used in order to generate random patterns.
        dtype: np.dtype, optional
            the type of the values used in this component.
        name: str, optional
            the name of the sensor.
        """
        self._nb_input = nb_input
        self._nb_output = nb_output
        self._xyz = xyz if xyz is not None else np.zeros(3, dtype=dtype)
        self._ori = ori if ori is not None else R.from_euler('z', 0, degrees=False)
        self.noise = noise
        self.dtype = dtype

        self.rng = rng
        self.name = name

    def reset(self):
        """
        This method is called whenever we want to re-initialise the sensor. It should implement the initialisation of
        the input and output elements of the sensor and other attributes useful for the calculations in teh sensing
        process that might change values when the sensor is used.
        """
        raise NotImplementedError()

    def _sense(self, *env, **kwenv):
        """
        This method should process the input environments and create the responses of the sensor.

        Parameters
        ----------
        env
            the positional arguments for input environments.
        kwenv
            the named erguments for input environments.

        Returns
        -------
        out: np.ndarray
            the responses produced by the sensor when it observed the environment.
        """
        raise NotImplementedError()

    def __call__(self, *env, callback=None, **kwenv):
        """
        When the sensor is called, the sensing process is executed and the output is calculated. Then the
        callback function is called (if provided), which gets as input the instance of the sensor itself. Finally,
        the output is returned.

        Parameters
        ----------
        env
            the positional arguments for input environments.
        kwenv
            the named erguments for input environments.
        callback: callable, optional
            Customised processing of the component every time that the component is called. It gets as input the
            component itself.

        Returns
        -------
        r_out: np.ndarray
            The output of the sensor given the input environments.
        """
        out = self._sense(*env, **kwenv)

        if callback is not None:
            callback(self)

        return out

    def copy(self):
        """
        Creates a clone of the instance.

        Returns
        -------
        copy: Sensor
            another instance of exactly the same class and parameters.
        """
        return copy(self)

    def __copy__(self):
        nb_in = self._nb_input
        nb_out = self._nb_output[0] if isinstance(self._nb_output, tuple) else self._nb_output
        sensor = self.__class__(nb_input=nb_in, nb_output=nb_out)
        for att in self.__dict__:
            sensor.__dict__[att] = copy(self.__dict__[att])

        return sensor

    def __repr__(self):
        return "Sensor(in=%d, out=%d, pos=(%.2f, %.2f, %.2f), ori=(%.2f, %.2f, %.2f), name='%s')" % (
            self._nb_input, self._nb_output,
            self.x, self.y, self.z,
            self.yaw_deg, self.pitch_deg, self.roll_deg, self.name
        )

    def rotate(self, d_ori, around_xyz=None):
        """
        Rotated the sensor around a 3D point.

        Parameters
        ----------
        d_ori: R
            the rotation to be applied
        around_xyz: np.ndarray, list
            the point around which the sensor will be rotated.
        """
        if around_xyz is None:
            around_xyz = [0, 0, 0]
        around_xyz = np.array(around_xyz)
        self._xyz = around_xyz + d_ori.apply(self._xyz - around_xyz)
        self._ori = self.ori * d_ori

    def translate(self, d_xyz):
        """
        Translates the sensor towards the 3D direction provided.

        Parameters
        ----------
        d_xyz: np.ndarray, list
            the 3D point that will be added to the position of the sensor.
        """
        self._xyz += np.array(d_xyz, dtype=self.dtype)

    @property
    def xyz(self):
        """
        The 3D position of the sensor.
        """
        return self._xyz

    @xyz.setter
    def xyz(self, v):
        """
        The position of the agent.

        Parameters
        ----------
        v: np.ndarray[float]

        See Also
        --------
        Agent.position
        """
        self._xyz[:] = v

    @property
    def x(self):
        """
        The position of the sensor in the X axis.
        """
        return self._xyz[0]

    @property
    def y(self):
        """
        The position of the sensor in the Y axis.
        """
        return self._xyz[1]

    @property
    def z(self):
        """
        The position of the sensor in the Z axis.
        """
        return self._xyz[2]

    @property
    def ori(self):
        """
        The 3D orientation of the sensor.
        """
        return self._ori

    @ori.setter
    def ori(self, v):
        """
        Parameters
        ----------
        v: R

        See Also
        --------
        Agent.orientation
        """
        self._ori = v

    @property
    def euler(self):
        """
        The 3D orientation of the sensor in Euler angles (yaw, pitch, roll) in rads.
        """
        return self._ori.as_euler('ZYX', degrees=False)

    @property
    def yaw(self):
        """
        The yaw orientation of the sensor in rads.
        """
        return self.euler[0]

    @property
    def pitch(self):
        """
        The pitch orientation of the sensor in rads.
        """
        return self.euler[1]

    @property
    def roll(self):
        """
        The roll orientation of the sensor in rads.
        """
        return self.euler[2]

    @property
    def euler_deg(self):
        """
        The 3D orientation of the sensor in Euler angles (yaw, pitch, roll) in degrees.
        """
        return self._ori.as_euler('ZYX', degrees=True)

    @property
    def yaw_deg(self):
        """
        The yaw orientation of the sensor in degrees.
        """
        return self.euler_deg[0]

    @property
    def pitch_deg(self):
        """
        The pitch orientation of the sensor in degrees.
        """
        return self.euler_deg[1]

    @property
    def roll_deg(self):
        """
        The roll orientation of the sensor in degrees.
        """
        return self.euler_deg[2]

    @property
    def position(self):
        """
        The 3D position of the sensor (the same as 'xyz').
        """
        return self._xyz

    @property
    def orientation(self):
        """
        The 3D orientation of the sensor (the same as 'ori').
        """
        return self._ori
