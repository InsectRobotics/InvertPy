"""
Helper functions for the Central Complex. Includes helpful optic flow transformations.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

import numpy as np


def image_motion_flow(velocity, v_heading, r_sensor):
    """
    Calculate optic flow based on movement.

    Parameters
    ----------
    velocity: np.ndarray
        translation (velocity - 3D).
    v_heading: np.ndarray
        agent heading direction (3D vector).
    r_sensor: np.ndarray
        relative directions of sensors on the agent (3D vectors).

    Returns
    -------
    flow: np.ndarray
        the optic flow.

    """
    flow = velocity - (r_sensor.T * velocity.dot(r_sensor.T)).T
    flow -= rotary_flow(v_heading, r_sensor)
    return flow


def rotary_flow(v_heading, r_sensor):
    """
    Clockwise rotation.

    Parameters
    ----------
    v_heading: np.ndarray
        agent heading direction (3D vector).
    r_sensor: np.ndarray
        relative directions of sensors on the agent (3D vectors).

    Returns
    -------
    flow: np.ndarray
        the 3D optic flow based on rotation.

    """
    return np.cross(v_heading, r_sensor)


def translatory_flow(r_sensor, r_pref):
    """

    Parameters
    ----------
    r_sensor: np.ndarray
        relative directions of sensors on the agent (3D vectors)
    r_pref: np.ndarray
        agent's preferred direction

    Returns
    -------
    flow: np.ndarray
        the 3D optic flow based on translation.

    """
    return np.cross(np.cross(r_sensor, r_pref), r_sensor)


def linear_range_model(t_flow, r_flow, w=1., n=0.):
    """
    Eq 5 in Franz & Krapp

    Parameters
    ----------
    t_flow: np.ndarray
        translatory flow (wrt preferred direction).
    r_flow: np.ndarray
        image motion flow.
    w: float, optional
        weight.
    n: float, optional
        noise.

    Returns
    -------

    """
    return w * ((t_flow * r_flow).sum(axis=1) + n).sum()


def tn_axes(heading, tn_prefs=np.pi/4):
    """
    Create the axes of the TN neurons.

    Parameters
    ----------
    heading: float
        the heading direction (yaw) of the agent.
    tn_prefs: float, optional
        the preference angles of the TN neurons.

    Returns
    -------
    axes: np.ndarray
        the TN axes.
    """
    return np.array([[np.sin(heading - tn_prefs), np.cos(heading - tn_prefs)],
                     [np.sin(heading + tn_prefs), np.cos(heading + tn_prefs)]])


def get_flow(heading, velocity, r_sensors):
    """
    This is the longwinded version that does all the flow calculations,
    piece by piece. It can be refactored down to flow2() so use that for
    performance benefit.

    Parameters
    ----------
    heading: float
        the heading (azimuth) direction of the agent.
    velocity: np.ndarray
        the 2D velocity of the agent.
    r_sensors: np.ndarray
        relative directions of sensors on the agent (3D vectors).

    Returns
    -------

    """
    translation = np.append(velocity, np.zeros(1))
    rotation = np.zeros(3)
    img_flow = image_motion_flow(translation, rotation, r_sensors)
    tn_pref = tn_axes(heading)

    flow_tn_1 = translatory_flow(r_sensors, tn_pref[0])
    flow_tn_2 = translatory_flow(r_sensors, tn_pref[1])

    lr_1 = linear_range_model(flow_tn_1, img_flow, w=.1)
    lr_2 = linear_range_model(flow_tn_2, img_flow, w=.1)

    return np.array([lr_1, lr_2])


def decode_vector(vector, gain=0.05):
    vec_reshaped = vector.reshape((2, -1))
    vec_shifted = np.array([np.roll(vec_reshaped[0], 1, axis=-1),
                            np.roll(vec_reshaped[1], -1, axis=-1)])
    signal = np.sum(vec_shifted, axis=0)

    fund_freq = np.fft.fft(signal)[1]
    angle = -np.angle(np.conj(fund_freq))
    distance = np.absolute(fund_freq) / gain

    return distance * np.exp(1j * angle)
