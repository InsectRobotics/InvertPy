import numpy as np


def get_steering(cx) -> float:
    """
    Outputs a scalar where sign determines left or right turn.

    Parameters
    ----------
    cx

    Returns
    -------

    """

    cpu1a = cx.r_cpu1[1:-1]
    cpu1b = np.array([cx.r_cpu1[-1], cx.r_cpu1[0]])
    motor = cpu1a @ cx.w_cpu1a2motor + cpu1b @ cx.w_cpu1b2motor
    output = motor[0] - motor[1]  # * .25  # to kill the noise a bit!
    return output


def image_motion_flow(velocity, v_heading, r_sensor):
    """
    :param velocity: translation (velocity - 3D)
    :type velocity: np.ndarray
    :param v_heading: agent heading direction (3D vector)
    :type v_heading: np.ndarray
    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    Calculate optic flow based on movement.
    """
    flow = velocity - (r_sensor.T * velocity.dot(r_sensor.T)).T
    flow -= rotary_flow(v_heading, r_sensor)
    return flow


def rotary_flow(v_heading, r_sensor):
    """
    Clockwise rotation
    :param v_heading: agent heading direction (3D vector)
    :type v_heading: np.ndarray
    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    :return:
    """
    return np.cross(v_heading, r_sensor)


def translatory_flow(r_sensor, r_pref):
    """
    :param r_sensor: relative directions of sensors on the agent (3D vectors)
    :type r_sensor: np.ndarray
    :param r_pref: agent's preferred direction
    :type r_pref: np.ndarray
    :return:
    """
    return np.cross(np.cross(r_sensor, r_pref), r_sensor)


def linear_range_model(t_flow, r_flow, w=1., n=0.):
    """
    Eq 5 in Franz & Krapp
    :param t_flow: translatory flow (wrt preferred direction)
    :type t_flow: np.ndarray
    :param r_flow: image motion flow
    :type r_flow: np.ndarray
    :param w: weight
    :type w: float
    :param n: noise
    :type n: float
    :return:
    """
    return w * ((t_flow * r_flow).sum(axis=1) + n).sum()


def tn_axes(heading, tn_prefs=np.pi/4):
    return np.array([[np.sin(heading - tn_prefs), np.cos(heading - tn_prefs)],
                     [np.sin(heading + tn_prefs), np.cos(heading + tn_prefs)]])


def get_flow(heading, velocity, r_sensors):
    """
    This is the longwinded version that does all the flow calculations,
    piece by piece. It can be refactored down to flow2() so use that for
    performance benefit.
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
