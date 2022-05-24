"""
Handlers for saving and loading sensor parameters.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "v1.0.0-alpha"
__maintainer__ = "Evripidis Gkanias"

from scipy.spatial.transform import Rotation as R

import numpy as np
import os


__dir__ = os.path.dirname(os.path.realpath(__file__))
"""
The local directory.
"""

__data_dir__ = os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'data'))
"""
Directory holding the data.
"""


def save_eye2csv(eye, filename=None):
    """Save the parameters of the eye in a CSV file.

    The parameters are saved in the {ROOT}/data/eye/{filename}.csv file.
    Each row is a different ommatidium and each column a different property of the ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle (in degrees),
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    eye
        the compound eye.
    filename: str, optional
        the name of the file to save the parameters. If None, it is the name of the eye.
    """
    if filename is None:
        filename = eye.name
    if not os.path.exists(os.path.dirname(filename)):
        filename = filename.replace('.csv', '')
        fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    else:
        fall = filename
    xyz = eye.omm_xyz
    q = eye.omm_ori.as_quat()
    rho = np.rad2deg(eye.omm_rho.reshape((-1, 1)))
    pol = eye.omm_pol.reshape((-1, 1))
    res = eye.omm_responsiveness.reshape((-1, 1))
    hue = eye.hue_sensitive
    if hue.shape[0] == 1:
        hue = np.vstack([hue] * xyz.shape[0])

    data = np.hstack([xyz, q, rho, pol, res, hue])

    np.savetxt(fall, data, delimiter=',')


def load_csv2eye(eye, filename, replace_name=True):
    """
    Load the parameters of the eye from a CSV file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    eye
        the compound eye to load the data to.
    filename: str
        the name of the file to load the parameters from.
    replace_name: bool, optional
        whether to replace the name of the eye with the name of the file.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')

    data = np.genfromtxt(fall, delimeter=',')
    eye._omm_xyz = data[..., 0:3]
    eye._omm_ori = R.from_quat(data[..., 3:7])
    eye._omm_rho = np.deg2rad(data[..., 7])
    eye._omm_pol = data[..., 8]
    eye._omm_res = data[..., 9]
    eye._c_sensitive = data[..., 10:15]

    if replace_name:
        eye.name = filename


def load_ommatidia_xyz(filename):
    """
    Loads the 3D positions of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.

    Returns
    -------
    xyz: np.ndarray
        a Nx3 matrix with the x, y and z positions of each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., :3]


def load_ommatidia_ori(filename):
    """
    Loads the 3D orientation of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.

    Returns
    -------
    ori: R
        a Rotation instance with N rotations, one of each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return R.from_quat(data[..., 3:7])


def load_ommatidia_rho(filename, degrees=True):
    """
    Loads the acceptance angle of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.
    degrees: bool, optional
        whether the acceptance angle should be returned in degrees or not.

    Returns
    -------
    rho: np.ndarray
        a N-dimensional array with the acceptance angles of each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 7] if degrees else np.deg2rad(data[..., 7])


def load_ommatidia_pol(filename):
    """
    Loads the polarisation sensitivity of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.

    Returns
    -------
    rho: np.ndarray
        a N-dimensional array with the polarisation sensitivity of each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 8]


def load_ommatidia_res(filename):
    """
    Loads the responsiveness of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.

    Returns
    -------
    rho: np.ndarray
        a N-dimensional array with the polarisation sensitivity of each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 9]


def load_ommatidia_irgbu(filename):
    """
    Loads the spectral sensitivity of the ommatidia from a file.
    The parameters are loaded from the {ROOT}/data/eye/{filename}.csv file.
    In the CSV file, each row should represent a different ommatidium and each column a different property of the
    ommatidium as follows:

    x, y, z, q_w , q_x, q_y, q_z, rho, pol, res, IR, R, G, B, UV

    where x, y, z is the position of the ommatidium in local coordinates,
    q_w, q_x, q_y, q_z is its orientation (in quaternion terms) relative to the eye's orientation,
    'rho' is its acceptance angle,
    'pol' is its polarisation sensitivity,
    'res' is its responsiveness, and
    IR, R, G, B, UV is its spectral sensitivity.

    Parameters
    ----------
    filename: str
        the name of the file.

    Returns
    -------
    rho: np.ndarray
        a Nx5 matrix with the spectral sensitivity for each of the ommatidia.
    """
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 10:15]


def reset_data_directory(data_dir):
    """
    Sets up the default directory of the data.

    Parameters
    ----------
    data_dir : str
        the new directory path.
    """
    global __data_dir__
    __data_dir__ = data_dir
