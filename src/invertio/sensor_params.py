from scipy.spatial.transform import Rotation as R

import numpy as np
import os


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data_dir__ = os.path.abspath(os.path.join(__dir__, '..', '..', 'data'))


def save_eye2csv(eye, filename: str = None):
    if filename is None:
        filename = eye.name
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    xyz = eye.omm_xyz
    q = eye.omm_ori.as_quat()
    rho = eye.omm_rho.reshape((-1, 1))
    pol = eye.omm_pol.reshape((-1, 1))
    hue = eye.hue_sensitive
    if hue.shape[0] == 1:
        hue = np.vstack([hue] * xyz.shape[0])

    data = np.hstack([xyz, q, rho, pol, hue])

    np.savetxt(fall, data, delimiter=',')


def load_csv2eye(eye, filename: str, replace_name=True):
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')

    data = np.genfromtxt(fall, delimeter=',')
    eye._omm_xyz = data[..., 0:3]
    eye._omm_ori = R.from_quat(data[..., 3:7])
    eye._omm_rho = data[..., 7]
    eye._omm_pol = data[..., 8]
    eye._c_sensitive = data[..., 9:14]

    if replace_name:
        eye.name = filename


def load_ommatidia_xyz(filename: str):
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., :3]


def load_ommatidia_ori(filename: str):
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return R.from_quat(data[..., 3:7])


def load_ommatidia_rho(filename: str):
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 7]


def load_ommatidia_pol(filename: str):
    filename = filename.replace('.csv', '')
    fall = os.path.join(__data_dir__, 'eyes', filename + '.csv')
    data = np.genfromtxt(fall, delimiter=',')
    return data[..., 8]
