"""
Package that contains some predefined plasticity (learning) rules.
"""

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2021, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"

import numpy as np

__init_dir__ = set(dir()) | {'__init_dir__'}


def dopaminergic(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    # When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed)
    # When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed)
    # When DAN = 0 no learning happens
    dop_fact = rein[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    return w + learning_rate * dop_fact * (r_pre + w - w_rest)


def prediction_error(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    # When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed)
    # When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed)
    # When KC = 0 no learning happens
    rein = rein[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    r_post = r_post[np.newaxis, ...]
    return w + learning_rate * r_pre * (rein - r_post + w_rest)


def hebbian(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    # When DAN > 0 and MBON > 0 increase the weight
    # When DAN <= 0 no learning happens
    rein = rein[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    r_post = r_post[np.newaxis, ...]
    return w + learning_rate * (rein * np.outer(r_pre, r_post) + w_rest)


def anti_hebbian(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    # When DAN > 0 and KC > 0 decrease the weight
    # When DAN <= 0 no learning happens
    rein = np.maximum(rein[np.newaxis, ...], 0)
    r_pre = r_pre[..., np.newaxis]
    return w + learning_rate * (-rein * (r_pre * w) + w_rest)


__learning_rules__ = set(dir()) - __init_dir__


def get_available_learning_rules():
    return list(__learning_rules__)


def get_learning_rule(learning_rule_name):
    if learning_rule_name in get_available_learning_rules():
        return eval(learning_rule_name)
    else:
        raise ValueError("Learning rule '%s' does not exist!" % learning_rule_name)
