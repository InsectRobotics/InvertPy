"""
Package that contains some predefined plasticity (learning) rules.

References:
    .. [1] Rescorla, R. A. & Wagner, A. R. A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement
       and nonreinforcement. in 64–99 (Appleton-Century-Crofts, 1972).


    .. [2] Hebb, D. O. The organization of behavior: A neuropsychological theory. (Psychology Press, 2005).


    .. [3] Smith, D., Wessnitzer, J. & Webb, B. A model of associative learning in the mushroom body. Biol Cybern 99,
       89–103 (2008).
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

__init_dir__ = set(dir()) | {'__init_dir__'}


def dopaminergic(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    """
    The dopaminergic learning rule introduced in Gkanias et al (2021). Reinforcement here is assumed to be the
    dopaminergic factor.

        tau * dw / dt = rein * [r_pre + w(t) - w_rest]

        tau = 1 / learning_rate

    When DAN > 0 and KC > W - W_rest increase the weight (if DAN < 0 it is reversed).
    When DAN > 0 and KC < W - W_rest decrease the weight (if DAN < 0 it is reversed).
    When DAN = 0 no learning happens.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    rein: np.ndarray
        the dopaminergic factor.
    learning_rate: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights
    """
    if rein.ndim > 1:
        dop_fact = rein[:, np.newaxis, ...]
    else:
        dop_fact = rein[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    d_w = learning_rate * dop_fact * (r_pre + w - w_rest)
    if d_w.ndim > 2:
        d_w = d_w.sum(axis=0)
    return w + d_w


def prediction_error(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    """
    The prediction-error learning rule introduced in [1]_.

        tau * dw / dt = r_pre * (rein - r_post - w_rest)

        tau = 1 / learning_rate

    When KC > 0 and DAN > W - W_rest increase the weight (if KC < 0 it is reversed).
    When KC > 0 and DAN < W - W_rest decrease the weight (if KC < 0 it is reversed).
    When KC = 0 no learning happens.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    rein: np.ndarray
        the reinforcement signal.
    learning_rate: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights

    Notes
    -----
    .. [1] Rescorla, R. A. & Wagner, A. R. A theory of Pavlovian conditioning: Variations in the effectiveness of
       reinforcement and nonreinforcement. in 64–99 (Appleton-Century-Crofts, 1972).
    """
    if rein.ndim > 1:
        rein = rein[:, np.newaxis, ...]
        r_post = r_post[:, np.newaxis, ...]
    else:
        rein = rein[np.newaxis, ...]
        r_post = r_post[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    d_w = learning_rate * r_pre * (rein - r_post + w_rest)
    if d_w.ndim > 2:
        d_w = d_w.sum(axis=0)
    return w + d_w


def hebbian(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    """
    The Hebbian learning rule introduced in [1]_.

        tau * dw / dt = rein * r_pre x r_post + w_rest

        tau = 1 / learning_rate

    When DAN > 0 and MBON > 0 increase the weight
    When DAN <= 0 no learning happens

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    rein: np.ndarray
        the reinforcement signal.
    learning_rate: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights

    Notes
    -----
    .. [1] Hebb, D. O. The organization of behavior: A neuropsychological theory. (Psychology Press, 2005).

    """
    if rein.ndim > 1:
        rein = rein[:, np.newaxis, ...]
        r_post = r_post[:, np.newaxis, ...]
    else:
        rein = rein[np.newaxis, ...]
        r_post = r_post[np.newaxis, ...]
    r_pre = r_pre[..., np.newaxis]
    d_w = learning_rate * (rein * np.outer(r_pre, r_post) + w_rest)
    if d_w.ndim > 2:
        d_w = d_w.sum(axis=0)
    return w + d_w


def anti_hebbian(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    """
    The anti-Hebbian learning rule introduced in [1]_.

        tau * dw / dt = -rein * r_pre x r_post + w_rest

        tau = 1 / learning_rate

    When DAN > 0 and KC > 0 decrease the weight.
    When DAN <= 0 no learning happens.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    rein: np.ndarray
        the reinforcement signal.
    learning_rate: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights

    Notes
    -----
    .. [1] Smith, D., Wessnitzer, J. & Webb, B. A model of associative learning in the mushroom body. Biol Cybern 99,
       89–103 (2008).
    """
    if rein.ndim > 1:
        rein = rein[:, np.newaxis, ...]
    else:
        rein = rein[np.newaxis, ...]
    rein = np.maximum(rein, 0)
    r_pre = r_pre[..., np.newaxis]
    d_w = learning_rate * (-rein * (r_pre * w) + w_rest)
    if d_w.ndim > 2:
        d_w = d_w.sum(axis=0)
    return w + d_w


def infomax(w, r_pre, r_post, rein, learning_rate=1., w_rest=1.):
    """
    The infomax learning rule introduced in [1]_ and used for navigation in [2]_.

        tau * dw / dt = 1 / N * (w - (r_post + r_pre) * r_pre . w

        tau = 1 / learning_rate.

    Parameters
    ----------
    w: np.ndarray
        the current synaptic weights.
    r_pre: np.ndarray
        the pre-synaptic responses.
    r_post: np.ndarray
        the post-synaptic responses.
    rein: np.ndarray
        the reinforcement signal.
    learning_rate: float, optional
        the learning rate.
    w_rest: np.ndarray | float
        the resting value for the synaptic weights.

    Returns
    -------
    w_post: np.ndarray
        the updated synaptic weights

    Notes
    -----
    .. [1] Bell, A. & Sejnowski, T. An information-maximization approach to blind seperation and blind deconvolution.
    Neural Comput 7, 1129-1159 (1995).

    .. [2] Baddeley, B., Graham, P., Husbands, P. & Philippides, A. A Model of Ant Route Navigation Driven by Scene
    Familiarity. Plos Comput Biol 8, e1002336 (2012).
    """
    d_w = learning_rate * (w - (r_post + r_pre) * np.dot(r_pre, w))

    return w + d_w


__learning_rules__ = set(dir()) - __init_dir__
"""
Names of all the learning rules in this package. 
"""


def get_available_learning_rules():
    """
    Returns a list with the all the predefined learning rules that are available for use.

    Returns
    -------
    lrs: list
        a list with the names of all the available learning rules.

    """
    return list(__learning_rules__)


def get_learning_rule(learning_rule_name):
    """
    Transforms the name of a learning rule into the respective function if its implementation exists.

    Parameters
    ----------
    learning_rule_name: str
        the name of the learning rule.

    Returns
    -------
    learning_rule: callable
        the function of the learning rule as a callable.
    """
    if learning_rule_name is None:
        return None
    elif learning_rule_name in get_available_learning_rules():
        return eval(learning_rule_name)
    else:
        raise ValueError("Learning rule '%s' does not exist!" % learning_rule_name)
