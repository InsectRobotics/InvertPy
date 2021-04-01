"""
Helpers for the InvertBrain package. Contains functions for whitening and random generators.
"""

from invertpy.__meta import *
from invertpy.__helpers import *

import numpy as np


def svd2pca(U, S, V, epsilon=10e-5):
    """
    Creates the PCA transformation matrix using the SVD.

    Parameters
    ----------
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray
    epsilon: float, optional

    Returns
    -------

    """
    return (np.diag(1. / np.sqrt(S + epsilon))).dot(U.T)


def svd2zca(U, S, V, epsilon=10e-5):
    """
    Creates the ZCA transformation matrix using the SVD.

    Parameters
    ----------
    U: np.ndarray
    S: np.ndarray
    V: np.ndarray
    epsilon: float, optional

    Returns
    -------

    """
    return U.dot(np.diag(1. / np.sqrt(S + epsilon))).dot(U.T)


def build_kernel(x, svd2ker, m=None, epsilon=10e-5, dtype='float32'):
    """
    Creates the transformation matrix of a dataset x using the given kernel function.

    Parameters
    ----------
    x: np.ndarray
    svd2ker: callable
    m: np.ndarray, optional
    epsilon: float, optional
        the smoothing parameter of the data
    dtype: np.dtype, optional

    Returns
    -------

    """
    shape = np.shape(x)

    # reshape the matrix in n x d, where:
    # - n: number of instances
    # - d: number of features

    x_flat = np.reshape(x, (shape[0], -1))
    n, d = np.shape(x_flat)

    # subtract the mean value from the data
    if m is None:
        m = np.mean(x_flat, axis=0)

    x_flat = x_flat - m

    # compute the correlation matrix
    C = np.dot(np.transpose(x_flat), x_flat) / n

    # compute the singular value decomposition
    U, S, V = np.linalg.svd(C)

    # compute kernel weights
    w = svd2ker(U, S, V, epsilon)

    return np.asarray(w, dtype=dtype)


def zca(x, shape=None, m=None, epsilon=10e-5, dtype='float32'):
    """
    The zero-phase component analysis (ZCA) kernel for whitening (Bell and Sejnowski, 1996).

    Parameters
    ----------
    x: np.ndarray
        the data to build the kernel from
    shape: list, optional
        the shape of the data
    m: np.ndarray, optional
        the mean values of the data
    epsilon: float, optional
        whitening constant, it prevents division by zero
    dtype: np.dtype, optional

    Returns
    -------
    w_zca: np.ndarray
        the ZCA whitening kernel
    """
    if shape is not None:
        x = x.reshape(tuple(shape))
    return build_kernel(x, svd2zca, m=m, epsilon=epsilon, dtype=dtype)


def pca(x, shape=None, m=None, epsilon=10e-5, dtype='float32'):
    """
    The principal component analysis (PCA) kernel for whitening.

    Parameters
    ----------
    x: np.ndarray
        the data to build the kernel from
    shape: list, optional
        the shape of the data
    m: np.ndarray, optional
        the mean values of the data
    epsilon: float, optional
        whitening constant, it prevents division by zero
    dtype: np.dtype, optional

    Returns
    -------
    w_pca: np.ndarray
        the PCA whitening kernel

    """
    if shape is not None:
        x = x.reshape(tuple(shape))
    return build_kernel(x, svd2pca, m=m, epsilon=epsilon, dtype=dtype)


def whitening(x, w=None, m=None, func=pca, epsilon=10e-5, reshape='first'):
    """
    Whitens the given data using the given parameters.
    By default it applies ZCA whitening.

    Parameters
    ----------
    x: np.ndarray
        the input data
    m: np.ndarray, optional
        the mean of the input data. If None, it is computed automatically.
    w: np.ndarray, optional
        the transformation matrix
    func: callable, optional
        the transformation we want to apply
    epsilon: float, optional
        whitening constant (10e-5 is typical for values around [-1, 1]
    reshape: str, optional
        the reshape option of the data; one of 'first' or 'last'. Default is first.

    Returns
    -------
    X: np.ndarray
        the transformed data.

    """
    if w is None:
        if 'first' in reshape:
            shape = (x.shape[0], -1)
        elif 'last' in reshape:
            shape = (-1, x.shape[-1])
        else:
            shape = None
        w = func(x, shape, epsilon)

    # whiten the input data
    shape = np.shape(x)
    x = np.reshape(x, (-1, np.shape(w)[0]))

    if m is None:
        m = np.mean(x, axis=0) if np.shape(x)[0] > 1 else np.zeros((1, np.shape(w)[0]))

    return np.reshape((x - m) @ w, shape)
