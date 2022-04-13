"""
Helpers for the invertpy.brain package. Contains functions for whitening and random generators.
"""

from invertpy.__helpers import *

import numpy as np


def svd2pca(U, S, V, epsilon=1e-5):
    """
    Creates the PCA transformation matrix using the SVD.

    Parameters
    ----------
    U: np.ndarray[float]
    S: np.ndarray[float]
    V: np.ndarray[float]
    epsilon: float, optional

    Returns
    -------
    np.ndarray[float]
    """
    D = np.diag(np.sqrt(S + epsilon))
    return U.dot(np.linalg.inv(D))


def svd2zca(U, S, V, epsilon=1e-5):
    """
    Creates the ZCA transformation matrix using the SVD.

    Parameters
    ----------
    U: np.ndarray[float]
    S: np.ndarray[float]
    V: np.ndarray[float]
    epsilon: float, optional

    Returns
    -------
    np.ndarray[float]
    """
    D = np.diag(np.sqrt(S + epsilon))
    return U.dot(np.linalg.inv(D)).dot(U.T)


def eig2pca(E, V, epsilon=1e-5):
    D = np.diag(np.sqrt(V + epsilon))
    return E.dot(np.linalg.inv(D))


def eig2zca(E, V, epsilon=1e-5):
    D = np.diag(np.sqrt(V + epsilon))
    return E.dot(np.linalg.inv(D)).dot(E.T)


def build_kernel(x, nb_out=None, svd2ker=None, eig2ker=None, m=None, epsilon=1e-5, dtype='float32'):
    """
    Creates the transformation matrix of a dataset x using the given kernel function.

    Parameters
    ----------
    x: np.ndarray
    svd2ker: callable
    eig2ker: callable
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
    n, d = x_flat.shape
    if nb_out is None:
        nb_out = n

    # subtract the mean value from the data
    if m is None:
        m = np.mean(x_flat, axis=0)

    x_mean = x_flat - m

    # compute the correlation matrix
    C = np.cov(x_mean, rowvar=False)
    # C = np.dot(np.transpose(x_flat), x_flat) / n

    if eig2ker is not None:
        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(C)

        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        V = eigen_values[sorted_index]
        E = eigen_vectors[:, sorted_index]

        # compute kernel weights
        w = eig2ker(E[:, :nb_out], V[:nb_out], epsilon)
    elif svd2ker is not None:
        # compute the singular value decomposition
        U, S, V = np.linalg.svd(C)

        # compute kernel weights
        w = svd2ker(U[:, :nb_out], S[:nb_out], V, epsilon)
    else:
        w = np.linalg.inv(C + np.diag(epsilon))

    return np.asarray(w, dtype=dtype)


def zca(x, nb_out=None, shape=None, m=None, epsilon=1e-5, method="eig", dtype='float32'):
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
    if method == "svd":
        return build_kernel(x, nb_out=nb_out, svd2ker=svd2zca, m=m, epsilon=epsilon, dtype=dtype)
    else:
        return build_kernel(x, nb_out=nb_out, eig2ker=eig2zca, m=m, epsilon=epsilon, dtype=dtype)


def pca(x, nb_out=None, shape=None, m=None, epsilon=1e-5, method="eig", dtype='float32'):
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
    if method == "svd":
        return build_kernel(x, nb_out=nb_out, svd2ker=svd2pca, m=m, epsilon=epsilon, dtype=dtype)
    else:
        return build_kernel(x, nb_out=nb_out, eig2ker=eig2pca, m=m, epsilon=epsilon, dtype=dtype)


def whitening(x, nb_out=None, w=None, m=None, func=pca, epsilon=1e-5, reshape='first'):
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
    if nb_out is None:
        if w is not None:
            nb_out = w.shape[1]
        else:
            nb_out = x.shape[:1]
    if w is None:
        if 'first' in reshape:
            shape = (x.shape[0], -1)
        elif 'last' in reshape:
            shape = (-1, x.shape[-1])
        else:
            shape = None
        w = func(x, nb_out, shape, m, epsilon)

    # whiten the input data
    shape = np.shape(x)
    x = np.reshape(x, (-1, np.shape(w)[0]))

    if m is None:
        m = np.mean(x, axis=0) if np.shape(x)[0] > 1 else np.zeros((1, np.shape(w)[0]))

    return np.reshape(np.dot(x - m, w), shape[:-1] + (nb_out,))
