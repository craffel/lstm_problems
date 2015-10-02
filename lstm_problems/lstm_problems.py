# -*- coding: utf-8 -*-
""" Various toy problems which are meant to test whether a model can learn
long-term dependencies.  Originally proposed in [1]_, with variants used more
recently in [2]_, [3]_, [4]_, etc.  In general, we follow the descriptions in
[1]_ because they are the most comprehensive; variants of the original tasks
have been used instead in the cited papers.

.. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory." Neural
computation 9.8 (1997): 1735-1780.

.. [2] Ilya Sutskever et al. "On the importance of initialization and momentum
in deep learning." Proceedings of the 30th international conference on machine
learning (ICML-13). 2013.

.. [3] Herbert Jaegar. "Long Short-Term Memory in Echo State Networks: Details
of a Simulation Study." Jacobs University Technical Report No. 27, 2012.

.. [4] James Martens and Ilya Sutskever. "Learning recurrent neural networks
with hessian-free optimization." Proceedings of the 28th International
Conference on Machine Learning (ICML-11). 2011.

.. [5] Quoc V. Le, Navdeep Jaitly, and Geoffrey E. Hinton. "A Simple Way to
Initialize Recurrent Networks of Rectified Linear Units." arXiv preprint
arXiv:1504.00941 (2015).
"""
import numpy as np
import functools


def gen_masked_sequences(min_length, n_sequences, sample, max_length=None):
    """ Generates two-dimensional sequences and masks of length randomly chosen
    between ``[min_length, max_length]`` where one dimension is randomly
    sampled using ``sample`` and the other dimension has -1 at the start and
    end of the sequence and has a 1 in the first ten sequence steps and another
    1 before ``min_length/2`` and is 0 otherwise.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    sample : function
        Function used to randomly sample values; should accept a keyword
        argument "size" which determines the sampled shape.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.
    """
    if max_length is None:
        # Compute the global maximum sequence length
        max_length = int(np.ceil(1.1*min_length))
    # When min_length and max_length is the same, make all lengths the same
    if min_length == max_length:
        lengths = min_length*np.ones(n_sequences, dtype=int)
    # Otherwise, randomly choose lengths from [min_length, max_length]
    else:
        lengths = np.random.randint(min_length, max_length, n_sequences)
    # Construct a mask by tiling arange and comparing it to lengths
    mask = (np.tile(np.arange(max_length), (n_sequences, 1)) <
            np.tile(lengths, (max_length, 1)).T)
    # Sample the noise dimension
    noise_dim = sample(size=(n_sequences, max_length, 1))
    # Mask out all entries past the end of each sequence
    noise_dim *= mask[:, :, np.newaxis]
    # Initialize mask dimension to all zeros
    mask_dim = np.zeros((n_sequences, max_length, 1))
    # First entry of each sequence in mask dimension is -1
    mask_dim[:, 0] = -1
    # End of sequence in mask dimension is -1
    mask_dim[np.arange(n_sequences), lengths - 1] = -1
    # First add index is always between 1 and 10
    N_1 = np.random.randint(1, 10, n_sequences)
    # Second add index is always between 1 and length/2
    # Using max_length instead of length[n] is a WLOG hack
    N_2 = np.random.choice(range(1, int(max_length/2 - 1)), n_sequences)
    # If N_1 = N_2 for any sequences, add 1 to avoid
    N_2[N_2 == N_1] = N_2[N_2 == N_1] + 1
    # Set the add indices to 1
    mask_dim[np.arange(n_sequences), N_1] = 1
    mask_dim[np.arange(n_sequences), N_2] = 1
    # Concatenate noise and mask dimensions to create data
    X = np.concatenate([noise_dim, mask_dim], axis=-1)
    return X, mask


def add(min_length, n_sequences, max_length=None):
    """ Generate sequences and target values for the "add" task, as described in
    [1]_ section 5.4.  Sequences are two dimensional where the first dimension
    are values sampled uniformly at random from [-1, 1] and the second
    dimension is either -1, 0, or 1: At the first and last steps, it is
    -1; at one of the first ten steps (``N_1``) it is 1; and at a step between
    0 and ``.5*min_length`` (``N_2``) it is
    also 1.  The goal is to predict ``0.5 + (X_1 + X_2)/4.0`` where ``X_1`` and
    ``X_2`` are the values of the first dimension at ``N_1`` and ``N_2``
    respectively.  For example, the target for the following sequence

    ``| 0.5 | -0.7 | 0.3 | 0.1 | -0.2 | ... | -0.5 | 0.9 | ... | 0.8 | 0.2 |
      | -1  |   0  |  1  |  0  |   0  |     |   0  |  1  |     |  0  | -1  |``

    would be ``.5 + (.3 + .9)/4 = .8``.  All generated sequences will be of
    length ``max_length``; the returned variable ``mask``
    can be used to determine which entries are in each sequence.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

    References
    ----------
    .. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    """
    # Get sequences
    X, mask = gen_masked_sequences(
        min_length, n_sequences,
        functools.partial(np.random.uniform, high=1., low=-1.), max_length)
    # Sum the entries in the third dimension where the second is 1
    y = np.sum((X[:, :, 0]*(X[:, :, 1] == 1)), axis=1)
    # Normalize targets to the range [0, 1]
    y = .5 + y/4.
    return X, y, mask


def multiply(min_length, n_sequences, max_length=None):
    """ Generate sequences and target values for the "multiply" task, as
    described in [1]_ section 5.5.  Sequences are two dimensional where the
    first dimension are values sampled uniformly at random from [0, 1] and the
    second dimension is either -1, 0, or 1: At the first and last steps, it is
    -1; at one of the first ten steps (``N_1``) it is 1; and at a step between
    0 and ``.5*min_length`` (``N_2``) it is also 1.  The goal is to predict
    ``X_1*X_2`` where ``X_1`` and ``X_2`` are the values of the first dimension
    at ``N_1`` and ``N_2`` respectively.  For example, the target for the
    following sequence

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      | -1  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  | -1  |``

    would be ``.3*.9 = .27``.  All generated sequences will be of
    length ``max_length``; the returned variable ``mask``
    can be used to determine which entries are in each sequence.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

    References
    ----------
    .. [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.
    """
    # Get sequences
    X, mask = gen_masked_sequences(
        min_length, n_sequences,
        functools.partial(np.random.uniform, high=1., low=0.), max_length)
    # Multiply the entries in the third dimension where the second is 1
    y = np.prod((X[:, :, 0]**(X[:, :, 1] == 1)), axis=1)
    return X, y, mask


def xor(min_length, n_sequences, max_length=None):
    """ Generate sequences and target values for the "XOR" task, as
    described in [1]_ section 4.1.  Sequences are two dimensional where the
    first dimension are binary values sampled uniformly at random from {0, 1}
    and the second dimension is either -1, 0, or 1: At the first and last
    steps, it is -1; at one of the first ten steps (``N_1``) it is 1; and at a
    step between 0 and ``.5*min_length`` (``N_2``) it is also 1.  The goal is
    to predict ``X_1^X_2`` where ``X_1`` and ``X_2`` are the values of the
    first dimension at ``N_1`` and ``N_2`` respectively.  For example, the
    target for the following sequence

    ``|  1 | 0 | 1 | 0 | 0 | ... | 1 | 1 | ... | 0 |  0 |
      | -1 | 0 | 1 | 0 | 0 |     | 0 | 1 |     | 0 | -1 |``

    would be ``1^1 = 0``.  All generated sequences will be of
    length ``max_length``; the returned variable ``mask``
    can be used to determine which entries are in each sequence.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    n_sequences : int
        Number of sequences to generate.
    max_length : int or None
        Maximum sequence length.  If supplied as `None`,
        ``int(np.ceil(1.1*min_length))`` will be used.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape
        ``(n_sequences, 1.1*min_length, 2)``, where the last
        dimension corresponds to the two sequences described above.
    y : np.ndarray
        Correct output for each sample, shape ``(n_sequences,)``.
    mask : np.ndarray
        A binary matrix of shape ``(n_sequences, 1.1*min_length)``
        where ``mask[i, j] = 1`` when ``j <= (length of sequence i)``
        and ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

    References
    ----------
    .. [1] James Martens and Ilya Sutskever. "Learning recurrent neural
    networks with hessian-free optimization." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.
    """
    # Get sequences
    X, mask = gen_masked_sequences(
        min_length, n_sequences,
        functools.partial(np.random.choice, a=[0, 1]), max_length)
    # X[:, :, 1] > 0 constructs a boolean matrix of the rows/columns which have
    # a 1 in the last dimension of X.  X[X[:, :, 1] > 0, 0] then masks the
    # entries of the random bit dimension accordingly.  The reshape converts
    # the resulting array back into a matrix, where entries are picked in
    # "fortran" order which allows them to be correctly reshaped to (2,
    # n_sequences).  Finally, the * uses the first dimension as the arguments
    # to logical_xor
    y = np.logical_xor(
        *np.reshape(X[X[:, :, 1] > 0, 0], (2, n_sequences), 'F'))
    return X, y, mask
