import math

import numpy as np
from scipy.special import erf

def vector_dist(x1, x2, norm_type='euclid'):
    if norm_type.lower() == 'euclid':
        ord = 2  # np.sqrt(np.sum(np.square(vector1-vector2)))
    elif norm_type.lower() == 'manhattan':
        ord = 1
    elif norm_type.lower() == 'chebyshev' or norm_type.lower() == 'cheb':
        ord = np.Inf
    else:
        raise Exception(f'Incorrect value "{norm_type}" for parameter "type"!')
    return np.linalg.norm(x2 - x1, ord=ord)


def generate_solution(size):
    return np.random.randint(0, 2, size).astype('float')


def get_transfer_function(transfer_fun='s1'):
    if transfer_fun.lower() == 's1':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-2.0 * x))
    elif transfer_fun.lower() == 's2':
        transfer = lambda x: 1.0 / (1.0 + np.e ** -x)
    elif transfer_fun.lower() == 's3':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 2.0))
    elif transfer_fun.lower() == 's4':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 3.0))
    elif transfer_fun.lower() == 's5':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-3.0 * x))
    elif transfer_fun.lower() == 'stan':
        transfer = lambda x: np.abs(2 / np.pi * np.arctan(x * np.pi / 2))
    elif transfer_fun.lower() == 'erf':
        transfer = lambda x: np.abs(erf(0.8862269254527579 * x)) #np.sqrt(np.pi/2)
    elif transfer_fun.lower() == 'tanh':
        transfer = lambda x: np.abs(np.tanh(x))
    elif transfer_fun.lower() == 'v3':
        transfer = lambda x: np.abs(x / (np.sqrt(1 + x*x)))
    else:
        raise Exception(f'Incorrect value "{transfer_fun}" for parameter "transfer_fun"!')
    return transfer


def standard_discrete(transfer_fun, x):
    x = transfer_fun(x)
    r = np.random.sample(x.shape)
    return np.where(x >= r, 1.0, 0.0)


def binarization(transfer, discretization):
    def decorator(x):
        if x.ndim == 1:
            return discretization(transfer, x)
        else:
            return np.array([discretization(transfer, e) for e in x], dtype=float)

    return decorator
