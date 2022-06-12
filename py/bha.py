import numpy as np
from numba import njit, float32


def standard_discrete(transfer_fun, x):
    x = transfer_fun(x)
    r = np.random.sample(x.shape)
    return np.where(x >= r, 1.0, 0.0)


def complement_discrete(transfer_fun, x):
    return np.array([abs(elem-1) if r <= transfer_fun(elem)
                     else 0
                     for r, elem in zip(np.random.sample(len(x)), x)])


def binarization(transfer, discretization):
    def decorator(x): #return np.vectorize(discretization, signature='(n),()->(n)')(x, transfer)
        #return discretization(transfer, x)
        return np.array([discretization(transfer, e) for e in x], dtype=float)

    return decorator


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


def standard_dist(f1, f_bh):
    return abs(f_bh - f1)


def get_transfer_function(transfer_fun='s1'):
    if transfer_fun.lower() == 's1':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-2.0*x))
    elif transfer_fun.lower() == 's2':
        transfer = lambda x: 1.0 / (1.0 + np.e ** -x)
    elif transfer_fun.lower() == 's11':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-3.0*x))
    elif transfer_fun.lower() == 's12':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-4.0*x))        
    elif transfer_fun.lower() == 's3':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 2.0))
    elif transfer_fun.lower() == 's4':
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-x / 3.0))
    else:
        raise Exception(f'Incorrect value "{transfer_fun}" for parameter "transfer_fun"!')
    return transfer


def black_hole_algorithm(table, costs, pop_size=40, max_iter=500, event_horizon='standard',
                         notation='CS', transfer_fun='S1', discrete_fun='standard', progress=None):
    transfer = get_transfer_function(transfer_fun)

    if discrete_fun.lower() == 'standard':
        discrete = standard_discrete
    else:
        discrete = standard_discrete

    binary_fun = binarization(transfer, discrete)
    repair_star = repair_solution(table, costs, notation)
    optimum, values = engine_bha(table, costs, pop_size, max_iter, binary_fun, repair_star, event_horizon, progress)
    return optimum, values


# нужна ли вообще
def generate_solution(size):
    return np.random.randint(0, 2, size)


def calc_fitness(stars, costs):
    dim = 0 if stars.ndim == 1 else 1
    return np.array(np.sum(np.multiply(stars, costs), dim), dtype='float32')


def calc_rotation(stars, black_hole):
    return stars + np.random.sample(stars.shape) * (np.tile(black_hole, (stars.shape[0], 1)) - stars)



def process_collapse(stars, stars_fitness, black_hole, bh_fitness, event_horizon):
    event_radius = bh_fitness / np.sum(stars_fitness)
    if event_horizon.lower() == 'standard':
        indexes = [i for i in range(len(stars))
                   if standard_dist(stars_fitness[i], bh_fitness) < event_radius]
    else:
        indexes = [i for i in range(len(stars))
                   if vector_dist(stars[i], black_hole, norm_type=event_horizon) < event_radius]

    for index in indexes:
        stars[index][:] = generate_solution(len(stars[0]))



def repair_solution(table, costs, notation):
    strs = {elem[0] for elem in table}
    columns = {elem[1] for elem in table}
    alpha = {f: s for f, s in zip(
        strs, [{elem[1] for elem in table if elem[0] == t} for t in strs]
    )}
    betta = {f: s for f, s in zip(
        columns, [{elem[0] for elem in table if elem[1] == t} for t in columns]
    )}

    def wrapped(solution):
        S = {i + 1 for i, e in enumerate(solution) if e == 1.0}
        w_num = {e1: e2 for e1, e2 in zip(
            strs, [len(S & alpha[i]) for i in strs]
        )}
        U = {e for e in w_num if w_num[e] == 0}
        while U:  # increase?
            row = U.pop()
            j = min(alpha[row],
                    key=lambda r: np.Inf if len(U & betta[r]) == 0
                    else costs[r-1] / len(U & betta[r]))
            S.add(j)
            for curr in betta[j]: w_num[curr] += 1
            U = U - betta[j]

        S = list(reversed(list(S)))
        for row in S[:]:
            for curr in betta[row]:
                if w_num[curr] < 2:
                    break
            else:
                S.remove(row)
                for c in betta[row]: w_num[c] -= 1
        S = np.array(S) - 1
        solution[:] = np.zeros(len(solution))
        solution[S] = 1.0

    return wrapped


def engine_bha(table, costs, pop_size, max_iter, binarization, repair_star, event_horizon, progress):
    stars = generate_solution((pop_size, len(costs)))
    _ = list(map(repair_star, stars))
    black_hole = np.zeros(len(costs))
    bh_fitness = np.Inf
    if progress: progress.start()

    for step in range(max_iter):
        stars_fitness = calc_fitness(stars, costs)
        min_fit_index = np.argmin(stars_fitness)

        if bh_fitness > stars_fitness[min_fit_index]:
            tmp = stars[min_fit_index].copy()
            stars[min_fit_index][:] = black_hole[:]
            black_hole[:] = tmp[:]
            bh_fitness, stars_fitness[min_fit_index] = stars_fitness[min_fit_index], bh_fitness

        stars = calc_rotation(stars, black_hole)
        stars = binarization(stars)
        process_collapse(stars, stars_fitness, black_hole, bh_fitness, event_horizon)

        _ = list(map(repair_star, stars)) 

        if progress is not None: progress.update(step + 1)
    return black_hole, bh_fitness
