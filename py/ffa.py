import numpy as np
from numba import njit

# python test_ffa.py --pop_size 30 --num_iter 80 --file ../tests/OR/scp41.txt --transfer s1 --dist euclid --progress yes --gamma_alter 0 --attractive no
#python test_ffa.py --pop_size 30 --num_iter 80 --file ../tests/OR/scp41.txt --transfer s1 --dist euclid --progress yes --gamma_alter 2 --attractive no --alpha 0.00002

# gamma = [0,1] / euclid(0000 - 1111)

def ffa_algorithm(table, costs, pop_size=30, max_iter=150, gamma=1.0, betta_0=1.0, notation='CS', transfer_fun='stan',
                  progress=None, distance='euclid', betta_pow=2, alpha=0.5, alpha_inf=None, alpha_0=None,
                  simple_attractive=False, gamma_alter=0, move_type=None):
    discrete = standard_discrete
    binary_fun = binarization(get_transfer_function(transfer_fun), discrete)
    repair_fun = repair_solution(table, costs, notation)
    fireflies = generate_solution((pop_size, len(costs)))
    curr_best = np.ones(len(costs))
    curr_best_intensity = np.inf

    if gamma_alter > 0:
        gamma = gamma / vector_dist(np.ones(len(costs)), np.zeros(len(costs)), distance) ** gamma_alter
        print(f'Gamma: {gamma}')

    if simple_attractive:
        get_attractive = calc_attractive_simple
    else:
        get_attractive = calc_attractive

    if alpha_0 is None or alpha_inf is None:
        get_alpha = None
    else:
        get_alpha = lambda t: alpha_inf + (alpha_0 - alpha_inf) * (np.e ** -t)

    #@njit
    def lambda_move_best(x1, x2, betta, alpha=0.1):
        U = np.random.uniform(-1, 1, x1.shape)
        return x1 + betta * (x2 - x1) + alpha * U * (x1 - curr_best)

    #@njit
    def lambda_move(x1, x2, betta, alpha=0.1):
        U = np.random.uniform(-1, 1, x1.shape)
        return x1 + betta * (x2 - x1) + alpha * U


    if move_type == None:
        move_fun = move_fireflies
    elif move_type.lower() == 'lambda_best':
        move_fun = lambda_move_best
    elif move_type.lower() == 'lambda':
        move_fun = lambda_move
    else:
        raise ValueError('Error')

    _ = list(map(repair_fun, fireflies))

    light_intensity = calc_fitness(fireflies, costs)

    count = 0
    # добавление случайного блуждания, в else скорее всего
    for step in range(max_iter):
        for i in range(len(fireflies)):
            #for j in range(0, i):
            for j in range(len(fireflies)-1, 0, -1):
                #print(light_intensity)
                if light_intensity[j] < light_intensity[i]:
                    #count += 1
                    fireflies[i] = move_fun(fireflies[i], fireflies[j],
                                                  get_attractive(betta_0, gamma,
                                                                 vector_dist(fireflies[i], fireflies[j], distance),
                                                                 betta_pow), alpha)
                    fireflies[i] = binary_fun(fireflies[i])
                    repair_fun(fireflies[i])
                    light_intensity[i] = calc_fitness(fireflies[i], costs)

        #_ = list(map(repair_fun, fireflies))
        #light_intensity = calc_fitness(fireflies, costs)

        if get_alpha:
            alpha = get_alpha(step)

        best = np.argmin(light_intensity)
        if curr_best_intensity > light_intensity[best]:
            curr_best_intensity = light_intensity[best]
            curr_best = fireflies[best].copy()
        if progress is not None: progress.update(step + 1)

    print(count)
    return curr_best, curr_best_intensity

#@njit
def move_fireflies(x1, x2, betta, alpha=0.1):
    rand = np.random.sample(len(x1))
    return x1 + betta * (x2 - x1) + alpha * (rand - 0.5)


@njit
def calc_attractive_simple(betta, gamma, r, m=2):
    return betta / (1 + gamma * (r ** m))


@njit
def calc_attractive(betta, gamma, r, m=2):
    return betta * np.e ** (-gamma * (r ** m))


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


def calc_fitness(solution, costs):
    dim = 0 if solution.ndim == 1 else 1
    return np.array(np.sum(np.multiply(solution, costs), dim), dtype='float32')


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
        transfer = lambda x: 1.0 / (1.0 + np.e ** (-3.0*x))
    elif transfer_fun.lower() == 'stan':
        transfer = lambda x: np.abs(2 / np.pi * np.arctan(x * np.pi / 2))
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
                    else costs[r - 1] / len(U & betta[r]))
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
