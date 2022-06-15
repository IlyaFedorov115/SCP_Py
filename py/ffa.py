import numpy as np
import supscp
from numba import njit
from scpmath import *


# python test_ffa.py --pop_size 20 --num_iter 80 --file ../tests/OR/scp41.txt --transfer s1 --dist euclid --progress yes --gamma_alter 0 --attractive no
# python test_ffa.py --pop_size 20 --num_iter 80 --file ../tests/OR/scp41.txt --transfer s1 --dist euclid --progress yes --gamma_alter 2 --attractive no --alpha 0.00002

# gamma = [0,1] / euclid(0000 - 1111)

def ffa_algorithm(table, costs, pop_size=30, max_iter=150, gamma=1.0, betta_0=1.0, notation='CS', transfer_fun='s1', discrete_fun='standard',
                  progress=None, distance='euclid', betta_pow=2, alpha=0.5, alpha_inf=None, alpha_0=None,
                  simple_attractive=False, gamma_alter=0, move_type=None):

    if discrete_fun.lower() == 'standard':
        discrete = standard_discrete
    else:
        discrete = elitist_discrete

    binary_fun = binarization(get_transfer_function(transfer_fun), discrete)
    repair_fun = supscp.repair_solution(table, costs, notation)
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

    # @njit
    def lambda_move_best(x1, x2, betta, alpha=0.1):
        U = np.random.uniform(-1, 1, x1.shape)
        return x1 + betta * (x2 - x1) + alpha * U * (x1 - curr_best)

    # @njit
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

    light_intensity = supscp.calc_fitness(fireflies, costs)

    for step in range(max_iter):
        for i in range(len(fireflies)):
            for j in range(len(fireflies) - 1, 0, -1):
                if light_intensity[j] < light_intensity[i]:
                    fireflies[i] = move_fun(fireflies[i], fireflies[j],
                                            get_attractive(betta_0, gamma,
                                                           vector_dist(fireflies[i], fireflies[j], distance),
                                                           betta_pow), alpha)
                    fireflies[i] = binary_fun(fireflies[i], best=curr_best)
                    repair_fun(fireflies[i])
                    light_intensity[i] = supscp.calc_fitness(fireflies[i], costs)

        if get_alpha:
            alpha = get_alpha(step)

        best = np.argmin(light_intensity)
        if curr_best_intensity > light_intensity[best]:
            curr_best_intensity = light_intensity[best]
            curr_best = fireflies[best].copy()
        if progress is not None: progress.update(step + 1)

    return curr_best, curr_best_intensity


# @njit
def move_fireflies(x1, x2, betta, alpha=0.1):
    rand = np.random.sample(len(x1))
    return x1 + betta * (x2 - x1) + alpha * (rand - 0.5)


@njit
def calc_attractive_simple(betta, gamma, r, m=2):
    return betta / (1 + gamma * (r ** m))


@njit
def calc_attractive(betta, gamma, r, m=2):
    return betta * np.e ** (-gamma * (r ** m))
