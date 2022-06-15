import numpy as np
import supscp
from numba import njit, float32
from scpmath import *

def black_hole_algorithm(table, costs, pop_size=40, max_iter=500, event_horizon='standard',
                         notation='CS', transfer_fun='S1', discrete_fun='standard', progress=None):
    transfer = get_transfer_function(transfer_fun)

    if discrete_fun.lower() == 'standard':
        discrete = standard_discrete
    else:
        discrete = standard_discrete

    binary_fun = binarization(transfer, discrete)
    repair_star = supscp.repair_solution(table, costs, notation)
    optimum, values = engine_bha(costs, pop_size, max_iter, binary_fun, repair_star, event_horizon, progress)
    return optimum, values




def engine_bha(costs, pop_size, max_iter, binarization, repair_star, event_horizon, progress):
    stars = generate_solution((pop_size, len(costs)))
    _ = list(map(repair_star, stars))
    black_hole = np.zeros(len(costs))
    bh_fitness = np.Inf
    if progress: progress.start()

    for step in range(max_iter):
        stars_fitness = supscp.calc_fitness(stars, costs)
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



def standard_dist(f1, f_bh):
    return abs(f_bh - f1)
