import numpy as np



def check_solution(table, solution):
    strs = {elem[0] for elem in table}
    columns = {elem[1] for elem in table}
    alpha = {f: s for f, s in zip(
        strs, [{elem[1] for elem in table if elem[0] == t} for t in strs]
    )}
    betta = {f: s for f, s in zip(
        columns, [{elem[0] for elem in table if elem[1] == t} for t in columns]
    )}

    sol = [i+1 for i, e in enumerate(solution) if e > 0]
    res_test = set()
    for e in sol:
        res_test |= betta[e]
    return len(res_test) == len(strs), len(strs), len(res_test)

    
