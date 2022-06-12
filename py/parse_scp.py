from pathlib import Path
import numpy as np


def parse_scp_file(filename: str):
    try:
        table = []
        costs = dict()
        cost_list = []
        with open(filename, 'r') as open_file:
            num_str, num_columns = map(int, open_file.readline().split())
            for line in open_file:
                cost_list += map(int, line.split())
                if len(cost_list) == num_columns:
                    break
            costs = {x + 1: y for x, y in zip(range(num_columns), cost_list)}

            str_index = 0
            for line in open_file:
                count_columns = int(line)
                curr_pairs = []
                for line_curr in open_file:
                    curr_pairs += [[str_index, y] for y in map(int, line_curr.split())]
                    if len(curr_pairs) == count_columns:
                        break
                table += curr_pairs
                str_index += 1
    except FileNotFoundError as err:
        print(f'Incorrect filename {filename}', err)
        raise

    return [table, costs, cost_list]



