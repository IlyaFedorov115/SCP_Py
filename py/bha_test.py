import bha
import parse_scp
import progressbar
import time
import sys
import argparse
import check_solution

#python main_test.py --pop_size 60 --num_iter 300 --file ./OR/scp41.txt --transfer s1 --dist euclid --progress yes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Black Hole algorithm for SCP problem testing')
    parser.add_argument('--pop_size', '-ps', help='Population size')
    parser.add_argument('--num_iter', '-ni', help='Num of iteration')
    parser.add_argument('--file', '-f', help='Filename of OR-Library')
    parser.add_argument('--transfer', '-t', help='Transfer function (s1|s2|s3|s4)')
    parser.add_argument('--discrete', help='Discretization function (standard|standard)')
    parser.add_argument('--dist', help='Distance between stars (vector|standard)')
    parser.add_argument('--progress', '-p', help='Need progressbar (y|yes|n|no)')

    args = parser.parse_args()
    if not args.pop_size or not args.file or not args.num_iter:
        raise Exception('Use help to read about parameters.')

    filename = args.file.strip("'\"")
    pop_size = int(args.pop_size)
    max_iter = int(args.num_iter)

    transfer = args.transfer or 's1'
    discrete = args.discrete or 'standard'
    distance = args.dist or 'euclid'

    widgets = [' [', progressbar.AnimatedMarker(), progressbar.Timer(format='elapsed time: %(elapsed)s'),
               '] ', progressbar.Bar('█'), ' (', progressbar.ETA(), ') ',]
    bar = progressbar.ProgressBar(max_value=max_iter, widgets=widgets) \
        if args.progress == 'yes' or args.progress == 'y' else None

    print(f'-----------Start testing algorithm on {filename}-------------')
    table, costs, cost_list = parse_scp.parse_scp_file(filename)
    optimum, values = bha.black_hole_algorithm(table, cost_list, pop_size=pop_size,
                                                      max_iter=max_iter, event_horizon=distance,notation='CS',
                                                      transfer_fun=transfer, discrete_fun=discrete, progress=bar)

    print(f'\nOptimum values {values}')
    print(f'\nSize of solution{len([e for e in optimum if e])}')
    print(f'\nCheck: {check_solution.check_solution(table, optimum)}')
