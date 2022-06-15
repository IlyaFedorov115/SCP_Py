import parse_scp
import progressbar
import time
import sys
import argparse
import ffa




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Black Hole algorithm for SCP problem testing')
    parser.add_argument('--pop_size', '-ps', help='Population size')
    parser.add_argument('--num_iter', '-ni', help='Num of iteration')
    parser.add_argument('--file', '-f', help='Filename of OR-Library')
    parser.add_argument('--transfer', '-t', help='Transfer function (s1|s2|s3|s4|stan)')
    parser.add_argument('--discrete', help='Discretization function (standard|standard)')
    parser.add_argument('--dist', help='Distance between stars (vector|standard)')
    parser.add_argument('--progress', '-p', help='Need progressbar (y|yes|n|no)')
    parser.add_argument('--gamma', '-g', help='Gamma for light, [0.01;100]')
    parser.add_argument('--betta', help='Betta_0 param')
    parser.add_argument('--betta_pow', help='Pow for calc betta=betta_0 * e ** (-gamma * r ** pow)')
    parser.add_argument('--alpha', help='Param alpha [0, 1]')
    parser.add_argument('--alpha_0', help='Param alpha_0')
    parser.add_argument('--alpha_oo', help='Param alpha_oo')
    parser.add_argument('--attractive', help='simple_attractive yes|no')
    parser.add_argument('--gamma_alter', help='Alter view on gamma')
    parser.add_argument('--move_type', help='Choose variant of moving function')

    args = parser.parse_args()
    if not args.pop_size or not args.file or not args.num_iter:
        raise Exception('Use help to read about parameters.')

    filename = args.file.strip("'\"")
    pop_size = int(args.pop_size)
    max_iter = int(args.num_iter)

    transfer = args.transfer or 's1'
    discrete = args.discrete or 'standard'
    distance = args.dist or 'euclid'
    if args.gamma:
        gamma = float(args.gamma)
    else:
        gamma = 1.0

    betta_0 = float(args.betta) if args.betta else 1.0
    betta_pow = float(args.betta_pow) if args.betta_pow else 2
    alpha = float(args.alpha) if args.alpha  else 0.5
    alpha_inf = float(args.alpha_oo) if args.alpha_oo else None
    alpha_0 = float(args.alpha_0) if args.alpha_0 else None
    gamma_alter = float(args.gamma_alter) if args.gamma_alter else 0
    move_type = args.move_type or None

    attractive = False
    if args.attractive:
        if args.attractive.lower() == 'yes' or args.attractive.lower() == 'y':
            attractive = True

    widgets = [' [', progressbar.AnimatedMarker(), progressbar.Timer(format='elapsed time: %(elapsed)s'),
               '] ', progressbar.Bar('█'), ' (', progressbar.ETA(), ') ',]
    bar = progressbar.ProgressBar(max_value=max_iter, widgets=widgets) \
        if args.progress == 'yes' or args.progress == 'y' else None

    print(f'-----------Start testing algorithm on {filename}-------------')
    table, costs, cost_list = parse_scp.parse_scp_file(filename)

    optimum, values = ffa.ffa_algorithm(table, cost_list, pop_size=pop_size, max_iter=max_iter, gamma=gamma, betta_0=betta_0, notation='CS',
                                                    transfer_fun=transfer, progress=bar, distance=distance, betta_pow=betta_pow,
                                                    alpha=alpha, alpha_inf=alpha_inf, alpha_0=alpha_0, simple_attractive=attractive,
                                                    gamma_alter=gamma_alter, move_type=move_type)



    #print(f'\nOptimal solution {optimum}')
    print(f'\nOptimum values {values}')
    print(f'\nSize of solution {len([e for e in optimum if e])}')
