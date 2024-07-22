#!/usr/bin/env python3

import argparse
import subprocess
import sys

import adam

def optiongrad(option, args):
    cutechess_args = ['cutechess-cli', '-concurrency', args.cpus,
                      '-each', 'tc=' + args.tc, 'proto=uci',
                      '-rounds', args.cpus,
                      '-games', '2',
                      '-openings', 'format=epd', 'file=' + args.book, 'order=random',
                      '-repeat',
                      '-engine', 'cmd=' + args.engine, 'name=bitbit+', 'option.' + option.name + '=' + str(option.value + option.delta),
                      '-engine', 'cmd=' + args.engine, 'name=bitbit-', 'option.' + option.name + '=' + str(option.value - option.delta),
                      '-draw', 'movenumber=60', 'movecount=8', 'score=20',
                      '-resign', 'movecount=3', 'score=800', 'twosided=true']
    
    cutechess = subprocess.run(cutechess_args, capture_output=True, text=True)
    
    elo = 0.0
    for line in cutechess.stdout.splitlines():
        if not line.startswith('Elo difference: '):
            continue
        line = line.split()
        elo = float(line[2])
    for line in cutechess.stderr.splitlines():
        print('error: ' + line)
    if cutechess.returncode != 0:
        sys.exit(1)

    score = 1.0 / (1.0 + 10.0 ** (-elo / 400.0))

    # There is no proper cost function and this is not actually a gradient.
    # However, it should still work resonably well.
    grad = (0.5 - score) / (2.0 * option.delta)

    return grad

def tuneoptions(options, args, t):
    for option in options:
        adam.mv(option, optiongrad(option, args), args)

    for option in options:
        adam.step(option, args, t)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('engine', type=str, help='Engine path')
    parser.add_argument('config', type=str, help='Configuration file')
    parser.add_argument('book', type=str, help='Opening book')
    parser.add_argument('cpus', type=str, help='Cpus')
    parser.add_argument('--tc', type=str, help='Time control', default='1+0.04')
    parser.add_argument('--beta1', type=float, help='beta1 hyperparameter', default=0.9)
    parser.add_argument('--beta2', type=float, help='beta2 hyperparameter', default=0.999)
    parser.add_argument('--epsilon', type=float, help='epsilon hyperparameter', default=1e-8)
    parser.add_argument('--alpha', type=float, help='alpha hyperparameter', default=1e-3)

    args = parser.parse_args()

    options = []

    with open(args.config) as config:
        for option in config:
            if option[0] == '#':
                continue
            option = option.split()
            # <option> <value> <delta> <learning rate> <m> <v>
            options.append(adam.Option(name=option[0], initial=float(option[1]), value=float(option[1]), delta=float(option[2]), lr=float(option[3]), m=0.0, v=0.0))

    t = 1
    while True:
        tuneoptions(options, args, t)
        for option in options:
            print(option.name + ': ' + str(option.value) + ' (' + str(option.initial) + ')')
        t += 1

    print(options)

if __name__ == '__main__':
    main()
