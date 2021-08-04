import numpy as np
import math
import statistics
from argparse import ArgumentParser
from tqdm import tqdm
import copy
times = []
Fgs = []


def Stretched_V_function(x: np.ndarray):
    fx = 0
    for i in range(x.size-1):
        squares = x[i] ** 2 + x[i+1] ** 2
        fx += ((squares) ** 0.25) * (1 + math.sin(50 * (squares ** 0.1)) ** 2)
    return fx


def De_Jong_function(x: np.ndarray):
    fx = 0
    for i in range(x.size):
        fx += abs(x[i])
    return fx


def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-D', '--dim', type=int,
                           default=2,
                           help='Specify size of dimention')
    argparser.add_argument('-F', '--function', type=int,
                           default=0,
                           help='Specify the objective_function')
    return argparser.parse_args()


def my_pso(objective_function, D=2):
    M = 30
    Xmin = -10
    Xmax = 10
    c = 1.494
    w = 0.729
    Tmax = 1000
    Cr = 10 ** -5
    X = np.random.uniform(low=Xmin, high=Xmax, size=(M, D))
    V = np.zeros((M, D))
    Fp = np.full((M), np.inf)
    Xp = np.zeros((M, D))
    Fg = np.inf
    Xg = np.zeros((D))
    
    for t in tqdm(range(Tmax)):
        for i in range(M):
            fi = objective_function(X[i])
            if fi < Fp[i]:
                Fp[i] = fi
                Xp[i] = X[i])
                if Fp[i] < Fg:
                    Fg = Fp[i]
                    Xg = Xp[i]
        if Fg < Cr:
            break
        r1 = np.random.uniform(size=(M, D))
        r2 = np.random.uniform(size=(M, D))
        V = w * V + c * r1 * (Xp - X) + c * r2 * (Xg - X)
        X = X + V
    times.append(t+1)
    Fgs.append(Fg)


if __name__ == "__main__":
    """
        実行方法
        python3 my_pso.py -D 2 -F 0

        -D 次元数
        指定しなければ 2

        -F 使用する関数 0 or 1
        0: De_Jong_function
        1: Stretched_V_function
        指定しなければ De_Jong_function
    """

    args = get_option()
    D = args.dim

    if args.function:
        objective_function = Stretched_V_function
    else:
        objective_function = De_Jong_function

    for _ in tqdm(range(100)):
        my_pso(objective_function, D)

    print(statistics.mean(Fgs))
    print(statistics.variance(Fgs))
    print(statistics.mean(times))
    print(statistics.variance(times))
