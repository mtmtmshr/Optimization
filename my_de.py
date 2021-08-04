import numpy as np
import math
import statistics
from argparse import ArgumentParser
from tqdm import tqdm
import random
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


def rand_ints_nodup(start, last, num):
    ns = []
    while len(ns) < num:
        n = random.randint(start, last)
        if not n in ns:
            ns.append(n)
    return ns


def my_de(objective_function, D=2):
    M = 50
    Xmin = -5
    Xmax = 5

    Tmax = 1000
    F_end = 10 ** -5
    X = np.random.uniform(low=Xmin, high=Xmax, size=(M, D))
    X_new = np.zeros((M, D))
    V = np.zeros((D))
    U = np.zeros((D))
    F = np.full((M), np.inf)
    F_best = np.inf

    Cr = 0.9
    Fw = 0.5

    for t in tqdm(range(Tmax)):
        for i in range(M):
            selected_M = rand_ints_nodup(0, M-1, 3)
            V = X[selected_M[0]] + Fw * (X[selected_M[1]] - X[selected_M[2]])
            jr = random.randint(0, D-1)
            for j in range(D):
                ri = random.random()
                if ri < Cr or j == jr:
                    U[j] = V[j]
                else:
                    U[j] = X[i][j]
            F_tmp = objective_function(U)

            if F_tmp < F[i]:
                F[i] = F_tmp
                X_new[i] = copy.deepcopy(U)
                if F_best > F_tmp:
                    F_best = F_tmp
                    X_best = copy.deepcopy(U)
            else:
                X_new[i] = copy.deepcopy(X[i])
        X = copy.deepcopy(X_new)

        if F_best < F_end:
            break
    times.append(t+1)
    Fgs.append(F_best)



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
        my_de(objective_function, D)

    print(statistics.mean(Fgs))
    print(statistics.variance(Fgs))
    print(statistics.mean(times))
    print(statistics.variance(times))
