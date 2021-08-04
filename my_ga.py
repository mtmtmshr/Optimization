import numpy as np
import math
import statistics
from argparse import ArgumentParser
from tqdm import tqdm
import copy
import random
import math
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


def init_x(D, w, wMax) -> np.array:
    while True:
        x = np.random.randint(2, size=D)
        if np.sum(np.array(w) * x) <= wMax:
            break
    return x


def select_parent(F, M, selected=-1):
    r = random.random()
    sum_f = np.sum(F)
    if selected >= 0:
        sum_f -= F[selected]
    p = len(F)-1
    for i in range(M):
        if i == selected:
            continue
        prob = F[i] / (sum_f+10**-8)
        if (r < prob):
            p = i
            break
        r -= prob
    return p


def my_ga(D=2):
    M = 20
    Pm = 0.05
    Tmax = 100
    F_best = 0
    X_best = np.zeros((D))
    X = np.zeros((M, D))
    F = np.zeros((M))
    X_next = np.zeros((M, D))
    if D == 5:
        w = [7, 5, 1, 9, 6]
        v = [50, 40, 10, 70, 55]
        wMax = 15
    elif D == 10:
        w = [3, 6, 5, 4, 8, 5, 3, 4, 8, 2]
        v = [70, 120, 90, 70, 130, 80, 40, 50, 30, 70]
        wMax = 20
    else:
        print(f"D={D} is not defined")
        sys.exit()

    for i in range(M):
        X[i] = init_x(D, w, wMax)

    for t in range(Tmax):
        for i in range(M):
            F[i] = np.sum(np.array(v) * X[i])
            if F[i] > F_best:
                F_best = F[i]
                X_best = copy.deepcopy(X[i])
        break
        for i in range(M):
            while True:
                p1 = select_parent(F, M)
                p2 = select_parent(F, M, p1)
                d1 = random.randint(0, D)
                while True:
                    d2 = random.randint(0, D)
                    if d1 != d2:
                        break

                if d1 > d2:
                    tmp = d1
                    d1 = d2
                    d2 = tmp
                for d in range(D):
                    if d <= d1 or d > d2:
                        X_next[i][d] = X[p1][d]
                    else:
                        X_next[i][d] = X[p2][d]
                for d in range(D):
                    if random.random() < Pm:
                        X_next[i][d] = 1 - X_next[i][d]
                if np.sum(np.array(w) * X_next) <= wMax:
                    break
        X = copy.deepcopy(X_next)

    print(f"(“解の⽬的関数値 Fg = {F_best}")
    print(X_best)

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
        my_ga(D)

    print(statistics.mean(Fgs))
    print(statistics.variance(Fgs))
    print(statistics.mean(times))
    print(statistics.variance(times))
