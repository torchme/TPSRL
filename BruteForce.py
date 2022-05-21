import pandas as pd
import numpy as np
import math
import itertools
import datetime

PATH_DF = 'data/'

def read_csv(path_way='df.csv'):
    df = pd.read_csv(PATH_DF + path_way, index_col='Unnamed: 0')
    df = df.iloc[:16, :16]
    N = df.shape[0]
    M = df.values

    return N, M, df

if __name__ == '__main__':
    N, M, df = read_csv('dist_vologda_matrix.csv')

    print(math.factorial(N-1))

    print()
    x_min = []
    g_min = 100000000
    print(datetime.datetime.now())
    for i, x in enumerate(itertools.permutations(range(1, N), N - 1)):
        x = (0,) + x
        g = np.sum(M[x[:-1], x[1:]]) + M[x[-1], x[0]]
        if g < g_min:
            g_min = g
            x_min = x
        if i % 10000000 == 0:
            print(i, x_min, g_min)
    print(datetime.datetime.now())
    print('Optimal solution:')
    print(x_min, g_min)