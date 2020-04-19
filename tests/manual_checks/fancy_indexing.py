import numpy as np

def index_select():
    print('Index select')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    print('x[:, [0, 2]]')
    print(x[:, [0, 2]])
    print('--------------------------')
    print('x[[1, 3], :]')
    print(x[[1, 3], :])
