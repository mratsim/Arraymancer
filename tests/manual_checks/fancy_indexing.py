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

def masked_select():
    print('Masked select')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    print('x[x > 50]')
    print(x[x > 50])
    print('--------------------------')
    print('x[x < 50]')
    print(x[x < 50])

def masked_axis_select():
    print('Masked axis select')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    print('x[:, np.sum(x, axis = 0) > 50]')
    print(x[:, np.sum(x, axis = 0) > 50])
    print('--------------------------')
    print('x[np.sum(x, axis = 1) > 50, :]')
    print(x[np.sum(x, axis = 1) > 50, :])

# index_select()
# masked_select()
# masked_axis_select()

print('\n#########################################\n')
print('Fancy mutation')

def index_fill():
    print('Index fill')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[:, [0, 2]] = -100')
    y[:, [0, 2]] = -100
    print(y)
    print('--------------------------')
    y = x.copy()
    print('y[[1, 3], :] = -100')
    y[[1, 3], :] = -100
    print(y)
    print('--------------------------')

index_fill()
