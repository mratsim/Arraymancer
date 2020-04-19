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

def masked_fill():
    print('Masked fill')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[y > 50] = -100')
    y[y > 50] = -100
    print(y)
    print('--------------------------')
    y = x.copy()
    print('y[y < 50] = -100')
    y[y < 50] = -100
    print(y)
    print('--------------------------')

def masked_axis_fill_value():
    print('Masked axis fill with value')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[:, y.sum(axis = 0) > 50] = -100')
    y[:, y.sum(axis = 0) > 50] = -100
    print(y)
    print('--------------------------')
    y = x.copy()
    print('y[y.sum(axis = 1) > 50, :] = -100')
    y[y.sum(axis = 1) > 50, :] = -100
    print(y)
    print('--------------------------')

def masked_axis_fill_tensor_invalid_1():
    # ValueError: shape mismatch:
    # value array of shape (4,) could not be broadcast
    # to indexing result of shape (2,4)
    print('Masked axis fill with tensor - invalid numpy syntax')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[:, y.sum(axis = 0) > 50] = np.array([10, 20, 30, 40])')
    y[:, y.sum(axis = 0) > 50] = np.array([10, 20, 30, 40])
    print(y)

def masked_axis_fill_tensor_valid_1():
    print('Masked axis fill with tensor - 1d tensor broadcasting')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[:, y.sum(axis = 0) > 50] = np.array([[10], [20], [30], [40]])')
    y[:, y.sum(axis = 0) > 50] = np.array([[10], [20], [30], [40]])
    print(y)
    print('--------------------------')
    y = x.copy()
    print('y[y.sum(axis = 1) > 50, :] = np.array([-10, -20, -30])')
    y[y.sum(axis = 1) > 50, :] = np.array([-10, -20, -30])
    print(y)
    print('--------------------------')

def masked_axis_fill_tensor_valid_2():
    print('Masked axis fill with tensor - multidimensional tensor')
    print('--------------------------')
    x = np.array([[ 4, 99,  2],
                [ 3,  4, 99],
                [ 1,  8,  7],
                [ 8,  6,  8]])

    print(x)
    print('--------------------------')
    y = x.copy()
    print('y[:, y.sum(axis = 0) > 50] = np.array([[10, 50], [20, 60], [30, 70], [40, 80]])')
    y[:, y.sum(axis = 0) > 50] = np.array([[10, 50],
                                           [20, 60],
                                           [30, 70],
                                           [40, 80]])
    print(y)
    print('--------------------------')
    y = x.copy()
    print('y[y.sum(axis = 1) > 50, :] = np.array([-10, -20, -30], [-40, -50, -60])')
    y[y.sum(axis = 1) > 50, :] = np.array([[-10, -20, -30],
                                           [-40, -50, -60]])
    print(y)
    print('--------------------------')

# index_fill()
# masked_fill()
# masked_axis_fill_value()
masked_axis_fill_tensor_invalid_1()
# masked_axis_fill_tensor_valid_1()
# masked_axis_fill_tensor_valid_2()
