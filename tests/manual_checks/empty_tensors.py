import numpy as np

def empty_create1():
    empty = np.ndarray([])

    print(empty)            # Why 0.007812501848093234 ?
    print(empty.shape)
    print(empty.strides)

    empty += 1              # 1.007812501848093234
    empty = empty * 10000

    print(empty)            # 10078.125018480932
    # print(empty[0])       # Crash

def empty_create2():
    a = np.ndarray([1, 1])
    empty = a[a < 0]

    print(empty)            # []
    print(empty.shape)
    print(empty.strides)

    empty += 1
    empty = empty * 10000

    print(empty)            # []

    # a += empty              # Crash non broadcastable
    # print(a)
    # a *= empty              # Crash non broadcastable
    # print(a)

def empty_default():
    a = np.ndarray([1, 1])
    empty = a[a < 0]

    print(empty.sum())  # 0.0
    print(empty.prod()) # 1.0
    print(empty.var())  # warning invalid then "NaN"

def indexing():
    r = np.ndarray([1, 1])
    print(r[r < 0])     # []


if __name__ == "__main__":
    empty_create1()
    print("-------------------------------")
    empty_create2()
    print("-------------------------------")
    empty_default()
    print("-------------------------------")
    indexing()
