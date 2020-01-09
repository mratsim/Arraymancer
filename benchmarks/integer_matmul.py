import numpy as np

def main(n):
    a = np.random.randint(100, size=(n,n)) # Default type is np.int which should also be int64
    b = a
    c = np.dot(a, b)
    print(c[n // 2][n // 2])


if __name__=='__main__':
    import sys

    if len(sys.argv) > 1:
        main(int(sys.argv[1]))
    else:
        main(100)

#########
# Results on i9-9980XE
# Skylake-X overclocked 4.1GHz all-core turbo,
# AVX2 4.0GHz all-coreAVX-512 3.5GHz all-core
# Input 1500x1500 random large int64 matrix

# Nim 1.0.4. Compilation option: "-d:danger -d:openmp"
# Julia v1.3.1
# Python 3.8.1 + Numpy-MKL 1.18.0

# Nim: 0.14s, 22.7Mb
# Julia: 1.67s, 246.5Mb
# Python Numpy: 5.69s, 75.9Mb
