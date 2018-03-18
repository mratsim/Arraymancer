import sys
import time
import numpy as np


dz = 0.1
z = 1000
space_steps = int(z / dz)
time_steps = 50_000
total_time = 1_000_000
dt = total_time / time_steps
alpha = 2e-4
starting_temp = 30
oscillations = 20
a_dz2 = alpha / dz**2


def f(T):
    return a_dz2 * (T[:-2] - 2 * T[1:-1] + T[2:])


def euler_solve(Ts):
    for t in range(time_steps-1):
        Ts[t+1, 1:-1] = Ts[t, 1:-1] + dt * f(Ts[t])
        Ts[t+1, -1] = Ts[t+1, -2]


start = time.time()

start_iterspeed = time.time()
Ts = starting_temp * np.ones((time_steps, space_steps), dtype=np.float64)
stop_iterspeed = time.time()

ts = np.linspace(0, time_steps, time_steps)
Ts[:, 0] = starting_temp - oscillations * np.sin(2 * np.pi * ts)

euler_solve(Ts)
print(Ts[45_000, 10])
print(Ts[45_000, 100])
print(Ts[45_000, 500])
stop = time.time()

print(sys.version)
np.__config__.show()
print("Numpy iteration speed - time taken: {} seconds".format(stop_iterspeed - start_iterspeed))
print("Numpy Euler solve - time taken: {} seconds".format(stop - start))

# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)

# Python 2.7 + Apple Accelerate
# Numpy iteration speed - time taken: 9.16847395897 seconds
# Numpy Euler solve - time taken: 16.3511238098 seconds
# 17.09s, 3921.8Mb

# Python 3 with Intel MKL
# Numpy iteration speed - time taken: 2.673659086227417 seconds
# Numpy Euler solve - time taken: 6.034733057022095 seconds
# 6.49s, 3836.5Mb


# Measurement on i7-970 (Hexa core 3.2GHz)
# Python 3.6
# 42.0046266041
# 34.8379557903
# 29.8974105039
# Numpy iteration speed - time taken: 1.1823573112487793 seconds
# Numpy Euler solve - time taken: 4.745648145675659 seconds
# 4.96s, 3842.0Mb
