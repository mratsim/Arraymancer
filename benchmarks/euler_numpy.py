import numpy as np, time, sys


dz = 0.01
z = 100
space_steps = int(z / dz)

time_steps = 50000
dt = 0.12 / time_steps
alpha = 2
starting_temp = 30
oscillations = 20

def f(T):
    d2T_dz2 = (T[:-2] - 2 * T[1:-1] + T[2:]) / dz**2
    return alpha * d2T_dz2


def euler_solve(Ts):
    for t in range(time_steps-1):
        Ts[t+1, 1:-1] = Ts[t, 1:-1] + dt * f(Ts[t])
        Ts[t+1, -1] = Ts[t+1, -2]
    return Ts

start = time.time()

ts = np.linspace(0, 0.12, time_steps)

start_iterspeed = time.time()
Ts = starting_temp * np.ones((time_steps, space_steps), dtype=np.float64)
stop_iterspeed = time.time()

Ts[:, 0] = starting_temp - oscillations * np.sin(2 * np.pi * ts / 12)

euler = euler_solve(Ts)

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
