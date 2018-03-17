import numpy as np, time


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

Ts = starting_temp * np.ones((time_steps, space_steps), dtype=np.float64)
Ts[:, 0] = starting_temp - oscillations * np.sin(2 * np.pi * ts / 12)

euler = euler_solve(Ts)

stop = time.time()
print("Numpy Euler solve - time taken: {} seconds".format(stop - start))


# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)
# Numpy Euler solve - time taken: 5.850845098495483 seconds
# 6.28s, 3836.5Mb (as measured by xtime.rb)
