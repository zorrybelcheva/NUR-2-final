import numpy as np
import matplotlib.pyplot as plt

seed_file = np.loadtxt('seed.txt', unpack=True)
seed = seed_file[-1].astype(np.int64)
print('\nFirst seed value = ', seed, '\n')


def trapezium(a, b, N, f, args=[]):
    h = (b - a) / (N + 1)

    # starting from a + h/8 to avoid possible problems
    # if f(a) = undef, which is a common situation
    s = (f(a + h/10, *args) + f(b, *args)) * h / 2
    for i in range(1, N - 1):
        a += h
        s += f(a, *args) * h
    return s


def integrate_to_eps(a,b, N, f, args=[], eps=1e-5):
    i = 0
    while True:
        s = trapezium(a, b, N, f, args)
        s_improved = trapezium(a, b, int(N*2), f, args)
        i += 1
        if abs(s_improved - s) < eps:
            break
        elif i > 50000:
            print('Too many iterations')
            break
    return s_improved


def integrate(func, a, b, eps, N):
    i = 0
    while True:
        step = (b-a)/N
        s = -0.5 * step * (func(a) + func(b))
        s0 = -0.5 * step * (func(a) + func(b))
        for i in range(N+1):
            # if i == N: s0 = s
            s = s0
            s0 += step*func(a+i*step)
        if abs(s-s0) < eps:
            break
        elif i > 10000000:
            print('Too many iterations')
            break
        else:
            N = 2*N
            i += 1
    return s0


def expansion(z, omega_m=0.3, omega_lambda=0.7):
    return (1+z)/(omega_m*(1+z)**3+omega_lambda)**(3/2)


def f(x):
    return np.exp(-x**2)


# i = integrate_to_eps(-1, 1, 100, f, eps=1e-6)
i = integrate(f, -1, 1, eps=1e-7, N=1000)
print(i)
