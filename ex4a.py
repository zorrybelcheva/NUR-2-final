import numpy as np


H_0 = 7.16*10**-11    # Hubble constant, units = yr^(-1)
omega_m = 0.3
omega_lambda = 0.7


# Integrate func from a to b until desired accuracy eps is met.
# Default is eps = 1e-5, starting number of steps is n = 1000.
# Breaks and returns most recent value if iterations are > 1e6
def integrate(func, a, b, eps=1e-5, n=1000):
    i = 0
    while True:
        step = (b-a)/n
        # Don't start at a to avoid potential singularities of the function:
        s = -0.5 * step * (func(a+step*0.0001) + func(b))
        s0 = -0.5 * step * (func(a+step*0.0001) + func(b))
        for k in range(1, n+1):
            # if i == N: s0 = s
            s = s0
            s0 += step*func(a+k*step)
        if abs(s-s0) < eps:
            break
        elif i > 1000000:
            print('Too many iterations')
            break
        else:
            n *= 2
            i += 1
    return s0


# Linear growth factor - only expression in the integral!
# -> expressed in terms of z
def growth_factor_z(z):
    return (1+z)/(omega_m*(1+z)**3+omega_lambda)**(3/2)


# Linear growth factor - only expression in the integral!
# -> expressed in terms of a = 1/(1+z)
def growth_factor_a(a):
    return (1/a**3)/(omega_m/a**3 + omega_lambda)**(3/2)


# Redshift dependent Hubble constant
def H(z):
    return H_0*np.sqrt(omega_m*(1+z)**3 + omega_lambda)


# Full expression for linear growth factor D(a), using pre-calculated
# numerically integrated value for the expression in the integral.
def D_a(a, A):
    return 5*omega_m/2*(omega_m/a**3 + omega_lambda)**(1/2)*A


# Finds the numerical derivative of function f at point b.
# Uses adaptive step size until a required precision is met.
# Default precision:    eps = 1e-12
def differentiate_at_point(f, b, eps=1e-12, args=[]):
    # Start with a large initial step-size h, calculate dy/dx for h:
    h = 0.01
    dydx = (f(b + h/2, *args) - f(b - h/2, *args)) / h

    i = 0  # to hold no. of iterations
    while True:
        h = h / 2
        d = (f(b + h/2, *args) - f(b - h/2, *args)) / h
        if abs(d - dydx) < eps:
            # print('d = ', d)
            # print('Number of iterations until precision is met: ', i)
            return d
        elif i >= 500:
            print('Too many iterations')
            break
        else:
            # proceed with improved value:
            i += 1
            dydx = d


# ------------------ EXERCISE 4A: NUMERICAL INTEGRAL -----------------
a_0 = 0
a_final = 1/51

A = integrate(growth_factor_a, a_0, a_final, eps=1e-10)
print('\nIntegral value: {:.7e}'.format(A))

D = 5*omega_m/2*np.sqrt(omega_m/a_final**3 + omega_lambda)*A
print('\nLinear growth factor at z = 50 (a = 1/51): {:.8e}'.format(D))
