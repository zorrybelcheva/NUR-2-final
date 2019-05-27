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


# ------------------ EXERCISE 4A: NUMERICAL DERIVATIVE -----------------
a_0 = 0
a_final = 1/51

A = integrate(growth_factor_a, a_0, a_final, eps=1e-10)

D_prime = -15/4*omega_m**2*H_0*A/a_final**3
print('\nFirst derivative of LGF at z = 50 (a = 1/51): {:.8e}'.format(D_prime))

D_prime_numerical = a_final*H(1/a_final - 1)*differentiate_at_point(D_a, b=a_final, eps=1e-13, args=[A])
print('\nFirst derivative of LGF at z = 50 (a = 1/51), numerical: {:.8e}'.format(D_prime_numerical))
