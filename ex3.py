import numpy as np
import matplotlib.pyplot as plt

H = 7.16*10**-11    # Hubble constant, units = yr^(-1)


# Scale factor:
def a(t):
    return (3/2*H*t)**(2/3)


# Derivative of scale factor:
def dadt(t):
    return (2*H**2/(3*t))**(1/3)


# Return derivatives at point t given conditions at previous point, in r
def second_order(t, r):
    D = r[0]
    y = r[1]
    dD = y
    dy = (2/3)*D/t**2 - 4*y/(3*t)
    return np.array([dD, dy])


# Runge-Kutta 4th order method for solving ODEs:
def rk4(t, r, h, f, args=[]):
    k1 = h*f(t, r, *args)
    k2 = h*f(t+0.5*h, r+0.5*k1, *args)
    k3 = h*f(t+0.5*h, r+0.5*k2, *args)
    k4 = h*f(t+h, r+k3, *args)
    r = r + 1/6*k1 + 1/3*k2 + 1/3*k3 + 1/6*k4
    return r


# Runge-Kutta 5th order method for solving ODEs:
def rk5(t, y, h, f, args):
    k1 = h*f(t, y, *args)
    k2 = h*f(t+1/5*h, y+1/5*k1, *args)
    k3 = h*f(t+3/10*h, y+3/40*k1+9/40*k2, *args)
    k4 = h*f(t+4/5*h, y+44/45*k1-56/15*k2+32/9*k3, *args)
    k5 = h*f(t+8/9*h, y+19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4, *args)
    k6 = h*f(t+h, y+9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5, *args)
    y += 35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6

    y_star = y + 5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k6

    return y


def error_estimate(y, y_star):
    return abs(y - y_star)


def err(N, delta, scale):
    err = 0
    for i in range(N):
        err += (delta[i]/scale[i])**2

    err = np.sqrt(err/N)
    return err


# Analytic solutions of the ODE in the problem, specify case of IC:
def analytic_D(t, initial_condition):
    initial_condition = int(initial_condition)
    if initial_condition == 1:
        return 3*t**(2/3)
    elif initial_condition == 2:
        return 10/t
    elif initial_condition == 3:
        return 3*t**(2/3) + 2/t


# ----------------- EXERCISE 3: SOLVING THE ODE --------------------
# List of initial conditions:
D_ini = [3, 10, 5]
y_ini = [2, -10, 0]
initial_conditions = list(zip(D_ini, y_ini))

# Placeholders for the variables:
t = np.linspace(1, 1000, 10000)
D = np.zeros(len(t))
y = np.zeros(len(t))    # y = dD/dt

condition = int(1)
eps = 1e-4

for D_ini, y_ini in initial_conditions:
    y[0] = y_ini
    D[0] = D_ini

    h = t[1]-t[0]   # step size

    for i in range(1, len(t)):
        r = np.array([D[i-1], y[i-1]])
        r = rk5(t[i-1], r, h, second_order, args=[])
        D[i] = r[0]
        y[i] = r[1]

    plt.figure(figsize=(6, 6))
    plt.plot(t, D, c='k', label='Numerical solution')
    plt.plot(t, analytic_D(t, initial_condition=condition), c='royalblue', linestyle=':', linewidth=4, label='Analytical solution')
    plt.title('$D(1) = $'+str(D_ini)+', $\quad D^{\prime}(1) = $'+str(y_ini))
    plt.xlabel('Time $t$, [years]')
    plt.ylabel('Linear growth factor $D$')
    plt.legend()
    plt.loglog()
    plt.tight_layout()
    plt.savefig('./plots/growth-factor-'+str(condition)+'.pdf')

    condition += 1
