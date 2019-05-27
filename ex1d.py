import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import kuiper
import time

beg = time.time()

seed_file = np.loadtxt('seed.txt', unpack=True)
seed = seed_file[-1].astype(np.int64)
print('\nFirst seed value = ', seed, '\n')


'''
    Linear congruential RNG:  using Borland C/C++ constants
    when float=True output is the seed and a random float;
    when float=False output is just the seed
    ------------------------------------------------------------------
    other possibilities for constants include:
    MLCG: a = 16807, c = 0, m = 2147483647 - okay, but 1 bin
          is underrepresented for some reason.
    other:a = 1664525, c = 1013904223, m = 2147483648*2 - not that good
'''


def lcg_float(float=True, a=22695477, c=1, m=2**32):  # Borland C/C++
    global seed
    seed = np.int64(seed)
    seed = (np.int64(np.int64(a*seed) + c)) % m
    if float:
        return np.float64(seed/m)
    else:
        return np.int64(seed)


# 64-bit XOR-shift RNG using 3 bitwise shifts;
# Outputs 64 bit integer, alters the global seed.
def xorshift():
    global seed
    seed = np.int64(seed)
    a1 = np.int64(13)
    a2 = np.int64(7)
    a3 = np.int64(17)
    seed = np.int64(seed ^ (seed >> a1))
    seed = np.int64(seed ^ (seed << a2))
    seed = np.int64(seed ^ (seed >> a3))
    return np.int64(seed)


# Combined RNG: calls xorshift() to shift the global seed,
# then uses lcg_float() to produce a random number in (0,1).
# Can generate a sample of the desired size by altering the
# respective size parameter. Default is 1.
def create_random(size=1):
    global seed
    if seed % 2 == 0:
        seed += 1

    rand = np.zeros(int(size))
    for i in range(int(size)):
        xorshift()
        rand[i] = lcg_float()
    return rand


# Applies Box-Muller transformation method to create 2D
# normally-distrubuted random numbers. Uses z1 and z2 drawn
# from a uniform distribution through create_random(). The
# size parameter can be changed to produce a sample of
# different size, default size = 1, meaning output is a
# number *pair*. Output is always 2D.
def create_rand_normal_2d(size=1, two_dim=False):
    z1 = create_random(size)
    z2 = create_random(size)
    rand1 = np.cos(2*np.pi*z2)*np.sqrt(-2*np.log(z1))
    rand2 = np.sin(2*np.pi*z2)*np.sqrt(-2*np.log(z1))
    if two_dim:
        return rand1, rand2
    else:
        return rand1


# Function that performs Kuiper test on sample x to test
# how likely it is that x was drawn from a distribution
# with the given cdf.
# Returns Kuiper statistic V and probability P.
# For more info see report.
def kuiper_test(x, cdf, args):
    def Q_kuiper(z):
        if z < 0.4:
            return 1
        else:
            v = np.exp(-2*z*z)
            Q = 2*((4*z*z-1)*v + (16*z*z-1)*v**4 + (32*z*z-1)*v**9)
        return Q

    # The bigger the number of bins, the better the KS test
    N = len(x)
    bin_number = int(100*(max(x)-min(x)))
    counts, bins = np.histogram(x, bins=bin_number)
    width = bins[1]-bins[0]
    bins += width
    dist = np.zeros(len(counts))
    c = sum(counts)
    counts_array = np.zeros(len(counts))
    for i in range(len(counts)):
        dist[i] = sum(counts[:i])/c-cdf(bins[i], *args)
        counts_array[i] = sum(counts[:i])

    V = abs(max(dist)) + abs(min(dist))
    z = V*(np.sqrt(N) + 0.155 + 0.24/np.sqrt(N))
    # print(z)
    P = Q_kuiper(z)
    return V, P


# Trapezium rule for function integration:
# integrate f from a to b in N steps; args = arguments to pass to function
def trapezium(a, b, N, f, args=[]):
    h = (b - a) / (N + 1)

    # starting from a + h/1000 to avoid possible problems
    # if f(a) = undef, which is a common situation
    s = (f(a + h*0.001, *args) + f(b, *args)) * h / 2
    for i in range(1, N - 1):
        a += h
        s += f(a, *args) * h
    return s


# Definition of Error function: calculates erf(x) through numerical integration
def error_func(x, sigma=1):
    def f(t):
        return np.exp(-t * t)
    erf = 2/np.sqrt(np.pi)*trapezium(-3*sigma, x, 1000, f, args=[])-1
    return erf


# Cumulative distribution function of Gaussian distribution:
# goes through error_func integrated numerically
def cdf_gaussian(x, mu=0, sigma=1):
    return 0.5*(1+error_func((x-mu)/(np.sqrt(2)*sigma)))


# ----------------- EXERCISE 1D: KUIPER TEST -------------------------
print('Performing Kuiper test...')
sample_sizes = np.logspace(1, 5, num=50)
p = np.zeros(50)
p_astropy = np.zeros(50)
kuiper_own = np.zeros(50)
kuiper_astropy = np.zeros(50)
index = 0

# Create a sample of the largest examined size (10^5) once, and use
# it to draw sample for later:
x = create_rand_normal_2d(size=sample_sizes[-1])

for sample_size in sample_sizes:
    x_drawn = x[:int(sample_size)]

    kuiper_astropy[index], p_astropy[index] = kuiper(x_drawn, cdf_gaussian)
    kuiper_own[index], p[index] = kuiper_test(x_drawn, cdf_gaussian, args=[])

    index += 1

fig, (ax2, ax1) = plt.subplots(2, 1)
ax1.plot(sample_sizes, kuiper_astropy, 'x', color='grey', label='Astropy Kuiper test')
ax1.plot(sample_sizes, kuiper_own, '.', color='k', label='My Kuiper test')
ax1.set_ylim(1e-3, 1)
ax1.set_xlabel('Random number sample size')
ax1.set_ylabel('Kuiper statistic, V')
ax1.loglog()
ax1.legend(fontsize='small', loc='lower left')

ax2.plot(sample_sizes, p, c='k', label='My probability')
ax2.plot(sample_sizes, p_astropy, c='grey', linestyle='--', label='Astropy probability')
ax2.set_ylim(1e-4, 1.2)
ax2.axhline(0.05, linestyle='--', c='maroon', label='2$\sigma$ confidence')
ax2.axhline(0.003, linestyle='--', c='crimson', label='3$\sigma$ confidence')
ax2.loglog()
ax2.legend(fontsize='small', loc='lower left')
ax2.set_ylabel('Probability / 1 - p-value')

fig.tight_layout()
fig.subplots_adjust(hspace=0.2)
fig.savefig('plots/Kuipertest.pdf')

end = time.time()

print('\nTime elapsed: ', end-beg, '\n')

# Saving the latest seed to the seed.txt file:
print('\nLatest seed value = ', seed)
seed_file = np.append(seed_file, seed)
np.savetxt('seed.txt', seed_file, fmt='%d')