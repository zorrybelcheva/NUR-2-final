import numpy as np
import matplotlib.pyplot as plt
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


# Function that performs KS test: is the sample x drawn from
# a distribution with the given cdf?
# Returns KS statistic D and probability P.
# See report and 'Numerical Recipes' for more info.
def ks_test(x, cdf, args):
    def Q(z):
        if z == 0:
            return 1
        elif z < 1.18:
            v = np.exp(-np.pi*np.pi/8/z/z)
            P = np.sqrt(2*np.pi)/z*(v + v**9 + v**25)
        else:
            v = np.exp(-2*z*z)
            P = 1 - 2*(v-v**4+v**9)
        return 1-P

    # The bigger the number of bins, the better the KS test
    N = len(x)
    bin_number = int(100*(max(x)-min(x)))
    counts, bins = np.histogram(x, bins=bin_number)
    width = bins[1]-bins[0]
    # bins += width
    dist = np.zeros(len(counts))
    counts_array = np.zeros(len(counts))

    for i in range(len(counts)):
        dist[i] = abs(sum(counts[:i])/N-cdf(bins[i], *args))
        counts_array[i] = sum(counts[:i])

    D = max(dist)
    N_sqrt = np.sqrt(N)
    z = D*(N_sqrt + 0.12 + 0.11/N_sqrt)
    return D, Q(z)


# Returns G(x | mu, sigma). Default: standard normal:
#                           G(x | mu = 0, sigma = 1)
def gaussian(x, mu=0, var=1.):
    G = 1/(np.sqrt(2*np.pi*var))*np.exp(-(x-mu)*(x-mu)/(2*var))
    return G


# Get the value of CDF(x) for sample = sample
def get_cdf_value(x, sample):
    # total = len(sample)
    # no = 0
    # # instead of using np.where():
    # for i in range(total):
    #     if sample[i] < x:
    #         no += 1
    return len(sample[sample <= x])/len(sample)


# --------------- EXERCISE 1E: 10 DATA SETS ---------------
sets = np.loadtxt('randomnumbers.txt')
set_size = len(sets[:, 0])
set_no = len(sets[0, :])

sample_sizes = np.logspace(1, np.log10(set_size), num=30).astype(np.int64)
ks_tests = np.zeros(30)
p = np.zeros(30)

sample = create_rand_normal_2d(size=sample_sizes[-1])

for i in range(set_no):
    j = 0

    for size in sample_sizes:
        x_drawn = sample[:size]
        # KS-test on the j-th data set, for [size] data points:
        ks_tests[j], p[j] = ks_test(sets[:size, i], get_cdf_value, [x_drawn])

        j += 1

    # mid = time.time()
    # print('After set '+str(j)+': ', mid - beg)

    fig, (ax2, ax1, ax3) = plt.subplots(3, 1, figsize=(7, 6))
    ax1.plot(sample_sizes, ks_tests, color='k', label='KS test statistic')
    ax1.set_ylim(1e-3, 1)
    ax1.set_xlabel('Sample size')
    ax1.set_ylabel('KS statistic, D')
    ax1.loglog()
    ax1.legend(fontsize='small', loc='lower left')

    ax2.plot(sample_sizes, p, c='k', label='Probability')
    ax2.set_ylim(1e-4, 1.2)
    ax2.axhline(0.05, linestyle='--', c='maroon', label='2$\sigma$ confidence')
    ax2.axhline(0.003, linestyle='--', c='crimson', label='3$\sigma$ confidence')
    ax2.loglog()
    ax2.legend(fontsize='small', loc='lower left')
    ax2.set_xlabel('Sample size')
    ax2.set_ylabel('Probability')
    ax2.set_title('Data set '+str(i+1))

    counts, bins = np.histogram(sets[:, i], bins=30)
    width = bins[1] - bins[0]
    bins += width
    c = max(counts)
    ax3.bar(bins[:-1], counts/c, facecolor='maroon', label='Data set ' + str(i+1))
    x = np.linspace(-5, 5, 1000)
    ax3.plot(x, gaussian(x)/max(gaussian(x)), c='k', label='Gaussian($\mu=0, \sigma=1)$')
    ax3.set_xlabel('True numbers')
    ax3.set_ylabel('Normalised bin count')
    ax3.legend(fontsize='small')
    ax3.set_xlim(min(-5, min(bins)), max(5, max(bins[:-1])))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    fig.savefig('plots/KStest-set'+str(i+1)+'.png', dpi=300)

end = time.time()

print('\nTime elapsed: ', end-beg, '\n')

# Saving the latest seed to the seed.txt file:
print('\nLatest seed value = ', seed)
seed_file = np.append(seed_file, seed)
np.savetxt('seed.txt', seed_file, fmt='%d')
