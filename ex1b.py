import numpy as np
import matplotlib.pyplot as plt

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
    a1 = np.int64(21)
    a2 = np.int64(35)
    a3 = np.int64(4)
    seed = seed ^ (seed >> a1)
    seed = seed ^ (seed << a2)
    seed = seed ^ (seed >> a3)
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


# Maps a standard normal sample to follow a Gaussian distribution
# with different mean mu and standard deviation sigma.
def mapping(sample, mu, sigma):
    sample_transformed = sample * sigma
    sample_transformed += mu
    return sample_transformed


# Returns G(x | mu, sigma). Default: standard normal:
#                           G(x | mu = 0, sigma = 1)
def gaussian(x, mu=0, var=1.):
    G = 1/(np.sqrt(2*np.pi*var))*np.exp(-(x-mu)**2/(2*var))
    return G


# ------------ EXERCISE 1B: Box-Muller method --------------------
print('1B: Box-Muller transforms...')
n = 1000
x1, x2 = create_rand_normal_2d(n, two_dim=True)
sample = x1, x2
mu = 3
sigma = 2.4
var = sigma*sigma
density_interval = np.ceil(6*sigma).astype(int)  # int no of bins for hist: covers (-3*sigma,3*sigma) = 6*sigma interval

# Reference Gaussian: covers 5-sigma interval:
x = np.linspace(mu-5*sigma, mu+5*sigma, 1000)
y = gaussian(x, mu, var)

x1_new = mapping(x1, mu, sigma)
x2_new = mapping(x2, mu, sigma)

print('Plotting histogram and Gaussian...')

plt.figure()
counts, bin_edges = np.histogram(x1_new, bins=density_interval)
width = bin_edges[1] - bin_edges[0]
bin_edges += 0.5 * width        # shift bin edges to plot bars centered <- important!
plt.bar(bin_edges[:-1], counts / n, width=width, color='royalblue', alpha=0.8, label='Normalised Box-Miller sample 1')

counts2, bin_edges2 = np.histogram(x2_new, bins=density_interval)
width2 = bin_edges2[1] - bin_edges2[0]
bin_edges2 += 0.5 * width2      # shift bin edges to plot bars centered <- important!
plt.bar(bin_edges2[:-1], counts2 / n, width=width2, color='maroon', alpha=0.8, label='Normalised Box-Miller sample 2')

plt.axvline(mu, c='dimgrey', label='New mean $\mu$ = '+str(mu))
plt.axvline(mu+sigma, c='navy', linestyle='--', alpha=0.8, label='1$\sigma$')
plt.axvline(mu-sigma, c='navy', linestyle='--', alpha=0.8)
plt.axvline(mu+2*sigma, c='slateblue', linestyle='--', alpha=0.5, label='2$\sigma$')
plt.axvline(mu-2*sigma, c='slateblue', linestyle='--', alpha=0.5)
plt.axvline(mu-2*sigma, c='slateblue', linestyle='--', alpha=0.5)
plt.axvline(mu+3*sigma, c='mediumorchid', linestyle='--', alpha=0.8, label='3$\sigma$')
plt.axvline(mu-3*sigma, c='mediumorchid', linestyle='--', alpha=0.8)
plt.axvline(mu+4*sigma, c='hotpink', linestyle='--', alpha=0.8, label='4$\sigma$')
plt.axvline(mu-4*sigma, c='hotpink', linestyle='--', alpha=0.8)
plt.plot(x, y, c='k', label='Gaussian, for comparison')
plt.legend(fontsize='xx-small', loc='upper right')
plt.xlabel('Random number')
plt.ylabel('Probability density function / normalised bin count')
plt.tight_layout()
plt.savefig('plots/box-muller.pdf')
# plt.close()

print('Plotting pairs of normal rands...')

plt.figure(figsize=(5, 5))
plt.plot(x1, x2, '.', markersize=6, c='k', alpha=0.5, label='Gaussian random number pairs')
plt.plot(0, 0, 'x', markersize=5, c='r', label='Center, (0,0)')
plt.xlabel('Random number')
plt.ylabel('Random number')
plt.legend(fontsize='xx-small', loc='lower right')
plt.tight_layout()
plt.savefig('plots/normal-rands-2d.pdf')
# plt.close()


# Saving the latest seed to the seed.txt file:
print('\nLatest seed value = ', seed)
seed_file = np.append(seed_file, seed)
np.savetxt('seed.txt', seed_file, fmt='%d')
