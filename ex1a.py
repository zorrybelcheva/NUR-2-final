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


def lcg_float(float=True, a=np.int64(22695477), c=1, m=np.int64(2**32)):  # Borland C/C++
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


def avg(x):
    return sum(x)/len(x)


def standard_dev(x):
    N = len(x)
    x_avg = avg(x)
    s = 0
    for i in range(N):
        s += (x[i]-x_avg)*(x[i]-x_avg)
    s = np.sqrt(s/N)
    return s


# ---------- FIRST TEST: random no.s against each other ----------
print('First test: rands against each other...\n')
N = 1000
xrand = create_random(size=N)

plt.figure(1)
plt.plot(xrand[:-1], xrand[1:], '.', markersize=1, color='k')
plt.xlabel('Random number value')
plt.ylabel('Random number value')
plt.title('Random floating point numbers against each other')
plt.tight_layout()
plt.savefig('plots/RNGtest1.pdf')


# ---------- SECOND TEST: random no.s against their index --------
print('Second test: rands against index...\n')
plt.figure(2)
plt.bar(range(N), xrand, width=1, facecolor='maroon')
plt.xlabel('Random number index')
plt.ylabel('Random number value')
plt.tight_layout()
plt.savefig('plots/RNGtest2.pdf')


# ------------ THIRD TEST: binning the random numbers ------------
print('Third test: uniformity...\n')
N2 = 1000000
xrand2 = create_random(size=N2)

plt.figure(3)
plt.hist(xrand2, bins=20, facecolor='grey')
plt.xlabel('Bins, width 0.05')
plt.ylabel('Count, numbers per bin')
plt.title('Testing uniformity of RNG')
plt.savefig('plots/RNGtest3.pdf')


# ------------------- Testing the RNG further: --------------------
# ---- Are the bin counts reasonably scattered around the mean? ---
stdev = standard_dev(np.histogram(xrand2, bins=20)[0])
mean = avg(np.histogram(xrand2, bins=20)[0])
print('Bins of 0.05 for the first 1 000 000 numbers:\n'
      'std = {:.1f}, mean = {:.1f}'.format(stdev, mean))

plt.figure()
plt.hist(xrand2, bins=20, facecolor='grey')
plt.ylim(49500, 50500)
plt.axhline(mean, c='black', label='Mean bin count')
plt.axhline(mean-2*stdev, c='blue', linestyle='--', label='2$\sigma$ interval')
plt.axhline(mean+2*stdev, c='blue', linestyle='--')
plt.legend(loc='upper right')
plt.title('Testing uniformity of RNG')
plt.xlabel('Bins, width 0.05')
plt.ylabel('Counts, numbers per bin')
plt.tight_layout()
plt.savefig('plots/RNGtest3_zoomin.pdf')


# Saving the latest seed to the seed.txt file:
print('\nLatest seed value = ', seed)
seed_file = np.append(seed_file, seed)
np.savetxt('seed.txt', seed_file, fmt='%d')
