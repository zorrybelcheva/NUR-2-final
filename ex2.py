import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# Function that creates a sample of complex numbers with
# real and imaginary parts that are normally distributed
def create_rand_complex(size, mu=0, sigma=1):
    re, im = create_rand_normal_2d(size)
    re = mapping(re, mu, sigma)
    im = mapping(im, mu, sigma)
    complex_rands = re + im*complex(0, 1)
    return complex_rands


# Calculates power spectrum from k = (kx, ky)
# returns sqrt(k**n)
def power_spectrum(kx, ky, n):
    if kx == 0 and ky == 0:
        return 0
    return np.sqrt(kx*kx + ky*ky)**n


size = 1024

# Placeholders for k = (kx, ky) vector:
kx = list(np.linspace(0, 512, num=513))+list(np.linspace(-511, -1, num=511))
ky = list(np.linspace(0, 512, num=513))+list(np.linspace(-511, -1, num=511))
k_ampl = np.zeros(shape=(size * size))

for n in [-1, -2, -3]:
    # Create random field by reshaping the vector of rands and FT it:
    field = np.fft.fft2(create_rand_normal_2d(size ** 2).reshape((size, size)))

    # Create k-amplitude vectorised matrix:
    for i in range(int(size)):
        for j in range(int(size)):
            k_ampl[j+i*size] = power_spectrum(kx[i], ky[j], n)

    # Map standard normal field to the desired standand dev. by multiplying,
    # as given by power spectrum law:
    field = field.reshape(size**2)
    field *= k_ampl
    # Reshape again and inverse FT:
    field_ft = np.fft.ifft2(field.reshape((size, size)))

    # -------------- Plotting the random fields: ---------------
    plt.figure(abs(n), figsize=(7, 5))
    im = plt.matshow(np.absolute(field_ft), fignum=abs(n))
    ax = plt.gca()
    plt.xlabel('x, kpc')
    plt.ylabel('y, kpc')
    plt.title('Gaussian random field, $n$ = '+str(n))
    ax.xaxis.tick_bottom()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax, label='Absolute value of $k$')
    plt.savefig('./plots/GaussianField'+str(n)+'.pdf')

print('\nSome statistics on the last random field:\n')
print('Average (absolute) real part: {:.6e}'.format(np.average(abs(field_ft.real.reshape(size**2)))))
print('Average (absolute) imaginary part: {:.6e}'.format(np.average(abs(field_ft.imag.reshape(size**2)))))
print('Average (absolute) imaginary part: {:.6e}'.format(np.average(abs(field_ft.imag.reshape(size**2)))))
print('Max ratio imag/real: {:.6e}'.format(max(abs(field_ft.imag.reshape(size**2)/field_ft.real.reshape(size**2)))))

end = time.time()

print('\nTime elapsed: ', end-beg, '\n')

# Saving the latest seed to the seed.txt file:
print('Latest seed value = ', seed)
seed_file = np.append(seed_file, seed)
np.savetxt('seed.txt', seed_file, fmt='%d')
