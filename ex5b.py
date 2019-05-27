import numpy as np
import matplotlib.pyplot as plt

# -------------------------  EXERCISE 5A --------------------------------


def add_points(dim, x, y, z):
    M = np.zeros((dim + 1, dim + 1, dim + 1))

    # if len([x]) == 1:
    #     M[int(round(x)), int(round(y)), int(round(z))] += 1
    # else:
    for x_point, y_point, z_point in list(zip(x, y, z)):
        M[int(round(x_point)), int(round(y_point)), int(round(z_point))] += 1

    return M


def get_value(x, cell):
    if abs(x - cell) <= 0.5:
        return 1
    else:
        return 0


np.random.seed(121)
positions = np.random.uniform(low=0, high=16, size=(3, 1024))
x = positions[0]
y = positions[1]
z = positions[2]

dim = 16

M = add_points(dim, x, y, z)

for z_value in [4, 9, 11, 14]:
    plt.imshow(M[:, :, int(z_value)])
    plt.colorbar(label='Mass of cell')
    plt.title('Slice at z = '+str(z_value))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('./plots/slice-'+str(z_value)+'.pdf')
    plt.close()


# ----------- EXERCISE 5B: INDIVIDUAL CELLS ---------------------------
moving_particle_x = np.linspace(0, 16, 1000)

cell4 = np.zeros(1000)
cell0 = np.zeros(1000)
for i in range(1000):
    cell4[i] = get_value(moving_particle_x[i], cell=4)
    cell0[i] = get_value(moving_particle_x[i], cell=0)

plt.figure(figsize=(7, 4))
plt.plot(moving_particle_x, cell4, c='k')
plt.xlabel('x range')
plt.ylabel('Cell 4 mass value')
plt.title('Value in cell 4 for different positions of a passing particle')
plt.tight_layout()
plt.savefig('./plots/passing-particle-4.pdf')
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(moving_particle_x, cell0, c='k')
plt.xlabel('x range')
plt.ylabel('Cell 0 mass value')
plt.title('Value in cell 0 for different positions of a passing particle')
plt.tight_layout()
plt.savefig('./plots/passing-particle-0.pdf')
plt.close()
