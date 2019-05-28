import numpy as np
import matplotlib.pyplot as plt

def add_points_cic(M, dim, x, y, z):

    for i in range(dim+1):
        for j in range(dim+1):
            for k in range(dim+1):
                vx = get_value_cic(i, x)
                vy = get_value_cic(j, y)
                vz = get_value_cic(k, z)
                volume = vx*vy*vz
                M[i, j, k] += volume
    return M


def get_value(x, cell):
    if abs(x - cell) <= 0.5:
        return 1
    else:
        return 0


def get_value_cic(x, cell):
    if abs(x - cell) <= 0.5:
        return 1 - abs(x-cell)
    else:
        return 0


def get_value_cic_3d(x, y, z, cell_x, cell_y, cell_z):
    if cell_x - 0.5 < x < cell_x + 0.5 and cell_y - 0.5 < y < cell_y + 0.5 and cell_z - 0.5 < z < cell_z + 0.5:
        dx = abs(cell_x - x)
        dy = abs(cell_y - y)
        dz = abs(cell_z - z)
        volume = dx*dy*dz
        return volume
    # if abs(x - cell) <= 0.5:
    #     return 1 - abs(x-cell)
    else:
        return 0



np.random.seed(121)
positions = np.random.uniform(low=0, high=16, size=(3, 1024))
x = positions[0]
y = positions[1]
z = positions[2]

dim = 16


# ----------- EXERCISE 5C: CLOUD IN CELL ---------------------------
moving_particle_x = np.linspace(0, 16, 1000)

cic4 = np.zeros(1000)
cic0 = np.zeros(1000)
for i in range(1000):
    cic0[i] = get_value_cic(moving_particle_x[i], cell=0)
    cic4[i] = get_value_cic(moving_particle_x[i], cell=4)

plt.figure(figsize=(7, 4))
plt.plot(moving_particle_x, cic4, c='k')
plt.xlabel('x range')
plt.ylabel('Cell 4 mass value')
plt.title('Value in cell 4 for different positions of a passing particle \nCloud In Cell')
plt.tight_layout()
plt.savefig('./plots/cic-4.pdf')
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(moving_particle_x, cic0, c='k')
plt.xlabel('x range')
plt.ylabel('Cell 0 mass value')
plt.title('Value in cell 0 for different positions of a passing particle')
plt.tight_layout()
plt.savefig('./plots/cic-0.pdf')
plt.close()


M = np.zeros((dim + 1, dim + 1, dim + 1))
for i in range(len(x)):
    M = add_points_cic(M, dim, x[i], y[i], z[i])

for z_value in [4, 9, 11, 14]:
    plt.imshow(M[:, :, int(z_value)])
    plt.colorbar(label='Mass of cell')
    plt.title('Cloud In Cell method \nSlice at z = ' + str(z_value))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('./plots/cic-slice-' + str(z_value) + '.pdf')
    plt.close()
