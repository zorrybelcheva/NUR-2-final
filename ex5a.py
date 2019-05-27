import numpy as np
import matplotlib.pyplot as plt

# -------------------------  EXERCISE 5A --------------------------------


def add_points(dim, x, y, z):
    M = np.zeros((dim + 1, dim + 1, dim + 1))

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
    plt.title('Slice at z = ' + str(z_value))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('./plots/slice-' + str(z_value) + '.pdf')
    plt.close()
