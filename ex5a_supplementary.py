import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.dist_from_origin = np.sqrt(x*x+y*y+z*z)


class Cell:
    def __init__(self, ind, x, y, z, width, points, mass=0):
        self.index = ind
        self.bound_x = x
        self.bound_y = y
        self.bound_z = z
        self.width = width
        self.points = points
        self.mass = mass

    def add_point(self, point_x, point_y, point_z):
        self.points.append(Point(point_x, point_y, point_z))


class Grid:
    def __init__(self, dimension, cell_width, points, cells):
        self.dimension = dimension
        self.points = points
        self.cells = cells
        self.cell_width = cell_width

        self.initialise_cells()
        self.initialise_grid_points()

    def add_point(self, x, y, z):
        self.points.append(Point(x, y, z))

    def get_points(self):
        return self.points

    def get_dimension(self):
        return self.dimension

    def initialise_cells(self):
        ind = 0

        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    self.cells.append(Cell(ind, i, j, k, width=self.cell_width, points=[]))
                    ind += 1

    def initialise_grid_points(self):
        self.grid_points = []

        # Not ideal to use 3 nested loops, but for such low dimension of
        # the problem this is not too bad.
        for i in range(int(self.dimension)):
            for j in range(int(self.dimension)):
                for k in range(int(self.dimension)):
                    self.grid_points.append([i, j, k])

        self.grid_points = np.array(self.grid_points)

    def assign_masses_ngp(self, particle_positions):
        x = particle_positions[0]
        y = particle_positions[1]
        z = particle_positions[2]

        for x_point, y_point, z_point in list(zip(x, y, z)):
            x_min = np.argmin(abs(x_point - self.grid_points[:, 0]))
            y_min = np.argmin(abs(y_point - self.grid_points[:, 1]))
            z_min = np.argmin(abs(z_point - self.grid_points[:, 2]))
            # ax.scatter(x_point, y_point, z_point, '.', c='grey', s=2)
            # ax.scatter(grid_points[x_min, 0], grid_points[y_min, 1], grid_points[z_min, 2], '.', c='r', s=1)
            # plt.close(fig)
            self.cells[self.grid_points[x_min, 0]*self.dimension**2 + self.grid_points[y_min, 1]*self.dimension + self.grid_points[z_min, 2]].mass += 1


# ----------- EXERCISE 5A: NEAREST GRID POINT METHOD ---------------------------

np.random.seed(121)
positions = np.random.uniform(low=0, high=16, size=(3, 1024))

dim = 16
grid = Grid(dimension=dim, cell_width=1, points=[], cells=[])

grid.assign_masses_ngp(particle_positions=positions)

for z_value in [4, 9, 11, 14]:
    plt.figure(figsize=(6, 6))
    masked = grid.grid_points[np.where(grid.grid_points[:, 2] == z_value)[0]]
    plt.plot(masked[:, 0], masked[:, 1], '.', c='k', label='Grid points')
    plt.title('z = '+str(z_value)+' slice ')
    plt.xlabel('x')
    plt.ylabel('y')
    ind = np.isclose(positions[2], z_value, atol=0.05)
    plt.plot(positions[0, ind], positions[1, ind], '.', c='red', label='z = '+str(z_value)+'+/- 0.05')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig('./plots/grid-slice-'+str(z_value)+'.pdf')
