import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import h5py

np.random.seed(121)


class Point:
    def __init__(self, x, y, mass, ID):
        self.x = x
        self.y = y
        self.mass = mass
        self.ID = ID


class Node:
    def __init__(self, x0, y0, width, height, points, moment=0, parent=None):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.points = points
        self.children = []
        self.is_leaf = False
        self.moment = moment
        self.parent = parent

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self):
        return self.points

    def get_parent(self):
        return self.parent


class QuadTree:
    def __init__(self, xlim, ylim, point_limit, width, height):
        self.point_limit = point_limit
        self.xlim = xlim
        self.ylim = ylim
        self.width = width
        self.height = height
        self.points = []
        self.root = Node(xlim, ylim, width, height, self.points)

    def get_points(self):
        return self.points

    def add_point(self, x, y, mass, ID):
        self.points.append(Point(x, y, mass, ID))

    def subdivide(self):
        self.root = subdivide(self.root, self.point_limit)

    def calculate_multipole(self):
        calculate_multipole(self.root)

    def graph(self, xlabel='', ylabel='', filename='./plots/quadtree', save=False, zoomreg=None, point=None):
        fig, ax = plt.subplots(1, figsize=(7, 7))
        plt.xlim(self.xlim, self.xlim + self.width)
        plt.ylim(self.ylim, self.ylim + self.height)

        children = find_children(self.root)

        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        ax.plot(x, y, '*', c='maroon', alpha=0.5)

        for child in children:
            ax.add_patch(patches.Rectangle((child.x0, child.y0), child.width, child.height, fill=False))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.show()

        if save:
            plt.savefig(filename+'.pdf')
            print('\nQuadtree plotted and saved!')

        if zoomreg is not None:
            plt.xlim(zoomreg[0], zoomreg[1])
            plt.ylim(zoomreg[2], zoomreg[3])
            plt.savefig(filename+'-zoom.pdf')
            print('\nZoom-in region also plotted and saved!')

        if point is not None:
            plt.plot(point.x, point.y, 'x', markersize=7, c='blue', label='Point ID = '+str(point.ID))
            plt.legend(loc='lower right')
            plt.savefig(filename+'-point.pdf')
            print('\nPoint of search also plotted and saved!')

        plt.close()


def search_point(node, ID):
    n_points = len(node.get_points())

    for i in range(n_points):
        if node.points[i].ID == ID:
            if node.is_leaf:
                print('Leaf reached: ({:.2f}, {:.2f}) + ({:.2f}, {:.2f}); \n\tn = 0 multipole moment: {:.3f}' \
                      .format(node.x0, node.y0, node.width, node.height, node.moment))
            else:
                print('In ({:.2f}, {:.2f}) + ({:.2f}, {:.2f}); \n\tn = 0 multipole moment: {:.3f}' \
                      .format(node.x0, node.y0, node.width, node.height, node.moment))
                for child in node.children:
                    search_point(child, ID)


def subdivide(node, point_limit):

    if len(node.points) == 0:
        node.is_leaf = True
        return

    elif int(len(node.points)) <= point_limit:
        node.is_leaf = True
        return

    elif len(node.points) > point_limit:
        w_reduced = node.width*0.5
        h_reduced = node.height*0.5

        points_in_sw = check_points(node.x0, node.y0, w_reduced, h_reduced, node.points)
        sw = Node(node.x0, node.y0, w_reduced, h_reduced, points_in_sw, parent=node)

        points_in_nw = check_points(node.x0, node.y0 + h_reduced, w_reduced, h_reduced, node.points)
        nw = Node(node.x0, node.y0 + h_reduced, w_reduced, h_reduced, points_in_nw, parent=node)

        points_in_se = check_points(node.x0 + w_reduced, node.y0, w_reduced, h_reduced, node.points)
        se = Node(node.x0 + w_reduced, node.y0, w_reduced, h_reduced, points_in_se, parent=node)

        points_in_ne = check_points(node.x0 + w_reduced, node.y0 + h_reduced, w_reduced, h_reduced, node.points)
        ne = Node(node.x0 + w_reduced, node.y0 + h_reduced, w_reduced, h_reduced, points_in_ne, parent=node)

        node.children = [sw, nw, se, ne]

        for child in node.children:
            subdivide(child, point_limit)

    return node


def check_points(x_node, y_node, width, height, points):
    true_points = []

    for point in points:
        if x_node < point.x <= x_node + width and y_node < point.y <= y_node + height:
            true_points.append(point)

    return true_points


def find_children(node):
    children = []

    if node.is_leaf:
        children = [node]
    else:
        for child in node.children:
            children += (find_children(child))

    return children


def calculate_multipole(node):
    for i in range(len(node.points)):
        node.moment += node.points[i].mass

    if not node.is_leaf:
        for child in node.children:
            calculate_multipole(child)


# ---------------- Simple test with 100 rands -----------------------
# tree = QuadTree(0, 0, 1, 1, 1)
# x = np.random.uniform(size=100)
# y = np.random.uniform(size=100)
#
# for i in range(len(x)):
#     tree.add_point(x[i], y[i])
#
# tree.subdivide()
# tree.graph()


# ---------------- EXERCISE 7: BARNES-HUT QUADTREE -----------------

particles = h5py.File('colliding.hdf5', 'r')['PartType4']
coords = particles['Coordinates']
masses = particles['Masses'][()]
IDs = particles['ParticleIDs'][()]
vel = particles['Velocities'][()]

xlim = 0
ylim = 0
width = 150
height = 150
point_limit = 12

gals = QuadTree(xlim, ylim, point_limit, width, height)

for i in range(len(coords[:, 0])):
    gals.add_point(coords[i, 0], coords[i, 1], masses[i], IDs[i])

zoomreg = [35, 115, 35, 115]

gals.subdivide()
gals.calculate_multipole()

print('\nSearching for point with ID = ', IDs[100])

search_point(gals.root, IDs[100])

print('\nTrue point coordinates: ', (coords[100, 0], coords[100, 1]))

p = Point(coords[100, 0], coords[100, 1], masses[100], IDs[100])

gals.graph(xlabel='x coordinate', ylabel='y coordinate', save=True, zoomreg=zoomreg, point=p)
