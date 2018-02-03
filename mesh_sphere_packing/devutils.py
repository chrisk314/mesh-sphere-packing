from __future__ import print_function

import os
from pprint import pprint

import numpy as np


def write_geomview(points, tris, fname):
    with open(fname, 'w') as f:
        f.write('OFF\n')
        f.write('%d %d %d\n' % (len(points), len(tris), 0))
        for p in points:
            f.write('%+1.15e %+1.15e %+1.15e\n' % (p[0], p[1], p[2]))
        for t in tris:
            f.write('3 %d %d %d\n' % (t[0], t[1], t[2]))


def output_segments_geomview(segments):
    for i, (points, tris) in enumerate(segments):
        write_geomview(points, tris, './segment_%d.off' % i)


def output_boundaries_geomview(boundaries):
    for i, (points, tris, _) in enumerate(boundaries):
        write_geomview(points, tris, './boundary_%d.off' % i)


def write_poly(points, edges, holes, fname):
    with open(fname, 'w') as f:
        f.write('%d 2 0 0\n' % len(points))
        for i, p in enumerate(points):
            f.write('%d %+1.15e %+1.15e\n' % (i, p[0], p[1]))
        f.write('%d 0\n' % len(edges))
        for i, e in enumerate(edges):
            f.write('%d %d %d\n' % (i, e[0], e[1]))
        f.write('%d\n' % len(holes))
        for i, h in enumerate(holes):
            f.write('%d %+1.15e %+1.15e\n' % (i, h[0], h[1]))


def output_boundaries_poly(boundaries):
    for i, (points, edges, holes) in enumerate(boundaries):
        write_poly(points, edges, holes, './boundary_%d.poly' % i)


def output_tetmesh_poly(points, facets, markers, holes):
    with open('mesh.poly', 'w') as f:
        f.write('%d 3 0 1\n' % len(points))
        for i, p in enumerate(points):
            f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, p[0], p[1], p[2]))
        f.write('%d 1\n' % len(facets))
        for i, (fac, m) in enumerate(zip(facets, markers)):
            f.write('1 0 %d\n%d %d %d %d\n' % (m, 3, fac[0], fac[1], fac[2]))
        if len(holes):
            f.write('%d\n' % len(holes))
            for i, h in enumerate(holes):
                f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, h[0], h[1], h[2]))
        else:
            f.write('0\n')


def plot_points_edges(points, edges):
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    lc = LineCollection(points[edges])
    fig = plt.figure()
    plt.gca().add_collection(lc)
    space = 0.05 * points.max()
    plt.xlim(points[:,0].min() - space, points[:,0].max() + space)
    plt.ylim(points[:,1].min() - space, points[:,1].max() + space)
    plt.plot(points[:,0], points[:,1], 'ro')
    fig.savefig('points_edges.png')
    plt.show()


def read_geomview(fname):
    with open(fname, 'r') as f:
        f.readline()
        counts = [int(tok) for tok in f.readline().strip().split()]
        points = np.array([
            [float(tok) for tok in f.readline().strip().split()]
            for i in range(counts[0])
        ])
        tris = np.array([
            [int(tok) for tok in f.readline().strip().split()[1:]]
            for i in range(counts[1])
        ])
    return points, tris


def load_geomview_files(prefix):
    geomfiles = [
        fname for fname in os.listdir(os.getcwd())
        if fname.startswith(prefix) and os.path.splitext(fname)[1] == '.off'
    ]
    geomfiles.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    return [read_geomview(fname) for fname in geomfiles]


def plot_segment_points_3d(data):
    """This function is for visualising sphere points to aid debugging/development
    """
    # TODO : remove this after development complete
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], zs=data[:,2])
    plt.show()


def plot_polys(polys):
    """This function is for visualising sphere segments to aid debugging/development
    """
    # TODO : remove this after development complete
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fc = ["crimson" if i%2 else "gold" for i in range(len(polys))]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fc, linewidths=1))
    plt.show()


def plot_added_points(particles, points, L, axis):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_artist(Rectangle([0., 0.], L[0], L[1], fill=False))
    circles = [
        Circle(p[[(axis+1)%3,(axis+2)%3]], radius=p[3], color='b', fill=False)
        for p in particles
    ]
    for c in circles:
        ax.add_artist(c)
    ax.scatter(points[:,0], points[:,1], marker='.')
    plt.axis('off')
    plt.xlim([-0.1, L[0]+0.1])
    plt.ylim([-0.1, L[1]+0.1])
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()


def test_tri_refinefunc(rfunc=None):
    from meshpy import triangle
    L = np.array([2., 2.])
    points = np.array([
        [0., 0.], [L[0], 0.], [L[0], L[1]], [0., L[1]]
    ])
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0]
    ])
    mesh_data = triangle.MeshInfo()
    mesh_data.set_points(points)
    mesh_data.set_facets(edges.tolist())

    max_volume = 0.2**2
    min_angle = 20.

    mesh = triangle.build(
        mesh_data,
        max_volume=max_volume,
        min_angle=min_angle,
        refinement_func=rfunc
    )

    # Extract triangle vertices from triangulation adding back x coord
    points = np.column_stack((
        np.zeros(len(mesh.points)), np.array(mesh.points))
    )
    tris = np.array(mesh.elements, dtype=np.int32)
    holes = np.empty((0,3), dtype=np.float64)

    return points, tris, holes


ONE_THIRD = 0.3333333333333333
TARGET_AREA = 0.2**2
Lx, Ly = 2., 2.
INV_Lx, INV_Ly = 1. / Lx, 1. / Ly
particles = np.array([[0.5, 0.5, 0.5, 0.4], [0.8, 1., 1., 0.6]]).tolist()
TARGET_AREA_GRID = [[TARGET_AREA, 0.1 * TARGET_AREA],[0.1 * TARGET_AREA, TARGET_AREA]]
nx, ny = 2, 2
inv_dx, inv_dy = nx / Lx, ny / Ly


def rfunc(vertices, area):
    (ox, oy), (dx, dy), (ax, ay) = vertices
    cx = ONE_THIRD * (ox + dx + ax)  # Triangle center x coord.
    cy = ONE_THIRD * (oy + dy + ay)  # Triangle center y coord.
    ix = int(cx * inv_dx)
    iy = int(cy * inv_dy)
    target_area = TARGET_AREA_GRID[ix][iy]
    return int(area > target_area)  # True -> 1 means refine
