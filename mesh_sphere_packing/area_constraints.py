import numpy as np
from numpy import linalg as npl

from mesh_sphere_packing import ONE_THIRD, GROWTH_LIMIT


class AreaConstraints(object):

    """Constructs grids of area constraints for triangulation of domain boundaries."""

    cutoff_factor = 5.  # Cutoff distance in units of characteristic length, ds
    cell_width = 1.     # Width of grid cells in units of characteristic length, ds

    def __init__(self, domain, sphere_pieces, ds):
        """Constructs AreaConstraints object.
        :param domain Domain: spatial domain for mesh.
        :param sphere_pieces list: list of SpherePiece objects.
        :param ds float: characteristic segment length.
        """
        self.ds = ds
        self.dA = 0.5 * ds**2
        self.dA_max = self.dA * GROWTH_LIMIT
        self.cutoff = self.cutoff_factor * self.ds
        self.L = domain.L

        self.spheres = np.array(
            [[*sp.sphere.x, sp.sphere.r] for sp in sphere_pieces]
        )

        self.i_loops = []
        for sp in sphere_pieces:
            self.i_loops += sp.sphere.i_loops

        self.inv_dx, self.inv_dy = 3 * [None], 3 * [None]
        self.build_area_constraint_grids()

    def filter_spheres(self, axis):
        """Get spheres which are close to the boundary along specified axis without
        actually crossing it.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :return: tuple containing arrays of sphere close to the lower and
        upper boundaries.
        :rtype: tuple.
        """
        r = self.spheres[:,3]
        close_lower = self.spheres[:,axis] - r < self.cutoff
        close_upper = self.spheres[:,axis] + r > self.L[axis] - self.cutoff

        # Make sure we don't include same particle twice (an unlikely scenario)
        close_upper = close_upper ^ (close_upper & close_lower)

        return self.spheres[close_lower], self.spheres[close_upper]

    def filter_i_loops(self, axis):
        """Return intersection loops which are on the boundary along specified axis.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :return: array containing intersection loop coordinates.
        :rtype: numpy.ndarray.
        """
        i_loops = np.array([
            [*il.c, il.r] for il in self.i_loops
            if il.i0 == axis
        ])
        if len(i_loops):
            i_loops[:,axis] = 0.
        else:
            i_loops = np.empty((0,4), dtype=np.float64)
        return i_loops

    def translate_upper_spheres(self, spheres_upper, axis):
        """Return coordinates of spheres close to upper boundary after applying
        translation across domain and mirroring of coordinates along specified axis.
        :param spheres_upper numpy.ndarray: coordinates of spheres close to upper bound.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :return: array of modified sphere coordinates.
        :rtype: numpy.ndarray.
        """
        spheres_upper[:,axis] -= self.L[axis]
        spheres_upper[:,axis] *= -1
        return spheres_upper

    def duplicate_edge_spheres(self, spheres, axis):
        """
        Creates duplicate spheres corresponding to periodic image particles close to
        the lower boundary along specified axis.
        :param spheres numpy.ndarray: coordinates of spheres.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :return: array of sphere coordinates with duplicates.
        :rtype: numpy.ndarray.
        """
        i1, i2 = (axis+1)%3, (axis+2)%3

        r = spheres[:,3]

        close_i1_lower = spheres[:,i1] - r < self.cutoff
        close_i1_upper = spheres[:,i1] + r > self.L[i1] - self.cutoff

        close_i2_lower = spheres[:,i2] - r < self.cutoff
        close_i2_upper = spheres[:,i2] + r > self.L[i2] - self.cutoff

        s_close_i1_lower = spheres[close_i1_lower]
        s_close_i1_lower[:,i1] += self.L[i1]
        s_close_i1_upper = spheres[close_i1_upper]
        s_close_i1_upper[:,i1] -= self.L[i1]

        s_close_i2_lower = spheres[close_i2_lower]
        s_close_i2_lower[:,i2] += self.L[i2]
        s_close_i2_upper = spheres[close_i2_upper]
        s_close_i2_upper[:,i2] -= self.L[i2]

        s_close_i1l_i2l = spheres[close_i1_lower & close_i2_lower]
        s_close_i1l_i2l[:,i1] += self.L[i1]
        s_close_i1l_i2l[:,i2] += self.L[i2]
        s_close_i1l_i2u = spheres[close_i1_lower & close_i2_upper]
        s_close_i1l_i2u[:,i1] += self.L[i1]
        s_close_i1l_i2u[:,i2] -= self.L[i2]
        s_close_i1u_i2l = spheres[close_i1_upper & close_i2_lower]
        s_close_i1u_i2l[:,i1] -= self.L[i1]
        s_close_i1u_i2l[:,i2] += self.L[i2]
        s_close_i1u_i2u = spheres[close_i1_upper & close_i2_upper]
        s_close_i1u_i2u[:,i1] -= self.L[i1]
        s_close_i1u_i2u[:,i2] -= self.L[i2]

        return np.vstack((
            spheres,
            s_close_i1_lower, s_close_i1_upper,
            s_close_i2_lower, s_close_i2_upper,
            s_close_i1l_i2l, s_close_i1l_i2u, s_close_i1u_i2l, s_close_i1u_i2u,
        ))

    def area_constraint_spheres(self, x, y, p_xy, elevation, rad):
        """Return value for area constraint factor at coordinates x, y based
        on particle positions.
        :param x float: x coordinate of area constraint grid cell.
        :param y float: y coordinate of area constraint grid cell.
        :param p_xy numpy.ndarray: projected particle coordinates in 2D.
        :param elevation numpy.ndarray: min distances of particles from boundary.
        :param rad numpy.ndarray: particle radii.
        :return: area constraint factor.
        :rtype: float.
        """
        delta = npl.norm(p_xy - np.array([x, y]), axis=1)
        nbrs = np.where(delta < rad)[0]
        if not len(nbrs):
            growth = GROWTH_LIMIT
        else:
            delta, elevation, rad = delta[nbrs], elevation[nbrs], rad[nbrs]
            g_min_part = 1. + (elevation / self.cutoff)**2. * (GROWTH_LIMIT - 1.)
            decay = (delta / rad)**2.5
            growth = np.min(g_min_part * (1. - decay) + GROWTH_LIMIT * decay)
        return growth * self.dA

    def area_constraint_i_loops(self, x, y, p_xy, rad):
        """Return value for area constraint factor at coordinates x, y based
        on intersection loop positions.
        :param x float: x coordinate of area constraint grid cell.
        :param y float: y coordinate of area constraint grid cell.
        :param p_xy numpy.ndarray: intersection loop coordinates in 2D.
        :param rad numpy.ndarray: intersection loop radii.
        :return: area constraint factor.
        :rtype: float.
        """
        delta = npl.norm(p_xy - np.array([x, y]), axis=1) - rad
        nbrs = np.where((delta > 0.) & (delta < self.cutoff))[0]
        if not len(nbrs):
            growth = GROWTH_LIMIT
        else:
            delta = delta[nbrs]
            decay = (delta / self.cutoff)**2.5
            growth = np.min((1. - decay) + GROWTH_LIMIT * decay)
        return growth * self.dA

    def constraint_grid(self, axis):
        """Construct grid of area constraint factors for boundary along specified axis.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :return: 2D array of area constraint factors.
        :rtype: numpy.ndarray.
        """
        width = self.cell_width * self.ds
        Lx, Ly = self.L[(axis+1)%3], self.L[(axis+2)%3]
        nx, ny = int(Lx / width), int(Ly / width)  # number of cells in the grid
        dx, dy = Lx / nx, Ly / ny
        self.inv_dx[axis], self.inv_dy[axis] = 1. / dx, 1. / dy

        x = np.arange(0.5 * dx, Lx, dx)
        y = np.arange(0.5 * dy, Ly, dy)
        # TODO : Improve variable naming here.

        sphere_constraints = [[GROWTH_LIMIT * self.dA for _x in x] for _y in y]

        if len(self.spheres[axis]):
            rad = self.spheres[axis][:,3]
            elevation = self.spheres[axis][:,axis] - rad
            elevation = np.where(elevation < 0., 0., elevation)
            p_xy = self.spheres[axis][:,((axis+1)%3,(axis+2)%3)]

            sphere_constraints = [
                [
                    self.area_constraint_spheres(_x, _y, p_xy, elevation, rad)
                    for _x in x
                ]
                for _y in y
            ]

        if len(self.i_loops[axis]):
            rad = self.i_loops[axis][:,3]
            il_xy = self.i_loops[axis][:,((axis+1)%3,(axis+2)%3)]

            i_loop_constraints = [
                [self.area_constraint_i_loops(_x, _y, il_xy, rad) for _x in x]
                for _y in y
            ]

            return np.minimum(sphere_constraints, i_loop_constraints)

        return sphere_constraints

    def build_area_constraint_grids(self):
        """Handles construction of area constraint grids for boundaries along each axis.
        """
        if len(self.spheres):
            p_ax = [
                self.filter_spheres(axis) for axis in range(3)
            ]
            p_ax = [
                np.vstack((p[0], self.translate_upper_spheres(p[1], axis)))
                for axis, p in enumerate(p_ax)
            ]
            p_ax = [
                self.duplicate_edge_spheres(p, axis)
                for axis, p in enumerate(p_ax)
            ]
            self.spheres = p_ax
        else:
            self.spheres = [[] for _ in range(3)]

        if len(self.i_loops):
            il_ax = [
                self.filter_i_loops(axis) for axis in range(3)
            ]
            il_ax = [
                self.duplicate_edge_spheres(il, axis)
                for axis, il in enumerate(il_ax)
            ]
            self.i_loops = il_ax
        else:
            self.i_loops = [[] for _ in range(3)]

        self.grid = [self.constraint_grid(axis) for axis in range(3)]
