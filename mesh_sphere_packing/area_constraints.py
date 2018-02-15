
import numpy as np
from numpy import linalg as npl

from mesh_sphere_packing import ONE_THIRD, GROWTH_LIMIT


class AreaConstraints(object):

    """Constructs grid of area constraints for triangulation of domain boundaries.
    """

    cutoff_factor = 5.  # Cutoff distance in units of characteristic length, ds
    cell_width = 1.     # Width of grid cells in units of characteristic length, ds

    def __init__(self, domain, particles, ds):
        self.ds = ds
        self.dA = 0.5 * ds**2
        self.dA_max = self.dA * GROWTH_LIMIT
        self.cutoff = self.cutoff_factor * self.ds
        self.L = domain.L
        self.particles = particles[:,1:]
        self.inv_dx, self.inv_dy = 3 * [None], 3 * [None]
        self.build_area_constraint_grid()

    def filter_particles(self, particles, axis):
        """Return particles which are close to the boundary along specified
        axis without actually crossing it.
        """
        r = particles[:,3]
        close_lower = particles[:,axis] - r < self.cutoff
        close_upper = particles[:,axis] + r > self.L[axis] - self.cutoff

        # Make sure we don't include same particle twice (an unlikely scenario)
        close_upper = close_upper ^ (close_upper & close_lower)

        out_lower = particles[:,axis] < 0.
        out_upper = particles[:,axis] > self.L[axis]

        return (
            particles[close_lower & ~out_lower],
            particles[close_upper & ~out_upper]
        )

    def translate_upper_particles(self, particles_upper, axis):
        """Return coordinates of particles close to upper boundary after applying
        translation across domain and mirroring of coordinates along specified axis.
        """
        particles_upper[:,axis] -= self.L[axis]
        particles_upper[:,axis] *= -1
        return particles_upper

    def duplicate_edge_particles(self, particles, axis):
        i1, i2 = (axis+1)%3, (axis+2)%3

        r = particles[:,3]

        close_i1_lower = particles[:,i1] - r < self.cutoff
        close_i1_upper = particles[:,i1] + r > self.L[i1] - self.cutoff

        close_i2_lower = particles[:,i2] - r < self.cutoff
        close_i2_upper = particles[:,i2] + r > self.L[i2] - self.cutoff

        p_close_i1_lower = particles[close_i1_lower]
        p_close_i1_lower[:,i1] += self.L[i1]
        p_close_i1_upper = particles[close_i1_upper]
        p_close_i1_upper[:,i1] -= self.L[i1]

        p_close_i2_lower = particles[close_i2_lower]
        p_close_i2_lower[:,i2] += self.L[i2]
        p_close_i2_upper = particles[close_i2_upper]
        p_close_i2_upper[:,i2] -= self.L[i2]

        p_close_i1l_i2l = particles[close_i1_lower & close_i2_lower]
        p_close_i1l_i2l[:,i1] += self.L[i1]
        p_close_i1l_i2l[:,i2] += self.L[i2]
        p_close_i1l_i2u = particles[close_i1_lower & close_i2_upper]
        p_close_i1l_i2u[:,i1] += self.L[i1]
        p_close_i1l_i2u[:,i2] -= self.L[i2]
        p_close_i1u_i2l = particles[close_i1_upper & close_i2_lower]
        p_close_i1u_i2l[:,i1] -= self.L[i1]
        p_close_i1u_i2l[:,i2] += self.L[i2]
        p_close_i1u_i2u = particles[close_i1_upper & close_i2_upper]
        p_close_i1u_i2u[:,i1] -= self.L[i1]
        p_close_i1u_i2u[:,i2] -= self.L[i2]

        return np.vstack((
            particles,
            p_close_i1_lower, p_close_i1_upper,
            p_close_i2_lower, p_close_i2_upper,
            p_close_i1l_i2l, p_close_i1l_i2u, p_close_i1u_i2l, p_close_i1u_i2u,
        ))

    def area_constraint(self, x, y, p_xy, elevation, rad):
        """Return value for area constraint factor at coordinates x, y based
        on particle positions.
        """
        # TODO : populate the area constraint grid with interpolated values of
        #      : some sizing function, which depends on the particle positions,
        #      : f(cx,cy) -> R, where cx and cy are the triangle center coordinates.

        delta = npl.norm(p_xy - np.array([x, y]), axis=1)
        nbrs = np.where(delta < rad)[0]
        if not len(nbrs):
            growth = GROWTH_LIMIT
        else:
            # TODO : This requires some tuning to get the desired refinement.
            g_min_part = 1. + (elevation / self.cutoff)**2. * (GROWTH_LIMIT - 1.)
            decay = (delta / rad)**2.5
            growth = np.min(g_min_part * (1. - decay) + GROWTH_LIMIT * decay)
        return growth * self.dA

    def constraint_grid(self, axis):
        width = self.cell_width * self.ds
        Lx, Ly = self.L[(axis+1)%3], self.L[(axis+2)%3]
        nx, ny = int(Lx / width), int(Ly / width)  # number of cells in the grid
        dx, dy = Lx / nx, Ly / ny
        self.inv_dx[axis], self.inv_dy[axis] = 1. / dx, 1. / dy

        x = np.arange(0.5 * dx, Lx, dx)
        y = np.arange(0.5 * dy, Ly, dy)
        # TODO : Improve variable naming here.

        rad = self.particles[axis][:,3]
        elevation = self.particles[axis][:,axis] - rad
        elevation = np.where(elevation < 0., 0., elevation)
        p_xy = self.particles[axis][:,((axis+1)%3,(axis+2)%3)]

        return [
            [self.area_constraint(_x, _y, p_xy, elevation, rad) for _x in x]
            for _y in y
        ]

    def build_area_constraint_grid(self):
        # TODO : Change this to use particle data read from file. For now mocking
        #      : up a single particle from the command line args
        p_ax = [
            self.filter_particles(self.particles, axis)
            for axis in range(3)
        ]
        p_ax = [
            np.vstack((p[0], self.translate_upper_particles(p[1], axis)))
            for axis, p in enumerate(p_ax)
        ]
        p_ax = [
            self.duplicate_edge_particles(p, axis)
            for axis, p in enumerate(p_ax)
        ]
        self.particles = p_ax
        self.grid = [self.constraint_grid(axis) for axis in range(3)]
