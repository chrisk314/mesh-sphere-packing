
import numpy as np

from mesh_sphere_packing import ONE_THIRD, GROWTH_LIMIT


class AreaConstraints(object):

    """Constructs grid of area constraints for triangulation of domain boundaries.
    """

    cutoff = 0.5        # Distance beyond which particles do not force added points
    cell_width = 2.     # Width of grid cells in units of characteristic length, ds

    def __init__(self, args, ds):
        # TODO : cutoff distance is dependent on mesh resolution and should exist
        #      : as part of state in some as yet to be implemented class. Same applies
        #      : to the domain dimensions.
        self.ds = ds
        self.dA = ds**2
        self.L = np.array(args.domain_dimensions)
        self.inv_dx, self.inv_dy = 3 * [None], 3 * [None]
        self.build_area_constraint_grid(args)

    def filter_particles(self, particles, axis):
        """Return particles which are close to the boundary along specified
        axis without actually crossing it.
        """
        close_lower = particles[:,axis] - particles[:,3] < self.cutoff
        close_upper = particles[:,axis] - particles[:,3] > self.L[axis] - self.cutoff

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
        # TODO : Need to duplicate particles which cross a boundary perpendicular
        #      : to axis as they will have geometry at both sides of domain.
        return particles

    def area_constraint(self, x, y):
        """Return value for area constraint factor at coordinates x, y based
        on particle positions.
        """
        # TODO : populate the area constraint grid with interpolated values of
        #      : some sizing function, which depends on the particle positions,
        #      : f(cx,cy) -> R, where cx and cy are the triangle center coordinates.
        growth = GROWTH_LIMIT
        return self.dA * growth

    def constraint_grid(self, particles, axis):
        width = self.cell_width * self.ds
        Lx, Ly = self.L[(axis+1)%3], self.L[(axis+2)%3]
        nx, ny = int(Lx / width), int(Ly / width)  # number of cells in the grid
        dx, dy = Lx / nx, Ly / ny
        self.inv_dx[axis], self.inv_dy[axis] = 1. / dx, 1. / dy

        x = np.arange(0.5 * dx, Lx, dx)
        y = np.arange(0.5 * dy, Ly, dy)
        # TODO : Improve variable naming here.
        return [[self.area_constraint(_x, _y) for _x in x] for _y in y]

    def build_area_constraint_grid(self, args):
        # TODO : Change this to use particle data read from file. For now mocking
        #      : up a single particle from the command line args
        particles = np.array([args.particle_center + [args.particle_radius]])

        p_ax = [
            self.filter_particles(particles, axis)
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
        self.grid = [
            self.constraint_grid(p, axis) for axis, p in enumerate(p_ax)
        ]
