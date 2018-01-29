
import numpy as np


def filter_particles(particles, cutoff, axis, L):
    """Return particles which are close to the boundary along specified
    axis without actually crossing it.
    """
    close_lower = particles[:,axis] - particles[:,3] < cutoff
    close_upper = particles[:,axis] - particles[:,3] > L - cutoff

    # Make sure we don't include same particle twice (an unlikely scenario)
    close_upper = close_upper ^ (close_upper & close_lower)

    out_lower = particles[:,axis] < 0.
    out_upper = particles[:,axis] > L
    return particles[close_lower & ~out_lower], particles[close_upper & ~out_upper]


def translate_upper_particles(particles_upper, axis, L):
    """Return coordinates of particles close to upper boundary after applying
    translation across domain and mirroring of coordinates along specified axis.
    """
    particles_upper[:,axis] -= L
    particles_upper[:,axis] *= -1
    return particles_upper


def area_constraint(x, y):
    """Return value for area constraint factor at coordinates x, y based
    on particle positions.
    """
    # TODO : implement this.
    return 0.01


def constraint_grid(particles, axis, L, ds):
    # TODO : Replace this magic number
    s = 2. * ds
    Lx, Ly = L[(axis+1)%3], L[(axis+2)%3]
    nx, ny = int(Lx / s), int(Ly / s)  # number of cells in the grid
    dx, dy = Lx / nx, Ly / ny

    x = np.arange(0.5 * dx, Lx, dx)
    y = np.arange(0.5 * dy, Ly, dy)
    # TODO : Improve variable naming here.
    return [[area_constraint(_x, _y) for _x in x] for _y in y]

    # xy_grid = np.array(np.meshgrid(x, y)).T
    # area_constraints = np.array([
        # area_constraint(xy) for xy in xy_grid
    # ]).reshape((nx, ny))
    # return area_constraints.tolist()


def build_area_constraint_grid(args, ds):
    # TODO : Change this to use particle data read from file. For now mocking
    #      : up a single particle from the command line args
    particles = np.array([args.particle_center + [args.particle_radius]])

    # TODO : cutoff distance is dependent on mesh resolution and should exist
    #      : as part of state in some as yet to be implemented class. Same applies
    #      : to the domain dimensions.
    cutoff = 0.5
    L = np.array(args.domain_dimensions)

    p_ax = [
        filter_particles(particles, cutoff, axis, L[axis])
        for axis in range(3)
    ]
    p_ax = [
        np.vstack((p[0], translate_upper_particles(p[1], axis, L[axis])))
        for axis, p in enumerate(p_ax)
    ]


    area_constraints = [
        constraint_grid(p, axis, L, ds) for axis, p in enumerate(p_ax)
    ]
    # TODO : populate the area constraint grid with interpolated values of
    #      : some sizing function, which depends on the particle positions,
    #      : f(cx,cy) -> R, where cx and cy are the triangle center coordinates.
    return area_constraints
