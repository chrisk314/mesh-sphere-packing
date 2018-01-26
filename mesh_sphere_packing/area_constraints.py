
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


def build_area_constraint_grid(args):
    # TODO : Change this to use particle data read from file. For now mocking
    #      : up a single particle from the command line args
    particles = np.array([args.particle_center + [args.particle_radius]])

    # TODO : cutoff distance is dependent on mesh resolution and should exist
    #      : as part of state in some as yet to be implemented class. Same applies
    #      : to the domain dimensions.
    cutoff = 0.5
    L = args.domain_dimensions

    p_ax = [
        filter_particles(particles, cutoff, axis, L[axis])
        for axis in range(3)
    ]
    p_ax = [
        np.vstack((p[0], translate_upper_particles(p[1], axis, L[axis])))
        for axis, p in enumerate(p_ax)
    ]

    area_constraints = []
    # TODO : populate the area constraint grid with interpolated values of
    #      : some sizing function, which depends on the particle positions,
    #      : f(cx,cy) -> R, where cx and cy are the triangle center coordinates.
    return area_constraints
