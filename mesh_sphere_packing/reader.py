
import numpy as np


class ParticleFileReaderException(Exception):
    pass


def read_particle_file(pfile):
    """Loads data from an ascii encoded text file specifying domain dimensions,
    periodic boundary flags, and particle center coordinates and radii in the
    following format:

        line 1      : <domain_extent_x> <domain_extent_y> <domain_extent_z>
        line 2      : <x_bound_periodic> <y_bound_periodic> <z_bound_periodic>
        line 3      : <p1_id> <p1_x> <p1_y> <p1_z> <p1_radius>
        ...
        line N+2    : <pN_id> <pN_x> <pN_y> <pN_z> <pN_radius>

    Here is an example particle data file:

        $ cat my_particle_file.txt
        2.0e-03 2.0e-3 2.0e-3
        0 1 1
        0 9.10502e-4 7.73356e-4 1.66188e-3 5.08645e-4
        1 3.83301e-4 9.34604e-4 6.5982se-6 5.50847e-4
        ...
        N 1.51262e-3 2.95234e-4 5.63176e-4 2.08111e-4

    Particles need not be specified in order of id or any other order. The ids are
    used to specify boundary conditions on the particle surfaces.
    :param fname: string specifying path of particle data file.
    """
    with pfile as f:
        try:
            L = np.array(
                [float(tok) for tok in f.readline().strip().split()[:3]]
            )
        except Exception as e:
            raise ParticleFileReaderException(
                'Domain extents in particle data file invalid.'
            ) from e
        try:
            PBC = np.array(
                [bool(tok) for tok in f.readline().strip().split()[:3]]
            )
        except Exception as e:
            raise ParticleFileReaderException(
                'PBC flags in particle data file invalid.'
            ) from e
        try:
            particles = np.loadtxt(f, dtype=np.float64)
        except Exception as e:
            raise ParticleFileReaderException(
                'Could not read particle data in particle data file.'
            ) from e
        try:
            assert particles.shape[0] > 0
        except AssertionError as e:
            raise ParticleFileReaderException(
                'No particles specified in particle data file.'
            ) from e
        try:
            assert particles.shape[1] == 5
        except AssertionError as e:
            raise ParticleFileReaderException(
                'Incorrect number of particle attributes in particle data file.'
            ) from e
        try:
            assert not np.any(particles[:,1:] < 0.)
        except AssertionError as e:
            raise ParticleFileReaderException(
                'Invalid values for particle data in particle data file.'
            ) from e
    return L, PBC, particles

