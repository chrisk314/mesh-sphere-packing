
import numpy as np
import yaml
from collections import namedtuple


class ParticleFileReaderError(Exception):
    pass


class ConfigFileReaderError(Exception):
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
    :param fname: path of particle data file.
    :returns: tuple consisting of a numpy array of domain dimensions, a numpy
    array of periodic boundary flags, and a numpy array of particle data.
    """
    with pfile as f:
        try:
            L = np.array(
                [float(tok) for tok in f.readline().strip().split()[:3]]
            )
        except Exception as e:
            raise ParticleFileReaderError(
                'Domain extents in particle data file invalid.'
            ) from e
        try:
            PBC = np.array(
                [bool(tok) for tok in f.readline().strip().split()[:3]]
            )
        except Exception as e:
            raise ParticleFileReaderError(
                'PBC flags in particle data file invalid.'
            ) from e
        try:
            particles = np.loadtxt(f, dtype=np.float64)
        except Exception as e:
            raise ParticleFileReaderError(
                'Could not read particle data in particle data file.'
            ) from e
        try:
            assert particles.shape[0] > 0
        except AssertionError as e:
            raise ParticleFileReaderError(
                'No particles specified in particle data file.'
            ) from e
        try:
            assert particles.shape[1] == 5
        except AssertionError as e:
            raise ParticleFileReaderError(
                'Incorrect number of particle attributes in particle data file.'
            ) from e
        try:
            assert not np.any(particles[:,1:] < 0.)
        except AssertionError as e:
            raise ParticleFileReaderError(
                'Invalid values for particle data in particle data file.'
            ) from e
    return L, PBC, particles


def read_config_file(cfile):
    """Loads optional user configuration from a .yaml file.
    :param cfile: path of yaml config file.
    :returns: namedtuple 'Config' object containing config.
    """
    # TODO : Describe available configurable options.
    config = {
        'particle_file': None,
        'allow_overlaps': False,  # Overlaps not currently supported
        'tetgen_rad_edge_ratio': 1.4,
        'tetgen_min_angle': 18.,
        'tetgen_max_volume': 1.0e-05,
        'surf_mesh_factor': 1.0e-01,
    }
    if cfile:
        with cfile as f:
            try:
                user_config = yaml.load(f.read())
            except yaml.YAMLError as e:
                raise ConfigFileReaderError('Config file invalid.') from e
        unsupported = list(set(user_config.keys()) - set(config.keys()))
        if unsupported:
            import warnings
            for k in unsupported:
                msg = 'Ignored unsupported configuration options: {}'.format(k)
                warnings.warn(msg)
                user_config.pop(k)
        # TODO : Should check user options are valid and apply argument type
        #      : conversions.
        config.update(user_config)
        config['allow_overlaps'] = bool(config['allow_overlaps'])
    return namedtuple('Config', config.keys())(**config)
