
import argparse
from argparse import FileType, RawTextHelpFormatter
from collections import namedtuple

import numpy as np
import yaml

from mesh_sphere_packing.splitsphere import Domain, duplicate_particles,\
    extend_domain

DESC = """Generates tetrahedral meshes of the interstitial spaces in packings of spheres
for finite volume simulations.

-p, --particle-file option expects an ascii encoded text file specifying domain
dimensions, periodic boundary flags, and particle center coordinates and radii
in the following format:

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

"""


class DataLoaderError(Exception):
    pass


class ParticleFileReaderError(Exception):
    pass


class ConfigFileReaderError(Exception):
    pass


def get_parser():
    parser = argparse.ArgumentParser(
        description=DESC, formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        '-V', '--version', action='version', version='0.1'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='output detailed information'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true', help='suppress all output'
    )
    parser.add_argument(
        '-cx', '--particle-center', nargs=3, type=float, required=False,
        metavar='X', help='list of particle center coordinates',
    )
    parser.add_argument(
        '-r', '--particle-radius', type=float, required=False,
        metavar='R', help='particle radius',
    )
    parser.add_argument(
        '-l', '--domain-dimensions', nargs=3, type=float, required=False,
        metavar='L', help='list of domain side lengths',
    )
    parser.add_argument(
        '--pbc', nargs=3, type=int, required=False,
        metavar='0|1', help='list of periodic boundary flags',
    )
    parser.add_argument(
        '-p', '--particle-file', type=FileType(mode='r'), required=False, metavar='FILE',
        help='path to file specifying particle data (see main description)'
    )
    parser.add_argument(
        '-c', '--config-file', type=FileType(mode='r'), required=False, metavar='FILE',
        help='path to yaml config file'
    )
    parser.set_defaults(func=load_data)
    return parser


def load_data(args):
    config = read_config_file(args.config_file)
    particle_file = args.particle_file or config.particle_file
    if particle_file:
        if isinstance(particle_file, str):
            particle_file = open(particle_file, 'r')
        try:
            L, PBC, particles = read_particle_file(particle_file)
        finally:
            particle_file.close()
    else:
        single_mode_missing_required = [
            not args.particle_center,
            not args.particle_radius,
            not args.domain_dimensions,
            not args.pbc,
        ]
        if any(single_mode_missing_required):
            raise DataLoaderError('Insufficient data provided to build mesh.')
        L = args.domain_dimensions
        PBC = args.pbc
        particles = np.array([
            [0] + args.particle_center + [args.particle_radius]
        ])
    particles = duplicate_particles(L, particles, config)
    L, particles = extend_domain(L, PBC, particles, config.segment_length)
    domain = Domain(L, PBC)
    return domain, particles, config


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

    def _readline(f):
        """Return next uncommented stripped line from open file ``f``."""
        while True:
            l = next(f).strip()
            if not l.startswith('#'):
                return l

    with pfile as f:
        try:
            L = np.array(
                [float(tok) for tok in _readline(f).split()[:3]]
            )
        except Exception as e:
            raise ParticleFileReaderError(
                'Domain extents in particle data file invalid.'
            ) from e
        try:
            PBC = np.array(
                [bool(int(tok)) for tok in _readline(f).split()[:3]]
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
        if particles.ndim == 1:
            particles = particles[np.newaxis,:]
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
        'segment_length': 1.0e-04,
        'output_format': ['vtk', 'poly', 'off'],
        'duplicate_particles': [False, False, False]
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
        try:
            assert sum(config['duplicate_particles']) <= 1
        except AssertionError as e:
            raise ConfigFileReaderError(
                'Only one axis can be specified for particle duplication'
            ) from e
        config['allow_overlaps'] = bool(config['allow_overlaps'])
    return namedtuple('Config', config.keys())(**config)
