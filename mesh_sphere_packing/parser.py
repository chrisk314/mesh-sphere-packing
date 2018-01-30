
import argparse
from argparse import FileType, RawTextHelpFormatter

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
    '-p', '--particle-file', type=FileType(mode='r'), required=False, metavar='FILE',
    help='path to file specifying particle data (see main description)'
)
parser.add_argument(
    '-c', '--config-file', type=FileType(mode='r'), required=False, metavar='FILE',
    help='path to yaml config file'
)
