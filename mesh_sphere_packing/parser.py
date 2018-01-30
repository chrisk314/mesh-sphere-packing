
import argparse

DESC = """Tetrahedral mesh generator to produce meshes of the interstitial
spaces in packings of spheres for finite volume simulations.
"""

parser = argparse.ArgumentParser()
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
    help='vector in R^3 specifying particle center coordinates <x y z>',
)
parser.add_argument(
    '-r', '--particle-radius', type=float, required=False,
    help='particle radius <r>',
)
parser.add_argument(
    '-l', '--domain-dimensions', nargs=3, type=float, required=False,
    help='vector in R^3 specifying domain side lengths <Lx Ly Lz>',
)
