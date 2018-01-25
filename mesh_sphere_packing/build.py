
import argparse

from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh


def build(args):
    segments = splitsphere(args)
    boundaries = boundarypslg(segments, args)
    mesh = build_tetmesh(segments, boundaries, args)
    mesh.write_vtk('mesh.vtk')


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
    '-cx', '--particle-center', nargs=3, type=float, required=True,
    help='vector in R^3 specifying particle center coordinates <x y z>',
)
parser.add_argument(
    '-r', '--particle-radius', type=float, required=True,
    help='particle radius <r>',
)
parser.add_argument(
    '-l', '--domain-dimensions', nargs=3, type=float, required=True,
    help='vector in R^3 specifying domain side lengths <Lx Ly Lz>',
)
parser.set_defaults(func=build)


if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)
