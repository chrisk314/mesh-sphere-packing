
import sys

from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh


def build(args):
    x, y, z, r, Lx, Ly, Lz = args
    segments = splitsphere(args)
    boundaries = boundarypslg(segments, Lx, Ly, Lz)
    return build_tetmesh(segments, boundaries, Lx, Ly, Lz)


def get_args(argv):
    # TODO : Replace with argparse.
    """Get command line arguments
    :return: sphere center coordinates, x, y, z, sphere radius, r,
    domain box side lengths, Lx, Ly, Lz.
    """
    try:
        return float(argv[1]), float(argv[2]), float(argv[3]), float(argv[4]),\
                float(argv[5]), float(argv[6]), float(argv[7])
    except IndexError:
        raise UserWarning('Must specify x0 y0 z0 r Lx Ly Lz')
    except ValueError:
        raise UserWarning('Invalid arguments')


if __name__ == '__main__':
    args = get_args(sys.argv)
    mesh = build(args)
    mesh.write_vtk('mesh.vtk')
