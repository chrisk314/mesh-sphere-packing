
from mesh_sphere_packing.parse import get_parser, load_data
from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh


def build(args):
    segments = splitsphere(args)
    boundaries = boundarypslg(segments, args)
    mesh = build_tetmesh(segments, boundaries, args)
    mesh.write_vtk('mesh.vtk')


if __name__ == '__main__':
    args = get_parser().parse_args()
    data = load_data(args)
    build(args)
