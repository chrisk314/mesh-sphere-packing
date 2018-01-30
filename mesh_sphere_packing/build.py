
from mesh_sphere_packing.parser import parser
from mesh_sphere_packing.reader import read_config_file, read_particle_file
from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh


def build(args):
    config = read_config_file(args.config_file)
    particle_file = args.particle_file or config.particle_file
    if particle_file:
        L, PBC, particles = read_particle_file(particle_file)
    segments = splitsphere(args)
    boundaries = boundarypslg(segments, args)
    mesh = build_tetmesh(segments, boundaries, args)
    mesh.write_vtk('mesh.vtk')


if __name__ == '__main__':
    args = parser.parse_args()
    build(args)
