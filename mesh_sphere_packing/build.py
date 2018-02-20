#!/usr/bin/env python

from mesh_sphere_packing import logger
from mesh_sphere_packing.parse import get_parser, load_data
from mesh_sphere_packing.splitsphere import splitsphere
from mesh_sphere_packing.boundarypslg import boundarypslg
from mesh_sphere_packing.tetmesh import build_tetmesh


def output_mesh(mesh, config):
    if not config.output_format:
        return
    logger.info('Outputting mesh in formats: {}'
        .format(', '.join(config.output_format))
    )
    if 'msh' in config.output_format:
        from mesh_sphere_packing.tetmesh import write_msh
        write_msh('%s.msh' % config.output_prefix, mesh)
    if 'multiflow' in config.output_format:
        from mesh_sphere_packing.tetmesh import write_multiflow
        write_multiflow('%s.h5' % config.output_prefix, mesh)
    if 'ply' in config.output_format:
        from mesh_sphere_packing.tetmesh import write_ply
        write_ply('%s.ply' % config.output_prefix, mesh)
    if 'poly' in config.output_format:
        from mesh_sphere_packing.tetmesh import write_poly
        write_poly('%s.poly' % config.output_prefix, mesh)
    if 'vtk' in config.output_format:
        mesh.write_vtk('%s.vtk' % config.output_prefix)


def build(domain, particles, config):
    logger.info('Starting mesh build process')

    sphere_pieces = splitsphere(domain, particles, config)
    boundaries = boundarypslg(domain, particles, sphere_pieces, config)
    mesh = build_tetmesh(domain, sphere_pieces, boundaries, config)
    output_mesh(mesh, config)

    logger.info('Completed')


if __name__ == '__main__':
    args = get_parser().parse_args()
    data = load_data(args)
    build(*data)
