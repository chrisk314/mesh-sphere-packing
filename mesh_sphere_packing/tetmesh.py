
from contextlib import contextmanager

import numpy as np
from numpy import linalg as npl
from meshpy import tet

from mesh_sphere_packing import logger


@contextmanager
def redirect_tetgen_output(fname='./tet.log'):
    import ctypes, io, os, sys

    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

    def _redirect_stdout(to_fd):
        libc.fflush(c_stdout)
        sys.stdout.close()
        os.dup2(to_fd, original_stdout_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    def extract_stats(f):
        while True:
            l = f.readline()
            if 'Statistics:' in l.decode('ascii'):
                stats = [f.readline().decode('ascii') for i in range(11)]
                npoints, ntets, nfaces, nedges = [
                    int(sl.split()[-1]) for sl in stats[7:]
                ]
                return 'Built mesh with {} points, {} tetrahedra, {} faces, and {} edges'\
                    .format(npoints, ntets, nfaces, nedges)

    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        tfile = open(fname, mode='w+b')
        _redirect_stdout(tfile.fileno())
        yield
        _redirect_stdout(saved_stdout_fd)
        tfile.seek(0, io.SEEK_SET)
        logger.info(extract_stats(tfile))
        tfile.close()
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


def write_poly(fname, mesh):
    points, faces, markers, holes = list(mesh.points), list(mesh.faces),\
        list(mesh.face_markers), list(mesh.holes)
    with open(fname, 'w') as f:
        f.write('%d 3 0 1\n' % len(points))
        for i, p in enumerate(points):
            f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, p[0], p[1], p[2]))
        f.write('%d 1\n' % len(faces))
        for i, (fac, m) in enumerate(zip(faces, markers)):
            f.write('1 0 %d\n%d %d %d %d\n' % (m, 3, fac[0], fac[1], fac[2]))
        if len(holes):
            f.write('%d\n' % len(holes))
            for i, h in enumerate(holes):
                f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, h[0], h[1], h[2]))
        else:
            f.write('0\n')


def write_multiflow(fname, mesh):
    from collections import defaultdict
    import h5py as h5

    points, elements, faces, markers, neighbours, adjacent_elements =\
        np.array(mesh.points), np.array(mesh.elements), np.array(mesh.faces),\
        np.array(mesh.face_markers), np.array(mesh.neighbors),\
        np.array(mesh.adjacent_elements)

    with h5.File(fname, 'w') as f:
        # Write node data.
        f['nodes'] = points.flatten()

        # Write cell data.
        cell_type = np.full((len(elements),1), 6)
        f['cells'] = np.hstack((cell_type, elements)).flatten()
        f['cellNodePtr'] = np.append([0], np.full(len(elements), 4).cumsum())

        # Write face data
        f['faces'] = faces.flatten()
        f['faceNodePtr'] = np.append([0], np.full(len(faces), 3).cumsum())

        # Write cell face connectivity data
        cell_faces = defaultdict(list)
        for idx, adj in enumerate(adjacent_elements):
            cell_faces[adj[0]].append(idx)
            cell_faces[adj[1]].append(idx)
        cell_faces.pop(-1)
        f['cellFaces'] = np.array(
            [item[1] for item in sorted(cell_faces.items(), key=lambda x: x[0])]
        ).flatten()

        f['cellFacePtr'] = np.append([0], np.full(len(elements), 4).cumsum())

        # Write cell neighbour data
        cell_nbr = neighbours[neighbours > -1]
        f['cellNeighbours'] = cell_nbr

        cell_nbr_ptr = np.empty(len(elements)+1, dtype=np.int32)
        cell_nbr_ptr[0] = 0
        cell_nbr_ptr[1:] = np.apply_along_axis(
            lambda x: np.where(x > -1)[0].shape[0], 1, neighbours
        ).cumsum()
        f['cellNeighbourPtr'] = cell_nbr_ptr

        # Write boundary marker data
        f['boundaryType'] = markers

        # Write header data.
        f['meshData'] = np.array([
            len(points),
            len(elements),
            len(faces),
            8,
            4 * len(elements),
            3 * len(faces),
            4 * len(elements),
            cell_nbr_ptr[-1]
        ])


def duplicate_lower_boundaries(lower_boundaries, L):
    upper_boundaries = []
    for i, (points, tris, holes) in enumerate(lower_boundaries):
        translate = np.array([[L[j] if j==i else 0. for j in range(3)]])
        points_upper = points.copy() + translate
        tris_upper = tris.copy()
        holes_upper = holes.copy() + translate
        upper_boundaries.append((points_upper, tris_upper, holes_upper))
    return lower_boundaries + upper_boundaries


def build_point_list(sphere_pieces, boundaries):
    vcount = 0
    all_points = []
    for points, tris in [(p.points, p.tris) for p in sphere_pieces]:
        all_points.append(points)
        tris += vcount
        vcount += points.shape[0]
    for points, tris, _ in boundaries:
        all_points.append(points)
        tris += vcount
        vcount += points.shape[0]
    return np.vstack(all_points)


def build_facet_list(sphere_pieces, boundaries):
    all_facets = [tris for _, tris, _ in boundaries]
    all_markers = [
        np.full(len(all_facets[0]), 1), np.full(len(all_facets[1]), 3),
        np.full(len(all_facets[2]), 5), np.full(len(all_facets[3]), 2),
        np.full(len(all_facets[4]), 4), np.full(len(all_facets[5]), 6),
    ]
    mark_offset = 7
    for p in sphere_pieces:
        all_facets.append(p.tris)
        all_markers.append(np.full(len(p.tris), p.sphere.id + mark_offset))
    return np.vstack(all_facets), np.hstack(all_markers)


def build_hole_list(sphere_pieces):
    # TODO : Ultimately each sphere segment will contain hole data
    all_holes = []
    for points in [p.points for p in sphere_pieces]:
        all_holes.append(0.5 * (points.max(axis=0) + points.min(axis=0)))
    return np.vstack(all_holes)


def build_tetmesh(domain, sphere_pieces, boundaries, config):
    logger.info('Building tetrahedral mesh')

    boundaries = duplicate_lower_boundaries(boundaries, domain.L)

    points = build_point_list(sphere_pieces, boundaries)

    # Fix boundary points to exactly zero
    for i in range(3):
        points[(np.isclose(points[:,i], 0.), i)] = 0.

    facets, markers = build_facet_list(sphere_pieces, boundaries)
    holes = build_hole_list(sphere_pieces)

    rad_edge = config.tetgen_rad_edge_ratio
    min_angle = config.tetgen_min_angle
    max_volume = config.tetgen_max_volume

    options = tet.Options('pq{}/{}nzfennYCV'.format(rad_edge, min_angle))
    options.quiet = False

    mesh = tet.MeshInfo()
    mesh.set_points(points)
    mesh.set_facets(facets.tolist(), markers=markers.tolist())
    mesh.set_holes(holes)

    with redirect_tetgen_output():
        return tet.build(
            mesh, options=options, verbose=True, max_volume=max_volume
        )
