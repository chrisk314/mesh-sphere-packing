
import numpy as np
from numpy import linalg as npl
from meshpy import tet


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
        np.full(len(all_facets[0]), 1), np.full(len(all_facets[3]), 2),
        np.full(len(all_facets[1]), 3), np.full(len(all_facets[4]), 4),
        np.full(len(all_facets[2]), 5), np.full(len(all_facets[5]), 6),
    ]
    fcount = 7
    for tris in [p.tris for p in sphere_pieces]:
        all_facets.append(tris)
        all_markers.append(np.full(len(tris), fcount))
        fcount += 1
    return np.vstack(all_facets), np.hstack(all_markers)


def build_hole_list(sphere_pieces):
    # TODO : Ultimately each sphere segment will contain hole data
    all_holes = []
    for points in [p.points for p in sphere_pieces]:
        all_holes.append(0.5 * (points.max(axis=0) + points.min(axis=0)))
    return np.vstack(all_holes)


def write_tetmesh_poly(fname, points, facets, markers, holes):
    with open(fname, 'w') as f:
        f.write('%d 3 0 1\n' % len(points))
        for i, p in enumerate(points):
            f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, p[0], p[1], p[2]))
        f.write('%d 1\n' % len(facets))
        for i, (fac, m) in enumerate(zip(facets, markers)):
            f.write('1 0 %d\n%d %d %d %d\n' % (m, 3, fac[0], fac[1], fac[2]))
        if len(holes):
            f.write('%d\n' % len(holes))
            for i, h in enumerate(holes):
                f.write('%5d %+1.15e %+1.15e %+1.15e\n' % (i, h[0], h[1], h[2]))
        else:
            f.write('0\n')


def build_tetmesh(domain, sphere_pieces, boundaries, config):

    boundaries = duplicate_lower_boundaries(boundaries, domain.L)

    points = build_point_list(sphere_pieces, boundaries)
    # Fix boundary points to exactly zero
    for i in range(3):
        points[(np.isclose(points[:,i], 0.), i)] = 0.

    facets, markers = build_facet_list(sphere_pieces, boundaries)
    holes = build_hole_list(sphere_pieces)

    rad_edge = config.tetgen_rad_edge_ratio
    min_angle = config.tetgen_min_angle
    max_volume = None  # 0.00001

    options = tet.Options('pq{}/{}YCV'.format(rad_edge, min_angle))

    mesh = tet.MeshInfo()
    mesh.set_points(points)
    mesh.set_facets(facets.tolist(), markers=markers.tolist())
    mesh.set_holes(holes)

    write_tetmesh_poly('mesh.poly', points, facets, markers, holes)

    return tet.build(mesh, options=options, max_volume=max_volume)
