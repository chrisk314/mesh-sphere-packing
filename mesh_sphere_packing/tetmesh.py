
import numpy as np
from numpy import linalg as npl
from meshpy import tet


def build_tetmesh(segments, boundaries, Lx, Ly, Lz):

    def duplicate_lower_boundaries(lower_boundaries, Lx, Ly, Lz):
        L = np.array([Lx, Ly, Lz])
        upper_boundaries = []
        for i, (points, tris, holes) in enumerate(lower_boundaries):
            translate = np.array([[L[j] if j==i else 0. for j in range(3)]])
            points_upper = points.copy() + translate
            tris_upper = tris.copy()
            holes_upper = holes.copy() + translate
            upper_boundaries.append((points_upper, tris_upper, holes_upper))
        return lower_boundaries + upper_boundaries

    def build_point_list(segments, boundaries):
        vcount = 0
        all_points = []
        for points, tris in segments:
            all_points.append(points)
            tris += vcount
            vcount += points.shape[0]
        for points, tris, _ in boundaries:
            all_points.append(points)
            tris += vcount
            vcount += points.shape[0]
        return np.vstack(all_points)

    def build_facet_list(segments, boundaries):
        all_facets = [tris for _, tris, _ in boundaries]
        all_markers = [
            np.full(len(all_facets[0]), 1), np.full(len(all_facets[3]), 2),
            np.full(len(all_facets[1]), 3), np.full(len(all_facets[4]), 4),
            np.full(len(all_facets[2]), 5), np.full(len(all_facets[5]), 6),
        ]
        fcount = 7
        for _, tris in segments:
            all_facets.append(tris)
            all_markers.append(np.full(len(tris), fcount))
            fcount += 1
        return np.vstack(all_facets), np.hstack(all_markers)

    def build_hole_list(segments):
        # TODO : Ultimately each sphere segment will contain hole data
        all_holes = []
        for points, _ in segments:
            all_holes.append(0.5 * (points.max(axis=0) + points.min(axis=0)))
        return np.vstack(all_holes)

    boundaries = duplicate_lower_boundaries(boundaries, Lx, Ly, Lz)

    points = build_point_list(segments, boundaries)
    # Fix boundary points to exactly zero
    for i in range(3):
        points[(np.isclose(points[:,i], 0.), i)] = 0.

    facets, markers = build_facet_list(segments, boundaries)
    holes = build_hole_list(segments)

    rad_edge = 1.4
    min_angle = 18.
    max_volume = None  # 0.00001
    # TODO : Don't mix and match between setting options with argument string
    #      : and option class attributes. Pick one and be consistent.
    options = tet.Options('pq{}/{}Y'.format(rad_edge, min_angle))
    options.docheck = 1
    options.verbose = 1

    mesh = tet.MeshInfo()
    mesh.set_points(points)
    mesh.set_facets(facets.tolist(), markers=markers.tolist())
    mesh.set_holes(holes)

    return tet.build(mesh, options=options, max_volume=max_volume)


def get_args(argv):
    """Get command line arguments
    :return: sphere center coordinates, x, y, z, sphere radius, r,
    domain box side lengths, Lx, Ly, Lz.
    """
    try:
        return float(argv[1]), float(argv[2]), float(argv[3]), argv[4], argv[5]
    except IndexError:
        raise UserWarning('Must specify Lx Ly Lz segment_file_prefix')
    except ValueError:
        raise UserWarning('Invalid arguments')


if __name__ == '__main__':
    import sys, devutils

    Lx, Ly, Lz, segment_prefix, boundary_prefix = get_args(sys.argv)
    segments = devutils.load_geomview_files(segment_prefix)
    boundaries = devutils.load_geomview_files(boundaries_prefix)
    mesh = build_tetmesh(segments, boundaries, Lx, Ly, Lz)
    mesh.write_vtk('mesh')
