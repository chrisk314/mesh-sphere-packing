
import numpy as np
from numpy import linalg as npl
from numpy import random as npr
from scipy.spatial.qhull import ConvexHull

from mesh_sphere_packing import TOL, ONE_THIRD


def gen_sphere_spiral_points(x, y, z, r, num_points=200):
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    vx = x + r * (np.cos(theta) * np.sin(phi))
    vy = y + r * (np.sin(theta) * np.sin(phi))
    vz = z + r * np.cos(phi)

    return np.column_stack((vx, vy, vz))


def filter_points(points, r, strength=0.15):
    """Return array of filtered points excluding points lying too close
    to domain boundaries.
    """
    # TODO : This is a quick and dirty fix to prevent removal of points
    #      : for particles which do not cross boundaries. Ultimately these
    #      : cases needs to be handled in a more generally, comprehensively.
    out_x = points[:, 0] < 0.
    out_y = points[:, 1] < 0.
    out_z = points[:, 2] < 0.

    if not np.any([out_x, out_y, out_z]):
        return points

    cutoff = strength * (len(points) / (4. * np.pi * r**2.))**-0.5
    close_x = np.abs(points[:,0]) < cutoff
    close_y = np.abs(points[:,1]) < cutoff
    close_z = np.abs(points[:,2]) < cutoff
    return points[~close_x & ~close_y & ~close_z]


def split_sphere_points_segments(points, x, y, z, r, Lx, Ly, Lz):
    # first split along x
    out_x = points[:, 0] < 0.
    out_y = points[:, 1] < 0.
    out_z = points[:, 2] < 0.

    # TODO : This is a quick and dirty fix to allow points for particles
    #      : inside the domain to be added without issue. Ultimately these
    #      : cases needs to be handled in a more generally, comprehensively.
    if not np.any([out_x, out_y, out_z]):
        return [points]

    segments = [
        points[~out_x & ~out_y & ~out_z],
        points[ out_x & ~out_y & ~out_z],
        points[~out_x &  out_y & ~out_z],
        points[ out_x &  out_y & ~out_z],
        points[~out_x & ~out_y &  out_z],
        points[ out_x & ~out_y &  out_z],
        points[~out_x &  out_y &  out_z],
        points[ out_x &  out_y &  out_z],
    ]

    # TODO : refactor this huge procedural beast of a function

    def get_add_phi_points(phi_1, phi_2, r, ds, fn_x, fn_y, fn_z):
        phi_disp = phi_2 - phi_1
        n_add = int(np.ceil(phi_disp * r / ds))
        d_phi = phi_disp / n_add
        phi_i = phi_1 + d_phi * np.arange(n_add+1)
        # return np.column_stack((fn_x(phi_i), fn_y(phi_i), fn_z(phi_i)))
        # TODO : Only guaranteed to require extra point at origin for sphere
        #      : crossing all three boundaries
        return np.concatenate((
            np.zeros((1,3)),
            np.column_stack((fn_x(phi_i), fn_y(phi_i), fn_z(phi_i)))
        ))

    r_inv = 1. / r
    ds = (len(points) / (4. * np.pi * r**2.))**-0.5

    # Curves on x boundary (y-z plane) -------------------------------------------------------------
    fn_x = lambda phi: np.full(len(phi), 0.)
    fn_y = lambda phi: y + r * np.cos(phi)
    fn_z = lambda phi: z + r * np.sin(phi)

    # curve 1
    phi_1 = np.arcsin(-z * r_inv)
    phi_2 = 0.5 * np.pi + np.arcsin(y * r_inv)
    curve_points = get_add_phi_points(phi_1, phi_2, r, ds, fn_x, fn_y, fn_z)
    segments[0] = np.concatenate((segments[0], curve_points), axis=0)
    segments[1] = np.concatenate((segments[1], curve_points), axis=0)

    # curve 2
    phi_3 = np.pi - phi_1
    curve_points = get_add_phi_points(phi_2, phi_3, r, ds, fn_x, fn_y, fn_z)
    segments[2] = np.concatenate((segments[2], curve_points), axis=0)
    segments[3] = np.concatenate((segments[3], curve_points), axis=0)

    # curve 3
    phi_4 = 2. * np.pi - phi_2
    curve_points = get_add_phi_points(phi_3, phi_4, r, ds, fn_x, fn_y, fn_z)
    segments[6] = np.concatenate((segments[6], curve_points), axis=0)
    segments[7] = np.concatenate((segments[7], curve_points), axis=0)

    # curve 3
    phi_5 = 2. * np.pi + phi_1
    curve_points = get_add_phi_points(phi_4, phi_5, r, ds, fn_x, fn_y, fn_z)
    segments[4] = np.concatenate((segments[4], curve_points), axis=0)
    segments[5] = np.concatenate((segments[5], curve_points), axis=0)

    # Curves on y boundary (x-z plane) -------------------------------------------------------------
    fn_x = lambda phi: x + r * np.sin(phi)
    fn_y = lambda phi: np.full(len(phi), 0.)
    fn_z = lambda phi: z + r * np.cos(phi)

    # curve 1
    phi_1 = np.arcsin(-x * r_inv)
    phi_2 = 0.5 * np.pi + np.arcsin(z * r_inv)
    curve_points = get_add_phi_points(phi_1, phi_2, r, ds, fn_x, fn_y, fn_z)
    segments[0] = np.concatenate((segments[0], curve_points), axis=0)
    segments[2] = np.concatenate((segments[2], curve_points), axis=0)

    # curve 2
    phi_3 = np.pi - phi_1
    curve_points = get_add_phi_points(phi_2, phi_3, r, ds, fn_x, fn_y, fn_z)
    segments[4] = np.concatenate((segments[4], curve_points), axis=0)
    segments[6] = np.concatenate((segments[6], curve_points), axis=0)

    # curve 3
    phi_4 = 2. * np.pi - phi_2
    curve_points = get_add_phi_points(phi_3, phi_4, r, ds, fn_x, fn_y, fn_z)
    segments[5] = np.concatenate((segments[5], curve_points), axis=0)
    segments[7] = np.concatenate((segments[7], curve_points), axis=0)

    # curve 3
    phi_5 = 2. * np.pi + phi_1
    curve_points = get_add_phi_points(phi_4, phi_5, r, ds, fn_x, fn_y, fn_z)
    segments[1] = np.concatenate((segments[1], curve_points), axis=0)
    segments[3] = np.concatenate((segments[3], curve_points), axis=0)

    # Curves on z boundary (x-y plane) -------------------------------------------------------------
    fn_x = lambda phi: x + r * np.cos(phi)
    fn_y = lambda phi: y + r * np.sin(phi)
    fn_z = lambda phi: np.full(len(phi), 0.)

    # curve 1
    phi_1 = np.arcsin(-y * r_inv)
    phi_2 = 0.5 * np.pi + np.arcsin(x * r_inv)
    curve_points = get_add_phi_points(phi_1, phi_2, r, ds, fn_x, fn_y, fn_z)
    segments[0] = np.concatenate((segments[0], curve_points), axis=0)
    segments[4] = np.concatenate((segments[4], curve_points), axis=0)

    # curve 2
    phi_3 = np.pi - phi_1
    curve_points = get_add_phi_points(phi_2, phi_3, r, ds, fn_x, fn_y, fn_z)
    segments[1] = np.concatenate((segments[1], curve_points), axis=0)
    segments[5] = np.concatenate((segments[5], curve_points), axis=0)

    # curve 3
    phi_4 = 2. * np.pi - phi_2
    curve_points = get_add_phi_points(phi_3, phi_4, r, ds, fn_x, fn_y, fn_z)
    segments[3] = np.concatenate((segments[3], curve_points), axis=0)
    segments[7] = np.concatenate((segments[7], curve_points), axis=0)

    # curve 3
    phi_5 = 2. * np.pi + phi_1
    curve_points = get_add_phi_points(phi_4, phi_5, r, ds, fn_x, fn_y, fn_z)
    segments[2] = np.concatenate((segments[2], curve_points), axis=0)
    segments[6] = np.concatenate((segments[6], curve_points), axis=0)

    return segments


def translate_segments(segments, Lx, Ly, Lz):
    translations = [
        np.array([0., 0., 0.]),
        np.array([Lx, 0., 0.]),
        np.array([0., Ly, 0.]),
        np.array([Lx, Ly, 0.]),
        np.array([0., 0., Lz]),
        np.array([Lx, 0., Lz]),
        np.array([0., Ly, Lz]),
        np.array([Lx, Ly, Lz]),
    ]
    for (points, tris), t in zip(segments, translations):
        points += t
    return segments


def triangulate_segment_points(point_sets, x, y, z):
    """For now this function returns a set of convex hulls
    :return: list of scipy.spatial.qhull.ConvexHull instances
    """
    # TODO : this function needs to strip out only the triangles which
    #      : correspond to the surface of the sphere from the triangles
    #      : in the convex hull generated by qhull.

    def _tri_vec_prods(tris, points):
        tri_points = points[tris]
        AB = tri_points[:, 1] - tri_points[:, 0]
        AC = tri_points[:, 2] - tri_points[:, 0]
        return np.cross(AB, AC)

    def get_tri_norms(tris, points):
        norms = _tri_vec_prods(tris, points)
        return np.divide(norms, npl.norm(norms, axis=1)[:, np.newaxis])

    def get_tri_centroids_norm(tris, points, sphere_center):
        c = ONE_THIRD * np.sum(points[tris], axis=1)
        c_rel = c - sphere_center
        return np.divide(c_rel, npl.norm(c_rel, axis=1)[:, np.newaxis])

    def reindex_tris(tris, points):
        count = 0
        reindex = {}
        points_reindexed = np.empty(points.shape)
        for t in tris:
            for idx in t:
                try:
                    reindex[idx]
                except KeyError:
                    reindex[idx] = count
                    points_reindexed[count] = points[idx]
                    count += 1
        return (
            points_reindexed[:count],
            np.array([[reindex[idx] for idx in t] for t in tris]),
        )

    def extract_surf_from_chull(chull, x, y, z):
        sphere_center = np.array([x, y, z])
        # TODO : Find a cleaner/simpler/more elegant way of extracting sphere triangles
        tri_norms = get_tri_norms(chull.simplices, chull.points)
        tri_pos_norms = get_tri_centroids_norm(
            chull.simplices, chull.points, sphere_center
        )
        mask = 1. - np.abs(np.sum(tri_pos_norms * tri_norms, axis=1)) < 0.1
        surf_tris = chull.simplices[mask]
        return reindex_tris(surf_tris, chull.points)

    chulls = [ConvexHull(points) for points in point_sets]
    return [extract_surf_from_chull(h, x, y, z) for h in chulls]


def smooth_segments(segments):
    # TODO : implement Laplacian smoothing of inner segment vertices
    return segments


def splitsphere(args):
    x, y, z = args.particle_center
    r = args.particle_radius
    Lx, Ly, Lz = args.domain_dimensions

    points = gen_sphere_spiral_points(x, y, z, r)
    points = filter_points(points, r)
    segment_point_sets = split_sphere_points_segments(
        points, x, y, z, r, Lx, Ly, Lz
    )
    segments = triangulate_segment_points(segment_point_sets, x, y, z)
    segments = smooth_segments(segments)
    return translate_segments(segments, Lx, Ly, Lz)
