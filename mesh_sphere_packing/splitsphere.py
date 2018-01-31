
import numpy as np
from numpy import linalg as npl
from numpy import random as npr
from scipy.spatial.qhull import ConvexHull

from mesh_sphere_packing import TOL, ONE_THIRD


class Domain(object):

    """Spatial cuboid shaped domain in R^3."""

    def __init__(self, L, PBC):
        self.L = np.array(L, dtype=np.float64)
        self.PBC = np.array(PBC, dtype=bool)
        self.volume = self.L.prod()


class Sphere(object):

    """Sphere in R^3 with a unique id."""

    def __init__(self, id, x, r):
        self.id = int(id)
        self.x = x
        self.r = r
        self.points = None

    def gen_spiral_points(self, num_points=200):
        indices = np.arange(0, num_points, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/num_points)
        theta = np.pi * (1 + 5**0.5) * indices

        self.points = np.empty((len(indices), 3))
        self.points[:,0] = self.x[0] + self.r * (np.cos(theta) * np.sin(phi))
        self.points[:,1] = self.x[1] + self.r * (np.sin(theta) * np.sin(phi))
        self.points[:,2] = self.x[2] + self.r * np.cos(phi)

        self.filter_points()

    def filter_points(self, strength=0.15):
        """Return array of filtered points excluding points lying too close
        to domain boundaries.
        """
        # TODO : This is a quick and dirty fix to prevent removal of points
        #      : for particles which do not cross boundaries. Ultimately these
        #      : cases needs to be handled in a more generally, comprehensively.
        out_x = self.points[:, 0] < 0.
        out_y = self.points[:, 1] < 0.
        out_z = self.points[:, 2] < 0.

        if not np.any([out_x, out_y, out_z]):
            return

        cutoff = strength * (len(self.points) / (4. * np.pi * self.r**2.))**-0.5

        close_x = np.abs(self.points[:,0]) < cutoff
        close_y = np.abs(self.points[:,1]) < cutoff
        close_z = np.abs(self.points[:,2]) < cutoff

        self.points = self.points[~close_x & ~close_y & ~close_z]

    def split(self, domain):

        self.domain = domain

        if not self.points:
            self.gen_spiral_points()

        # first split along x
        out_x = self.points[:, 0] < 0.
        out_y = self.points[:, 1] < 0.
        out_z = self.points[:, 2] < 0.

        # TODO : This is a quick and dirty fix to allow points for particles
        #      : inside the domain to be added without issue. Ultimately these
        #      : cases needs to be handled in a more generally, comprehensively.
        if not np.any([out_x, out_y, out_z]):
            # TODO : numpy array is returned here but this function returns
            #      : SpherePiece objects elsewhere!
            return [self.points]

        sphere_pieces = [
            self.points[~out_x & ~out_y & ~out_z],
            self.points[ out_x & ~out_y & ~out_z],
            self.points[~out_x &  out_y & ~out_z],
            self.points[ out_x &  out_y & ~out_z],
            self.points[~out_x & ~out_y &  out_z],
            self.points[ out_x & ~out_y &  out_z],
            self.points[~out_x &  out_y &  out_z],
            self.points[ out_x &  out_y &  out_z],
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

        r_inv = 1. / self.r
        ds = (len(self.points) / (4. * np.pi * self.r**2.))**-0.5

        # Curves on x boundary (y-z plane) -------------------------------------------------------------
        fn_x = lambda phi: np.full(len(phi), 0.)
        fn_y = lambda phi: self.x[1] + self.r * np.cos(phi)
        fn_z = lambda phi: self.x[2] + self.r * np.sin(phi)

        # curve 1
        phi_1 = np.arcsin(-self.x[2] * r_inv)
        phi_2 = 0.5 * np.pi + np.arcsin(self.x[1] * r_inv)
        curve_points = get_add_phi_points(phi_1, phi_2, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[0] = np.concatenate((sphere_pieces[0], curve_points), axis=0)
        sphere_pieces[1] = np.concatenate((sphere_pieces[1], curve_points), axis=0)

        # curve 2
        phi_3 = np.pi - phi_1
        curve_points = get_add_phi_points(phi_2, phi_3, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[2] = np.concatenate((sphere_pieces[2], curve_points), axis=0)
        sphere_pieces[3] = np.concatenate((sphere_pieces[3], curve_points), axis=0)

        # curve 3
        phi_4 = 2. * np.pi - phi_2
        curve_points = get_add_phi_points(phi_3, phi_4, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[6] = np.concatenate((sphere_pieces[6], curve_points), axis=0)
        sphere_pieces[7] = np.concatenate((sphere_pieces[7], curve_points), axis=0)

        # curve 3
        phi_5 = 2. * np.pi + phi_1
        curve_points = get_add_phi_points(phi_4, phi_5, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[4] = np.concatenate((sphere_pieces[4], curve_points), axis=0)
        sphere_pieces[5] = np.concatenate((sphere_pieces[5], curve_points), axis=0)

        # Curves on y boundary (x-z plane) -------------------------------------------------------------
        fn_x = lambda phi: self.x[0] + self.r * np.sin(phi)
        fn_y = lambda phi: np.full(len(phi), 0.)
        fn_z = lambda phi: self.x[2] + self.r * np.cos(phi)

        # curve 1
        phi_1 = np.arcsin(-self.x[0] * r_inv)
        phi_2 = 0.5 * np.pi + np.arcsin(self.x[2] * r_inv)
        curve_points = get_add_phi_points(phi_1, phi_2, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[0] = np.concatenate((sphere_pieces[0], curve_points), axis=0)
        sphere_pieces[2] = np.concatenate((sphere_pieces[2], curve_points), axis=0)

        # curve 2
        phi_3 = np.pi - phi_1
        curve_points = get_add_phi_points(phi_2, phi_3, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[4] = np.concatenate((sphere_pieces[4], curve_points), axis=0)
        sphere_pieces[6] = np.concatenate((sphere_pieces[6], curve_points), axis=0)

        # curve 3
        phi_4 = 2. * np.pi - phi_2
        curve_points = get_add_phi_points(phi_3, phi_4, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[5] = np.concatenate((sphere_pieces[5], curve_points), axis=0)
        sphere_pieces[7] = np.concatenate((sphere_pieces[7], curve_points), axis=0)

        # curve 3
        phi_5 = 2. * np.pi + phi_1
        curve_points = get_add_phi_points(phi_4, phi_5, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[1] = np.concatenate((sphere_pieces[1], curve_points), axis=0)
        sphere_pieces[3] = np.concatenate((sphere_pieces[3], curve_points), axis=0)

        # Curves on z boundary (x-y plane) -------------------------------------------------------------
        fn_x = lambda phi: self.x[0] + self.r * np.cos(phi)
        fn_y = lambda phi: self.x[1] + self.r * np.sin(phi)
        fn_z = lambda phi: np.full(len(phi), 0.)

        # curve 1
        phi_1 = np.arcsin(-self.x[1] * r_inv)
        phi_2 = 0.5 * np.pi + np.arcsin(self.x[0] * r_inv)
        curve_points = get_add_phi_points(phi_1, phi_2, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[0] = np.concatenate((sphere_pieces[0], curve_points), axis=0)
        sphere_pieces[4] = np.concatenate((sphere_pieces[4], curve_points), axis=0)

        # curve 2
        phi_3 = np.pi - phi_1
        curve_points = get_add_phi_points(phi_2, phi_3, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[1] = np.concatenate((sphere_pieces[1], curve_points), axis=0)
        sphere_pieces[5] = np.concatenate((sphere_pieces[5], curve_points), axis=0)

        # curve 3
        phi_4 = 2. * np.pi - phi_2
        curve_points = get_add_phi_points(phi_3, phi_4, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[3] = np.concatenate((sphere_pieces[3], curve_points), axis=0)
        sphere_pieces[7] = np.concatenate((sphere_pieces[7], curve_points), axis=0)

        # curve 3
        phi_5 = 2. * np.pi + phi_1
        curve_points = get_add_phi_points(phi_4, phi_5, self.r, ds, fn_x, fn_y, fn_z)
        sphere_pieces[2] = np.concatenate((sphere_pieces[2], curve_points), axis=0)
        sphere_pieces[6] = np.concatenate((sphere_pieces[6], curve_points), axis=0)

        # TODO : Implement logic to generate translations in a general way.
        translations = [
            np.array([0., 0., 0.]),
            np.array([1., 0., 0.]),
            np.array([0., 1., 0.]),
            np.array([1., 1., 0.]),
            np.array([0., 0., 1.]),
            np.array([1., 0., 1.]),
            np.array([0., 1., 1.]),
            np.array([1., 1., 1.]),
        ]

        return [
            SpherePiece(self, points, trans) for points, trans
            in zip(sphere_pieces, translations)
        ]


class SpherePiece(object):

    """Piece of a split sphere resulting from the intersection between
    a sphere and 0 to 3 planes representing the boundaries of a cuboid
    domain. The surface of the sphere is represented by a triangulated
    point set in R^3.
    """

    def __init__(self, sphere, points, trans_flag):
        self.sphere = sphere
        self.domain = sphere.domain
        self.trans_flag = trans_flag
        self.points = points

    def construct(self):
        self.triangulate_surface_points()
        self.apply_laplacian_smoothing()
        self.translate_points()

    def triangulate_surface_points(self):

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
            self.points = points_reindexed[:count]
            self.tris = np.array([[reindex[idx] for idx in t] for t in tris])

        def extract_surface_tris_from_chull(chull):
            # TODO : Find a cleaner/simpler/more elegant way of extracting sphere triangles.
            #      : Could check if all x, y, or z components are the same - i.e. triangle
            #      : lies in a plane.
            tri_norms = get_tri_norms(chull.simplices, chull.points)
            tri_pos_norms = get_tri_centroids_norm(
                chull.simplices, chull.points, self.sphere.x
            )
            mask = 1. - np.abs(np.sum(tri_pos_norms * tri_norms, axis=1)) < 0.1
            surf_tris = chull.simplices[mask]
            return surf_tris

        chull = ConvexHull(self.points)
        surf_tris = extract_surface_tris_from_chull(chull)
        reindex_tris(surf_tris, chull.points)

    def apply_laplacian_smoothing(self):
        # TODO : implement Laplacian smoothing of inner sphere piece vertices
        pass

    def translate_points(self):
        self.x = self.sphere.x + self.trans_flag * self.domain.L
        self.points += self.x - self.sphere.x


def splitsphere(domain, particles, config):
    particle = particles[0]
    sphere = Sphere(particle[0], particle[1:4], particle[4])
    sphere_pieces = sphere.split(domain)
    for sphere_piece in sphere_pieces:
        sphere_piece.construct()
    return sphere_pieces
