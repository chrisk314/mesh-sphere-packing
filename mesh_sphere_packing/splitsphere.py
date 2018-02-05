
import numpy as np
from numpy import linalg as npl
from numpy import random as npr
from scipy.spatial.qhull import ConvexHull

from mesh_sphere_packing import TOL, ONE_THIRD


def flatten(l):
    """Return flattened list."""
    flat = []
    for x in l:
        if isinstance(x, list):
            flat += flatten(x)
        else:
            flat += [x]
    return flat


def reindex_tris(points, tris):
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
    points = points_reindexed[:count]
    tris = np.array([[reindex[idx] for idx in t] for t in tris])
    return points, tris


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

    def initialise_points(self, ds):
        self.ds = ds
        num_points = int(4. * np.pi * self.r**2 / ds**2)
        self.gen_spiral_points(num_points=num_points)
        self.filter_points()
        self.min = self.points.min(axis=0)
        self.max = self.points.max(axis=0)

    def gen_spiral_points(self, num_points=200):
        indices = np.arange(0, num_points, dtype=float) + 0.5

        phi = np.arccos(1 - 2*indices/num_points)
        theta = np.pi * (1 + 5**0.5) * indices

        self.points = np.empty((len(indices), 3))
        self.points[:,0] = self.x[0] + self.r * (np.cos(theta) * np.sin(phi))
        self.points[:,1] = self.x[1] + self.r * (np.sin(theta) * np.sin(phi))
        self.points[:,2] = self.x[2] + self.r * np.cos(phi)

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

    def split_axis_recursive(self, points, axis, trans):
        """Return nested list of points sets resulting from recursive splitting
        of sphere along the three coordinate axes.
        """
        cross_low = self.min[axis] < 0.
        cross_high = self.max[axis] > self.domain.L[axis]
        if cross_low or cross_high:
            self.split_axis[axis] = True
            trans_in, trans_out = trans.copy(), trans.copy()
            if cross_low:
                out = points[:, axis] < 0.
                trans_out[axis] = 1.
            elif self.max[axis] > self.domain.L[axis]:
                out = self.points[:, axis] > self.domain.L[axis]
                trans_out[axis] = -1.
                self.bound_high[axis] = True
            if axis < 2:
                return [
                    self.split_axis_recursive(points[~out], axis+1, trans_in),
                    self.split_axis_recursive(points[out], axis+1, trans_out)
                ]
            return [(points[~out], trans_in), (points[out], trans_out)]
        if axis < 2:
            return self.split_axis_recursive(points, axis+1, trans)
        return points, trans

    def split(self, domain):

        def curve_intersection_points(c, r, i, i1, i2, L):
            # Check for splits of this curve along axis i1
            ci_points = np.full((2,3), c[i])
            cnt = 0
            # If the curve just touches a bound along i1 add a single point...
            if np.isclose(c[i1] + r, L[i1]):    # ...on the upper bound...
                ci_points[cnt,i1] = L[i1]
                ci_points[cnt,i2] = c[i2]
                cnt += 1
            elif np.isclose(c[i1] - r, 0.):     # ...or on the lower bound.
                ci_points[cnt,i1] = 0.
                ci_points[cnt,i2] = c[i2]
                cnt += 1
            # Otherwise, if the sphere is split along i1 the curve may cross the bounds.
            elif self.split_axis[i1]:
                if c[i1] + r > L[i1]:   # Add two points at upper bound along i1...
                    ci_points[cnt:cnt+1,i1] = L[i1]
                elif c[i1] - r < 0.:    # ...or add two points at lower bound along i1...
                    ci_points[cnt:cnt+1,i1] = 0.
                else:                   # ...or add no points at bounds along i1.
                    return ci_points[:cnt]
                di1 = ci_points[cnt,i1] - c[i1]
                di2 = np.sqrt(r**2 - di1**2)
                ci_points[cnt,i2] = c[i2] + di2
                ci_points[cnt+1,i2] = c[i2] - di2
                cnt += 2
            return ci_points[:cnt] - c

        def get_add_phi_points(phi1, phi2, r, cx, i, ds):
            i1, i2 = (i+1)%3, (i+2)%3
            phi_disp = phi2 - phi1
            n_add = int(np.ceil(phi_disp * r / ds))
            phi = phi1 + (phi_disp / n_add) * np.arange(n_add+1)
            points = np.empty((len(phi),3))
            points[:,i] = cx[i]
            points[:,i1] = cx[i1] + r * np.cos(phi)
            points[:,i2] = cx[i2] + r * np.sin(phi)
            return points

        # Construct analytical intersection curves of sphere on boundaries
        def iloop_zones(points, axis, L):
            com = points.mean(axis=0)
            z = ~((0. < com) & (com < L))
            z[axis] = False
            z1 = 1*z[2] + 2*z[1] + 4*z[0]
            z[axis] = True
            z2 = 1*z[2] + 2*z[1] + 4*z[0]
            return (z1, z2)

        self.domain = domain

        if self.points is None:
            self.initialise_points()

        self.split_axis = np.full(3, False)  # True if the particle is split along axis
        self.bound_high = np.full(3, False)  # True/false if particle crosses low/high bound

        # Partition points of sphere into regions either side of boundaries
        sphere_pieces = flatten([
            self.split_axis_recursive(self.points, 0, np.zeros(3, dtype=np.float64))
        ])

        if not np.any(self.split_axis):
            # No splits so return the entire sphere
            return [SpherePiece(self, self.points, np.zeros(3), is_hole=True)]

        # Filter out empty pieces
        sphere_pieces = [
            (points, trans) for (points, trans) in sphere_pieces if len(points)
        ]
        sphere_pieces, translations = [list(tup) for tup in zip(*sphere_pieces)]
        # Construct zone to piece mapping
        zone_map = {
            1*bool(t[2]) + 2*bool(t[1]) + 4*bool(t[0]): idx
            for idx, t in enumerate(translations)
        }

        ci = []
        for i in range(3):
            if self.split_axis[i]:
                i1, i2 = (i+1)%3, (i+2)%3
                c = self.x.copy()
                c[i] = self.bound_high[i] * self.domain.L[i]
                r = np.sqrt(self.r**2 - (self.x[i] - c[i])**2)

                ci_points = np.vstack((
                    curve_intersection_points(c, r, i, i1, i2, domain.L),
                    curve_intersection_points(c, r, i, i2, i1, domain.L)
                ))

                # Sort points by angle 0 -> 2*pi
                if len(ci_points):
                    phi = np.angle(ci_points[:,i1] + 1j * ci_points[:,i2])
                    phi = np.sort(np.where(phi < 0., 2*np.pi+phi, phi))
                    phi = np.append(phi, phi[0] + 2 * np.pi)
                else:
                    phi = np.array([0., 2 * np.pi])

                # Add points to intersection curve
                for phi1, phi2 in zip(phi[:-1], phi[1:]):
                    add_points = get_add_phi_points(phi1, phi2, r, c, i, self.ds)
                    z1, z2 = iloop_zones(add_points, i, self.domain.L)
                    sphere_pieces[zone_map[z1]] = np.vstack((
                        sphere_pieces[zone_map[z1]], add_points
                    ))
                    sphere_pieces[zone_map[z2]] = np.vstack((
                        sphere_pieces[zone_map[z2]], add_points
                    ))

        return [
            SpherePiece(self, points, trans)
            for points, trans in zip(sphere_pieces, translations)
        ]


class SpherePiece(object):

    """Piece of a split sphere resulting from the intersection between
    a sphere and 0 to 3 planes representing the boundaries of a cuboid
    domain. The surface of the sphere is represented by a triangulated
    point set in R^3.
    """

    def __init__(self, sphere, points, trans_flag, is_hole=False):
        self.sphere = sphere
        self.domain = sphere.domain
        self.trans_flag = trans_flag
        self.points = points
        self.is_hole = is_hole

    def construct(self):
        self.triangulate_surface_points()
        if not self.is_hole:
            self.apply_laplacian_smoothing()
            self.translate_points()

    def triangulate_surface_points(self):

        def tri_centroid_disp(tris, points, sphere):
            c = ONE_THIRD * np.sum(points[tris], axis=1)
            return npl.norm(c - sphere.x, axis=1)

        def extract_surface_tris_from_chull(chull, sphere):
            tc_disp = tri_centroid_disp(chull.simplices, chull.points, sphere)
            mask = tc_disp > 0.8 * sphere.r
            surf_tris = chull.simplices[mask]
            return surf_tris

        if self.is_hole:
            chull = ConvexHull(self.points)
            self.points, self.tris = chull.points, chull.simplices
        else:
            com = self.points.mean(axis=0)
            add_point = self.sphere.x - 2. * (com - self.sphere.x)
            chull = ConvexHull(np.vstack((self.points, add_point)))
            surf_tris = extract_surface_tris_from_chull(chull, self.sphere)
            self.points, self.tris = reindex_tris(chull.points, surf_tris)

    def apply_laplacian_smoothing(self):
        # TODO : implement Laplacian smoothing of inner sphere piece vertices
        pass

    def translate_points(self):
        self.x = self.sphere.x + self.trans_flag * self.domain.L
        self.points += self.x - self.sphere.x


def splitsphere(domain, particles, config):
    sphere_pieces = []
    for p in particles:
        sphere = Sphere(p[0], p[1:4], p[4])
        sphere.initialise_points(config.segment_length)
        sphere_pieces += sphere.split(domain)
    for sphere_piece in sphere_pieces:
        sphere_piece.construct()
    return sphere_pieces
