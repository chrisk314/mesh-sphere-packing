
import numpy as np
from numpy import linalg as npl
from numpy import random as npr
from scipy.spatial.qhull import ConvexHull

from mesh_sphere_packing import logger, TOL, ONE_THIRD


def flatten(l):
    """Return flattened list."""
    flat = []
    for x in l:
        flat += flatten(x) if isinstance(x, list) else [x]
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


def extend_domain(L, PBC, particles, ds):
    for axis in range(3):
        if not PBC[axis]:
            pad_extra = 0.5 * particles[:,4].min()

            pad_low = np.min(particles[:,axis+1] - particles[:,4])
            pad_low -= pad_extra
            pad_low = abs(pad_low) if pad_low < 0. else 0.

            pad_high = np.max(particles[:,axis+1] + particles[:,4]) - L[axis]
            pad_high += pad_extra
            pad_high = pad_high if pad_high > 0. else 0.

            L[axis] += pad_low + pad_high
            particles[:,axis+1] += pad_low
    return L, particles


def duplicate_particles(L, particles, config):
    if not any(config.duplicate_particles):
        return particles

    axis = [i for i, v in enumerate(config.duplicate_particles) if v][0]

    idx_dup_lower = np.where(particles[:,axis+1] - particles[:,4] < 0.)
    idx_dup_upper = np.where(particles[:,axis+1] + particles[:,4] > L[axis])

    trans_dup_lower = np.zeros(3, dtype=np.float64)
    trans_dup_lower[axis] = L[axis]
    trans_dup_upper = np.zeros(3, dtype=np.float64)
    trans_dup_upper[axis] = -1. * L[axis]

    particles_dup_lower = particles[idx_dup_lower]
    particles_dup_lower[:,(1,2,3)] += trans_dup_lower
    particles_dup_upper = particles[idx_dup_upper]
    particles_dup_upper[:,(1,2,3)] += trans_dup_upper

    return np.vstack((particles, particles_dup_lower, particles_dup_upper))


class Domain(object):

    """Cuboid shaped spatial domain in R^3."""

    def __init__(self, L, PBC):
        self.L = np.array(L, dtype=np.float64)
        self.PBC = np.array(PBC, dtype=bool)
        self.volume = self.L.prod()


class IntersectionPoints(object):

    def __init__(self, i_loop, points):
        self.i_loop = i_loop
        self.points = points

    @property
    def is_full_loop(self):
        return self.i_loop.is_full_loop

    @property
    def hole_point(self):
        return np.mean(self.points, axis=0)

    @property
    def origin(self):
        if len(self.i_loop.ci_points) < 2:
            return self.i_loop.c
        if len(self.i_loop.ci_points) == 2:
            return self.i_loop.c + np.mean(self.i_loop.ci_points, axis=0)
        return self.i_loop.sphere.bound_high * self.i_loop.domain.L

    def __str__(self):
        return '{}'.format(self.points)


class IntersectionLoop(object):

    """Represents the intersection curve between a sphere and a plane."""

    def __init__(self, sphere, domain, axis):
        self.sphere = sphere
        self.domain = domain
        self.i0, self.i1, self.i2 = axis, (axis+1)%3, (axis+2)%3

        self.set_analytical_curve()
        self.set_axis_intersection_points()
        self.add_phi_points()

    def set_analytical_curve(self):
        i0 = self.i0
        self.c = self.sphere.x.copy()
        self.c[i0] = self.sphere.bound_high[i0] * self.domain.L[i0]
        self.r = np.sqrt(self.sphere.r**2 - (self.sphere.x[i0] - self.c[i0])**2)

    def set_axis_intersection_points(self):
        self.ci_points = np.vstack((
            self.curve_intersection_points(self.i1, self.i2),
            self.curve_intersection_points(self.i2, self.i1)
        ))
        self.is_full_loop = not bool(len(self.ci_points))

    def curve_intersection_points(self, i1, i2):
        # Check for splits of this curve along axis i1
        ci_points = np.full((2,3), self.c[self.i0])
        cnt = 0
        # If the curve just touches a bound along i1, add a single point...
        if np.isclose(self.c[i1] + self.r, self.domain.L[i1]):  # ...on the upper bound...
            ci_points[cnt,i1] = self.domain.L[i1]
            ci_points[cnt,i2] = self.c[i2]
            cnt += 1
        elif np.isclose(self.c[i1] - self.r, 0.):  # ...or on the lower bound.
            ci_points[cnt,i1] = 0.
            ci_points[cnt,i2] = self.c[i2]
            cnt += 1
        # Otherwise, if the sphere is split along i1 the curve may cross the bounds.
        elif self.sphere.split_axis[i1]:
            # Add two points at upper bound along i1...
            if self.c[i1] + self.r > self.domain.L[i1]:
                ci_points[cnt:cnt+2,i1] = self.domain.L[i1]
            # ...or add two points at lower bound along i1...
            elif self.c[i1] - self.r < 0.:
                ci_points[cnt:cnt+2,i1] = 0.
            # ...or add no points at bounds along i1.
            else:
                return ci_points[:cnt]
            di1 = ci_points[cnt,i1] - self.c[i1]
            di2 = np.sqrt(self.r**2 - di1**2)
            ci_points[cnt,i2] = self.c[i2] + di2
            ci_points[cnt+1,i2] = self.c[i2] - di2
            cnt += 2
        return ci_points[:cnt] - self.c

    def add_phi_points(self):
        if len(self.ci_points):
            phi = np.angle(
                self.ci_points[:,self.i1] + 1j * self.ci_points[:,self.i2]
            )
            phi = np.sort(np.where(phi < 0., 2*np.pi+phi, phi))
            phi = np.append(phi, phi[0] + 2 * np.pi)
        else:
            phi = [0., 2. * np.pi]
        # Add points to intersection curve
        self.added_points = []
        for phi1, phi2 in zip(phi[:-1], phi[1:]):
            add_points = self._add_phi_points(phi1, phi2)
            self.added_points.append((
                self.iloop_zones(add_points),
                IntersectionPoints(self, add_points)
            ))

    def _add_phi_points(self, phi1, phi2):
        phi_disp = phi2 - phi1
        n_add = int(np.ceil(phi_disp * self.r / self.sphere.ds))
        phi = phi1 + (phi_disp / n_add) * np.arange(n_add+1)
        points = np.empty((len(phi),3))
        points[:,self.i0] = self.c[self.i0]
        points[:,self.i1] = self.c[self.i1] + self.r * np.cos(phi)
        points[:,self.i2] = self.c[self.i2] + self.r * np.sin(phi)
        return points

    def iloop_zones(self, points):
        com = points.mean(axis=0)
        z = ~((0. < com) & (com < self.domain.L))
        z[self.i0] = False
        z1 = 1*z[2] + 2*z[1] + 4*z[0]
        z[self.i0] = True
        z2 = 1*z[2] + 2*z[1] + 4*z[0]
        return (z1, z2)

    def __str__(self):
        return 'c: {},\nr: {},\ni_points: {}\n'.format(
            self.c, self.r, [(x[0], x[1].points) for x in self.added_points]
        )


class Sphere(object):

    """Sphere in R^3 with a unique id."""

    def __init__(self, id, x, r, config):
        self.config = config
        self.id = int(id)
        self.x = x
        self.r = r
        self.points = None
        self.i_loops = []

    def initialise_points(self):
        self.ds = self.config.segment_length
        num_points = int(4. * np.pi * self.r**2 / self.ds**2)
        self.gen_spiral_points(num_points=num_points)
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
        """Filter out points lying too close to domain boundaries."""
        cutoff = strength * self.ds

        close_ax = [np.isclose(
            self.points[:,i], self.bound_high[i] * self.domain.L[i],
            atol=cutoff, rtol=0.
        ) for i in range(3)]

        self.points = self.points[~close_ax[0] & ~close_ax[1] & ~close_ax[2]]

    def set_split_planes(self):
        self.split_axis = np.full(3, False)  # True if sphere is split along axis
        self.bound_high = np.full(3, False)  # True/false if crossing high/low bound
        for axis in range(3):
            if self.min[axis] < 0.:
                self.split_axis[axis] = True
            elif  self.max[axis] > self.domain.L[axis]:
                self.split_axis[axis] = True
                self.bound_high[axis] = True

    def split_axis_recursive(self, points, axis, trans):
        """Return nested list of point sets resulting from recursive splitting
        of sphere along the three coordinate axes.
        """
        if self.split_axis[axis]:
            trans_in, trans_out = trans.copy(), trans.copy()
            if self.bound_high[axis]:
                out = points[:, axis] > self.domain.L[axis]
                trans_out[axis] = -1.
            else:
                out = points[:, axis] < 0.
                trans_out[axis] = 1.
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

        self.domain = domain

        if self.points is None:
            self.initialise_points()

        self.set_split_planes()

        if not np.any(self.split_axis):
            # No splits so return the entire sphere
            return [SpherePiece(self, self.points, np.zeros(3), [], is_hole=True)]

        self.filter_points()

        # Partition points of sphere into regions either side of boundaries
        sphere_pieces = flatten([
            self.split_axis_recursive(self.points, 0, np.zeros(3, dtype=np.float64))
        ])
        sphere_pieces, translations = [list(tup) for tup in zip(*sphere_pieces)]

        # Construct zone to piece mapping
        zone_map = {
            1*bool(t[2]) + 2*bool(t[1]) + 4*bool(t[0]): idx
            for idx, t in enumerate(translations)
        }

        i_loop_points = [[] for sp in sphere_pieces]

        for i in range(3):
            if not self.split_axis[i]:
                continue

            ci = IntersectionLoop(self, self.domain, i)

            if not ci.is_full_loop:
                for (z1, z2), i_points in ci.added_points:
                    i_loop_points[zone_map[z1]].append(i_points)
                    i_loop_points[zone_map[z2]].append(i_points)
            else:
                (z1, z2), i_points = ci.added_points[0]
                if len(i_points.points) < 3:
                    # Intersection loop so small that only one segment lies on
                    # the boundary. Replace it with a single point.
                    i_points.points = np.mean(i_points.points, axis=0)
                    if len(sphere_pieces[zone_map[z1]]):
                        # Only one piece will contain points. Add the extra point
                        # to this piece and leave the other empty.
                        i_loop_points[zone_map[z1]].append(i_points)
                    else:
                        i_loop_points[zone_map[z2]].append(i_points)
                else:
                    if len(sphere_pieces[zone_map[z1]])\
                            and len(sphere_pieces[zone_map[z2]]):
                        i_loop_points[zone_map[z1]].append(i_points)
                        i_loop_points[zone_map[z2]].append(i_points)
                    else:
                        if not len(sphere_pieces[zone_map[z2]]):
                            z1, z2 = z2, z1
                        # Open intersection loop on boundary but no points on
                        # one side. Add an extra point on the sphere surface.
                        surface_point = self.x.copy()
                        surface_point[i] += self.r * np.sign(ci.c[i] - self.x[i])
                        sphere_pieces[zone_map[z1]] = surface_point.reshape((1,3))
                        i_loop_points[zone_map[z1]].append(i_points)
                        i_loop_points[zone_map[z2]].append(i_points)

            self.i_loops.append(ci)

        return [
            SpherePiece(self, points, trans, i_points)
            for points, trans, i_points
            in zip(sphere_pieces, translations, i_loop_points)
            if len(points) or len(i_points)
        ]


class SpherePiece(object):

    """Piece of a split sphere resulting from the intersection between
    a sphere and 0 to 3 planes representing the boundaries of a cuboid
    domain. The surface of the sphere is represented by a triangulated
    point set in R^3.
    """

    def __init__(
            self, sphere, points, trans_flag, i_points_list,
            is_hole=False
        ):
        self.sphere = sphere
        self.domain = sphere.domain
        self.trans_flag = trans_flag
        self.is_hole = is_hole
        self.points = points
        self.i_points_list = i_points_list

    def construct(self):
        self.triangulate_surface_points()
        if not self.is_hole:
            self.apply_laplacian_smoothing()
            self.translate_points()
        else:
            self.handle_points_near_boundaries()

    def handle_points_near_boundaries(self, strength=0.10):
        """Move points lying too close to domain boundaries to prevent bad tets."""
        cutoff = strength * self.sphere.ds
        self.sphere.bound_high = self.sphere.x > self.domain.L / 2.
        dR = 0.05 * self.sphere.ds

        close_ax = [np.isclose(
            self.points[:,i], self.sphere.bound_high[i] * self.domain.L[i],
            atol=cutoff, rtol=0.
        ) for i in range(3)]

        for i in range(3):
            if not np.any(close_ax[i]):
                continue
            if self.sphere.bound_high[i]:
                hemisphere_points = self.points[:,i] > self.sphere.x[i]
            else:
                hemisphere_points = self.points[:,i] < self.sphere.x[i]
            dx = np.abs(self.points[hemisphere_points, i] - self.sphere.x[i])
            adjustment = dR * (dx / self.sphere.r)**2.
            adjustment *= (1. - 2. * self.sphere.bound_high[i])
            self.points[hemisphere_points,i] += adjustment

    def i_loop_points(self):
        return np.vstack([
            i_points.points for i_points in self.i_points_list
        ])

    def i_loop_origin_points(self):
        return np.vstack([
            i_points.origin for i_points in self.i_points_list
        ])

    def extract_surface_tris_from_chull(self, chull):

        def tri_vec_prods(tris, points):
            tri_points = points[tris]
            AB = tri_points[:, 1] - tri_points[:, 0]
            AC = tri_points[:, 2] - tri_points[:, 0]
            return np.cross(AB, AC)

        def get_tri_norms(tris, points):
            norms = tri_vec_prods(tris, points)
            return np.divide(norms, npl.norm(norms, axis=1)[:, np.newaxis])

        def get_tri_centroids_norm(tris, points, sphere_center):
            c = ONE_THIRD * np.sum(points[tris], axis=1)
            c_rel = c - sphere_center
            return np.divide(c_rel, npl.norm(c_rel, axis=1)[:, np.newaxis])

        def origin_to_tri_vec(tris, points, origin):
            c = ONE_THIRD * np.sum(points[tris], axis=1)
            c_rel = c - origin
            return np.divide(c_rel, npl.norm(c_rel, axis=1)[:, np.newaxis])

        tri_norms = get_tri_norms(chull.simplices, chull.points)
        mask = np.full(len(chull.simplices), True)
        for op in self.i_loop_origin_points():
            ot_norm = origin_to_tri_vec(chull.simplices, chull.points, op)
            op_mask = np.abs(np.sum(ot_norm * tri_norms, axis=1)) < 0.01
            mask *= ~op_mask
        return chull.simplices[mask]

    def triangulate_surface_points(self):
        if self.is_hole:
            chull = ConvexHull(self.points)
            self.points, self.tris = chull.points, chull.simplices
        else:
            chull_points = np.vstack((
                self.points,
                self.i_loop_points(),
                self.i_loop_origin_points()
            ))
            chull = ConvexHull(chull_points)
            surf_tris = self.extract_surface_tris_from_chull(chull)
            self.points, self.tris = reindex_tris(chull.points, surf_tris)

    def apply_laplacian_smoothing(self):
        # TODO : implement Laplacian smoothing of inner sphere piece vertices
        pass

    def translate_points(self):
        self.x = self.sphere.x + self.trans_flag * self.domain.L
        self.points += self.x - self.sphere.x


def splitsphere(domain, particles, config):
    logger.info('Splitting input particles')
    sphere_pieces = []
    for p in particles:
        sphere = Sphere(p[0], p[1:4], p[4], config)
        sphere.initialise_points()
        sphere_pieces += sphere.split(domain)
    for sphere_piece in sphere_pieces:
        sphere_piece.construct()
    return sphere_pieces
