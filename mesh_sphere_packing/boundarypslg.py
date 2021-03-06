import numpy as np
from numpy import linalg as npl
from meshpy import triangle

from mesh_sphere_packing import logger, ONE_THIRD, GROWTH_LIMIT
from mesh_sphere_packing.area_constraints import AreaConstraints

# TODO : change nomenclature. Segment is used in geometry to refer to an
#      : edge connecting two points. Here segment is used to refer to part
#      : of a sphere surface. This is confusing...


class PSLG(object):

    """Stores geometry and topology of a Planar Straigh Line Graph."""

    def __init__(self, points, edges, holes):
        """Constructs PSLG object.
        :param points numpy.ndarray: array of PSLG vertex coordinates.
        :param adges numpy.ndarray: array of PSLG edges (vertex topology).
        :param holes numpy.ndarray: array of coordinates of holes in the PSLG.
        """
        self.points = points
        self.edges = edges
        self.holes = holes


class BoundaryPLC(object):

    """Stores geometry and topology of a Piecewise Linear Complex forming a domain
    boundary.
    """

    def __init__(self, points, tris, holes):
        """Constructs BoundaryPLC object.
        :param points numpy.ndarray: array of PLC vertex coordinates.
        :param adges numpy.ndarray: array of PLC tris (vertex topology).
        :param holes numpy.ndarray: array of coordinates of holes in the PLC.
        """
        self.points = points
        self.tris = tris
        self.holes = holes


def build_boundary_PSLGs(domain, sphere_pieces, ds):
    """Constructs PSLGs for domain boundaries. Each boundary is represented by a
    Planar Straight Line Graph consisting of set of vertices and edges corresponding
    to the union of all intersection loops which lie on the boundary and all boundary
    perimeter vertices and edges.
    :param domain Domain: spatial domain for mesh.
    :param sphere_pieces list: list of SpherePiece objects.
    :param ds float: characteristic segment length.
    :return: list of PSLG objects for the lower bounds along each coordinate axis.
    :rtype: list.
    """
    # TODO : Break up this function a bit.

    def compile_points_edges(sphere_pieces):
        """Produces consolidated arrays containing all SpherePiece vertices and
        edges.
        :param sphere_pieces list: list of SpherePiece objects.
        :return: tuple of arrays of vertex coordinates and topology.
        :rtype: tuple.
        """
        def build_edge_list(tris, points):
            v_adj = np.zeros(2*[points.shape[0]], dtype=np.int32)
            v_adj[tris[:,0], tris[:,1]] = v_adj[tris[:,1], tris[:,0]] = 1
            v_adj[tris[:,1], tris[:,2]] = v_adj[tris[:,2], tris[:,1]] = 1
            v_adj[tris[:,2], tris[:,0]] = v_adj[tris[:,0], tris[:,2]] = 1
            return np.array(np.where(np.triu(v_adj) == 1), dtype=np.int32).T

        vcount = 0
        all_points = []
        all_edges = []
        for points, tris in [(p.points, p.tris) for p in sphere_pieces]:
            edges = build_edge_list(tris, points)
            edges += vcount
            vcount += len(points)
            all_points.append(points)
            all_edges.append(edges)
        return np.vstack(all_points), np.vstack(all_edges)

    def refined_perimeter(perim, axis, ds):
        """Adds additional vertices to subdivide perimeter edge segments.
        :param perim numpy.ndarray: array of vertices intersecting perimeter.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :param ds float: characteristic segment length.
        :return: array of vertices intersecting refined perimeter.
        :rtype: numpy.ndarray.
        """

        def filter_colocated_points(perim, axis):
            delta = np.diff(perim[:,axis])
            keep_idx = np.hstack(([0], np.where(~np.isclose(delta,0.))[0] + 1))
            return perim[keep_idx]

        perim = filter_colocated_points(perim, axis)
        refined_points = [perim[0]]
        for e in [[i, i+1] for i in range(perim.shape[0]-1)]:
            e_len = perim[e[1], axis] - perim[e[0], axis]
            ne = int(np.ceil(e_len / ds))
            if ne > 1:
                dse = e_len / ne
                add_points = np.zeros((ne,3))
                add_points[:,axis] = dse * np.arange(1,ne+1)
                refined_points.append(perim[e[0]] + add_points)
        return np.vstack(refined_points)

    def add_holes(sphere_pieces):
        """Add hole points to boundary PSLGs.
        :param sphere_pieces list: list of SpherePiece objects.
        :return: array of hole point vertices.
        :rtype: numpy.ndarray.
        """
        # TODO : this is a placeholder function. Ultimately holes need to
        #      : be created at the point when a sphere is split into pieces.
        holes = [[] for _ in range(3)]
        for i in range(3):
            j, k = (i+1)%3, (i+2)%3
            for points, tris in [(p.points, p.tris) for p in sphere_pieces]:
                points_ax = points[np.isclose(points[:,i], 0.)]
                if points_ax.shape[0]:
                    holes[i].append([
                        0.5 * (points_ax[:,j].max() + points_ax[:,j].min()),
                        0.5 * (points_ax[:,k].max() + points_ax[:,k].min())
                    ])
            holes[i] = np.vstack(holes[i]) if len(holes[i])\
                else np.empty((0,2), dtype=np.float64)
        return holes

    def reindex_edges(points, points_ax, edges_ax):
        """Reindexes edges along a given axis.
        :param points numpy.ndarray: all point coordinates.
        :param points_ax numpy.ndarray: indices of points intersecting boundary.
        :param edges_ax numpy.ndarray: edges interecting boundary.
        :return: tuple of arrays of point coordinates and reindexed edges.
        :rtype: tuple.
        """
        points_segment = points[points_ax]
        reindex = {old: new for new, old in enumerate(np.where(points_ax)[0])}
        for i, (v0, v1) in enumerate(edges_ax):
            edges_ax[i] = np.array([reindex[v0], reindex[v1]])
        return points_segment, edges_ax

    def build_perim_edge_list(points_pieces, perim_refined):
        """Construct list of perimeter edges for boundary.
        :param points_pieces numpy.ndarray: sphere points intersecting boundary.
        :param perim_refined numpy.ndarray: refined perimeter points.
        :return: array of perimeter edge topology for boundary.
        :rtype: numpy.ndarray.
        """
        # Need to adjust edge indices for perimeter segments
        v_count = len(points_pieces)
        perim_edges = 4 * [None]
        for j in range(4):
            v_count_perim = len(perim_refined[j])
            perim_vidx = np.empty(v_count_perim, dtype=np.int32)
            mask = np.full(v_count_perim, True)
            v_count_new = 0
            for i, p in enumerate(perim_refined[j]):
                vidx = np.where(np.isclose(npl.norm(points_pieces - p, axis=1), 0.))[0]
                if len(vidx):
                    mask[i] = False
                    perim_vidx[i] = vidx[0]
                else:
                    perim_vidx[i] = v_count + v_count_new
                    v_count_new += 1
            perim_edges[j] = np.array([
                [perim_vidx[k], perim_vidx[k+1]] for k in range(v_count_perim-1)
            ])
            perim_refined[j] = perim_refined[j][mask]
            v_count += v_count_new
        return perim_edges

    def add_point_plane_intersections(hole_pieces, axis, domain):
        """Adds points for sphere which just "touch" the boundary at a single point.
        :param hole_pieces list: list of SpherePiece objects.
        :param axis int: ordinal value of axis 0:x, 1:y, 2:z.
        :param domain Domain: spatial domain for mesh.
        :return: array of points touching boundary (may be empty).
        :rtype: numpy.ndarray.
        """
        added_points = []
        for hole_piece in hole_pieces:
            if np.isclose(hole_piece.sphere.min[axis], 0.):
                close = np.where(np.isclose(hole_piece.points[:,axis], 0.))[0]
                for idx in close:
                    added_points.append(hole_piece.points[idx])
            elif np.isclose(hole_piece.sphere.max[axis], domain.L[axis]):
                close = np.where(np.isclose(hole_piece.points[:,axis], domain.L[axis]))[0]
                trans = np.zeros(3)
                trans[axis] = -domain.L[axis]
                for idx in close:
                    added_points.append(hole_piece.points[idx] + trans)
        if added_points:
            return np.vstack(added_points)
        else:
            return np.empty((0,3), dtype=np.float64)

    L = domain.L
    PBC = domain.PBC

    sphere_pieces_holes = [p for p in sphere_pieces if p.is_hole]
    sphere_pieces = [p for p in sphere_pieces if not p.is_hole]

    # TODO : Optimise this by compliling only edges from sphere piece
    #      : intersection loops rather than considering all edges.
    if len(sphere_pieces):
        points, edges = compile_points_edges(sphere_pieces)
    else:
        points = np.empty((0,3), dtype=np.float64)
        edges = np.empty((0,2), dtype=np.int32)

    # Get edges and points on each boundary
    edges_ax = [
        edges[np.all(np.isclose(points[edges,i], 0.), axis=1)]
        for i in range(3)
    ]
    points_ax = [np.isclose(points[:,i], 0.) for i in range(3)]

    # Fix boundary points to exactly zero
    for i in range(3):
        points[(points_ax[i], i)] = 0.

    # reindex edge vertices
    points_pieces, edges_ax = [list(x) for x in zip(*[
        reindex_edges(points, points_ax[i], edges_ax[i]) for i in range(3)
    ])]
    perim = []
    perim_refined = []
    perim_segs = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    perim_edges = []

    for i in range(3):
        perim.append(4 * [None])
        perim_refined.append(4 * [None])
        # Rotate coordinate system by cyclic permutation of axes
        points_pieces[i][:,(0,1,2)] = points_pieces[i][:,(i,(i+1)%3,(i+2)%3)]

        corners = np.array([
            [0., 0., 0.], [0., L[1], 0.], [0., L[1], L[2]], [0., 0., L[2]]
        ])

        points_on_perim = 4 * [None]
        points_on_perim[0] = np.isclose(points_pieces[i][:, 2], 0.)
        points_on_perim[1] = np.isclose(points_pieces[i][:, 1], L[1])
        points_on_perim[2] = np.isclose(points_pieces[i][:, 2], L[2])
        points_on_perim[3] = np.isclose(points_pieces[i][:, 1], 0.)

        for j in range(4):
            axis = 1 + j % 2
            if PBC[axis] and j >= 2:
                continue
            perim[i][j] = np.vstack(
                (corners[perim_segs[j]], points_pieces[i][points_on_perim[j]])
            )
            if PBC[axis]:
                translate = np.array([0., 0., -L[2]]) if axis == 1\
                    else np.array([0., L[1], 0.])
                translated_points = points_pieces[i][points_on_perim[j + 2]]\
                    + translate
                perim[i][j] = np.vstack((perim[i][j], translated_points))
            perim[i][j] = perim[i][j][perim[i][j][:, axis].argsort()]
            perim_refined[i][j] = refined_perimeter(perim[i][j], axis, ds)
            if PBC[axis]:
                perim_refined[i][j+2] = perim_refined[i][j] - translate

        # Add the corner points so that duplicate coners can be filtered out
        # in build_perim_edge_list
        points_pieces[i] = np.append(points_pieces[i], corners, axis=0)

        perim_edges.append(
            build_perim_edge_list(points_pieces[i], perim_refined[i])
        )

        # Put coordinates back in proper order for this axis
        points_pieces[i][:,(i,(i+1)%3,(i+2)%3)] = points_pieces[i][:,(0,1,2)]

        L = L[np.newaxis, (1, 2, 0)][0]

    # TODO : refactor so boundary PSLG is built during above loop avoiding subsequent loops

    # add holes
    pslg_holes = add_holes(sphere_pieces)

    # Add points which lie on the boundaries from hole particles
    added_points = [
        add_point_plane_intersections(sphere_pieces_holes, i, domain)
        for i in range(3)
    ]

    # Group together segment and perimeter points and edges for each axis
    boundary_pslgs = []
    for i in range(3):
        pslg_points = np.vstack((
            points_pieces[i][:,((i+1)%3,(i+2)%3)],
            np.vstack(perim_refined[i])[:,(1,2)],
            added_points[i][:,((i+1)%3,(i+2)%3)]
        ))
        pslg_edges = np.vstack((edges_ax[i], np.vstack(perim_edges[i])))
        boundary_pslgs.append(PSLG(pslg_points, pslg_edges, pslg_holes[i]))
    return boundary_pslgs


def triangulate_PSLGs(pslgs, area_constraints):
    """Triangulates lower boundaries along each coordinate axis using Shewchuk's
    Triangle library.
    :param pslgs list: list of PSLG objects for the boundaries.
    :param area_constraints AreaConstraints: object storing area constraint grids for
    quality triangulation.
    :return: list of BoundaryPLC objects for the triangulated boundaries.
    :rtype: list.
    """
    triangulated_boundaries = []
    for i, pslg in enumerate(pslgs):

        target_area_grid = area_constraints.grid[i]
        inv_dx = area_constraints.inv_dx[i]
        inv_dy = area_constraints.inv_dy[i]

        def rfunc(vertices, area):
            (ox, oy), (dx, dy), (ax, ay) = vertices
            cx = ONE_THIRD * (ox + dx + ax)  # Triangle center x coord.
            cy = ONE_THIRD * (oy + dy + ay)  # Triangle center y coord.
            ix = int(cx * inv_dx)
            iy = int(cy * inv_dy)
            target_area = target_area_grid[iy][ix]
            return int(area > target_area)  # True -> 1 means refine

        # Set mesh info for triangulation
        mesh_data = triangle.MeshInfo()
        mesh_data.set_points(pslg.points)
        mesh_data.set_facets(pslg.edges.tolist())
        if len(pslg.holes):
            mesh_data.set_holes(pslg.holes)

        # Call triangle library to perform Delaunay triangulation
        max_volume = area_constraints.dA_max
        min_angle = 20.

        mesh = triangle.build(
            mesh_data,
            max_volume=max_volume,
            min_angle=min_angle,
            allow_boundary_steiner=False,
            refinement_func=rfunc
        )

        # Extract triangle vertices from triangulation adding back x coord
        points = np.column_stack((np.zeros(len(mesh.points)), np.array(mesh.points)))
        points = points[:,(-i%3,(1-i)%3,(2-i)%3)]
        tris = np.array(mesh.elements)
        holes = np.column_stack((np.zeros(len(mesh.holes)), np.array(mesh.holes)))
        holes = holes[:,(-i%3,(1-i)%3,(2-i)%3)]

        triangulated_boundaries.append(BoundaryPLC(points, tris, holes))
    return triangulated_boundaries


def boundarypslg(domain, sphere_pieces, config):
    """Handles creation of high quality triangulations of the domain boundaries.
    :param domain Domain: spatial domain for mesh.
    :param sphere_pieces list: list of SpherePiece objects.
    :param config Config: configuration for mesh build.
    :return: list of BoundaryPLC objects for the triangulated boundaries.
    :rtype: list.
    """
    logger.info('Triangulating domain boundaries')
    ds = config.segment_length

    boundary_PSLGs = build_boundary_PSLGs(domain, sphere_pieces, ds)
    area_constraints = AreaConstraints(domain, sphere_pieces, ds)

    return triangulate_PSLGs(boundary_PSLGs, area_constraints)
