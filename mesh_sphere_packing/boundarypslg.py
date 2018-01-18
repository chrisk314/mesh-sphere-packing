#!/usr/bin/env python
from __future__ import print_function

import os
import sys
from pprint import pprint

import numpy as np
from numpy import linalg as npl
from meshpy import triangle, tet

WITH_PBC = True

# TODO : change nomenclature. Segment is used in geometry to refer to an
#      : edge connecting two points. Here segment is used to refer to part
#      : of a sphere surface. This is confusing...


def write_geomview(points, tris, fname):
    with open(fname, 'w') as f:
        f.write('OFF\n')
        f.write('%d %d %d\n' % (len(points), len(tris), 0))
        for p in points:
            f.write('%+1.15e %+1.15e %+1.15e\n' % (p[0], p[1], p[2]))
        for t in tris:
            f.write('3 %d %d %d\n' % (t[0], t[1], t[2]))


def output_boundaries_geomview(boundaries):
    for i, (points, tris, _) in enumerate(boundaries):
        write_geomview(points, tris, './boundary_%d.off' % i)


def write_poly(points, edges, holes, fname):
    with open(fname, 'w') as f:
        f.write('%d 2 0 0\n' % len(points))
        for i, p in enumerate(points):
            f.write('%d %+1.15e %+1.15e\n' % (i, p[0], p[1]))
        f.write('%d 0\n' % len(edges))
        for i, e in enumerate(edges):
            f.write('%d %d %d\n' % (i, e[0], e[1]))
        f.write('%d\n' % len(holes))
        for i, h in enumerate(holes):
            f.write('%d %+1.15e %+1.15e\n' % (i, h[0], h[1]))


def output_boundaries_poly(boundaries):
    for i, (points, edges, holes) in enumerate(boundaries):
        write_poly(points, edges, holes, './boundary_%d.poly' % i)


def plot_points_edges(points, edges):
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection
    lc = LineCollection(points[edges])
    fig = plt.figure()
    plt.gca().add_collection(lc)
    space = 0.05 * points.max()
    plt.xlim(points[:,0].min() - space, points[:,0].max() + space)
    plt.ylim(points[:,1].min() - space, points[:,1].max() + space)
    plt.plot(points[:,0], points[:,1], 'ro')
    fig.savefig('points_edges.png')
    plt.show()


def read_geomview(fname):
    with open(fname, 'r') as f:
        f.readline()
        counts = [int(tok) for tok in f.readline().strip().split()]
        points = np.array([
            [float(tok) for tok in f.readline().strip().split()]
            for i in range(counts[0])
        ])
        tris = np.array([
            [int(tok) for tok in f.readline().strip().split()[1:]]
            for i in range(counts[1])
        ])
    return points, tris


def load_segments_geomview(prefix):
    geomfiles = [
        fname for fname in os.listdir(os.getcwd())
        if fname.startswith(prefix) and os.path.splitext(fname)[1] == '.off'
    ]
    geomfiles.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    return [read_geomview(fname) for fname in geomfiles]


def create_cgal_polyhedron_3(segments):
    from CGAL.CGAL_Kernel import Point_3, Segment_3, Triangle_3, Plane_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
    s = segments[0]
    P = Polyhedron_3()
    cgal_points = [Point_3(*p) for p in s[0]]
    for t in s[0]:
        P.make_triangle(*[cgal_points[v] for v in t])
    return P


def tet_cube():
    points = np.array([
        [0.,0.,0.], [0.,1.,0.], [0.,1.,1.], [0.,0.,1.],
        [1.,0.,0.], [1.,1.,0.], [1.,1.,1.], [1.,0.,1.]
    ])
    facets = np.array([
        [0,1,2,3], [0,4,5,1], [0,3,7,4],
        [6,2,1,5], [6,7,3,2], [6,5,4,7]
    ])

    mesh_data = tet.MeshInfo()
    mesh_data.set_points(points)
    mesh_data.set_facets(facets)

    rad_edge = 1.4
    min_angle = 18.
    max_volume = None  # 0.00001
    #options = tet.Options('pq{}/{}Y'.format(rad_edge, min_angle))
    options = tet.Options('pq')

    return tet.build(mesh_data, options=options, max_volume=max_volume)


def tet_cube_with_tet(opt_str=None):
    points = np.array([
        [0.,0.,0.], [0.,1.,0.], [0.,1.,1.], [0.,0.,1.],
        [1.,0.,0.], [1.,1.,0.], [1.,1.,1.], [1.,0.,1.],
        [0.5,0.6,0.5],[0.5,0.4,0.5],[0.6,0.5,0.5],[0.5,0.5,0.6]
    ])
    facets = np.array([
        [0,1,2,3], [0,4,5,1], [0,3,7,4],
        [6,2,1,5], [6,7,3,2], [6,5,4,7],
        [8,9,10], [8,11,9], [8,11,10], [9,11,10]
    ])

    mesh_data = tet.MeshInfo()
    mesh_data.set_points(points)
    mesh_data.set_facets(facets)
    mesh_data.set_holes(np.array([[0.51,0.51,0.51]]))

    max_volume = None  # 0.00001
    options = tet.Options(opt_str if opt_str else 'pq')

    return tet.build(mesh_data, options=options, max_volume=max_volume)


def get_args(argv):
    """Get command line arguments
    :return: sphere center coordinates, x, y, z, sphere radius, r,
    domain box side lengths, Lx, Ly, Lz.
    """
    try:
        return float(argv[1]), float(argv[2]), float(argv[3]), argv[4]
    except IndexError:
        raise UserWarning('Must specify Lx Ly Lz segment_file_prefix')
    except ValueError:
        raise UserWarning('Invalid arguments')


def build_boundary_PSLGs(segments, Lx, Ly, Lz):
    # TODO : Break up this function a bit.

    # Build list of edges for segment
    def build_edge_list(tris, points):
        v_adj = np.zeros(2*[points.shape[0]], dtype=np.int64)
        v_adj[tris[:,0], tris[:,1]] = v_adj[tris[:,1], tris[:,0]] = 1
        v_adj[tris[:,1], tris[:,2]] = v_adj[tris[:,2], tris[:,1]] = 1
        v_adj[tris[:,2], tris[:,0]] = v_adj[tris[:,0], tris[:,2]] = 1
        return np.array(np.where(np.triu(v_adj) == 1)).T

    def refined_perimeter(perim, axis):
        # TODO : remove after development
        return perim
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

    def add_holes(segments):
        # TODO : this is a placeholder function. Ultimately holes need to
        #      : be created at the point when a sphere is split into pieces.
        holes = [[] for _ in range(3)]
        for i in range(3):
            j, k = (i+1)%3, (i+2)%3
            for points, tris in segments:
                points_ax = points[np.isclose(points[:,i], 0.)]
                if points_ax.shape[0]:
                    holes[i].append([
                        0.5 * (points_ax[:,j].max() + points_ax[:,j].min()),
                        0.5 * (points_ax[:,k].max() + points_ax[:,k].min())
                    ])
            holes[i] = np.vstack(holes[i]) if len(holes[i]) else []
        return holes

    def reindex_edges(points, points_ax, edges_ax)
        edges_flat = edges_ax.flatten()
        edges_flat_reindexed = np.zeros(
            edges_ax.shape[0] * edges_ax.shape[1], dtype=np.int64
        )
        points_segment = np.empty((points_ax.sum(),3))
        reindex = {}
        count = 0
        for j, vidx in enumerate(edges_flat):
            try:
                edges_flat_reindexed[j] = reindex[vidx]
            except KeyError:
                points_segment[count] = points[vidx]
                reindex[vidx] = count
                edges_flat_reindexed[j] = count
                count += 1
        return points_segments, edges_flat_reindexed.reshape(edges_ax.shape)

    L = np.array([Lx, Ly, Lz])

    # TODO : refactor to support multiple sphere segments
    points, tris  = segments[0]

    edges = build_edge_list(tris, points)

    # TODO : get target point separation from segment properties.
    #      : For now, get this from mean edge length.
    ds = np.mean(npl.norm(points[edges[:,0]] - points[edges[:,1]], axis=1))

    # Get edges and points on each boundary
    edges_ax = [
        edges[np.all(np.isclose(points[edges,i], 0.), axis=1)]
        for i in range(3)
    ]
    points_ax = [np.isclose(points[:,i], 0.) for i in range(3)]

    # Fix boundary points to exactly zero
    for i in range(3):
        points[(points_ax[i], i)] = 0.

    perim = perim_refined = 3 * [4 * [None]]
    perim_segs = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])

    for j in range(3):
        corners = np.array([
            [0., 0., 0.], [0., L[1], 0.], [0., L[1], L[2]], [0., 0., L[2]]
        ])

        points_on_perim = 4 * [None]
        points_on_perim[0] = np.isclose(points[points_ax[j], 2], 0.)
        points_on_perim[1] = np.isclose(points[points_ax[j], 1], L[1])
        points_on_perim[2] = np.isclose(points[points_ax[j], 2], L[2])
        points_on_perim[3] = np.isclose(points[points_ax[j], 1], 0.)

        for i in range(2 + 2 * int(not WITH_PBC)):
            axis = 1 + i % 2
            perim[j][i] = np.vstack(
                (corners[perim_segs[i]], points[points_ax[j]][points_on_perim[i]])
            )
            if WITH_PBC:
                translate = np.array([0., 0., -L[2]]) if axis == 1\
                    else np.array([0., L[1], 0.])
                translated_points = points[points_ax[j]][points_on_perim[i + 2]] + translate
                perim[j][i] = np.vstack((perim[j][i], translated_points))
            perim[j][i] = perim[j][i][perim[j][i][:, axis].argsort()]
            perim_refined[j][i] = refined_perimeter(perim[j][i], axis)
            if WITH_PBC:
                perim[j][i+2] = perim[j][i] - translate
                perim_refined[j][i+2] = perim_refined[j][i] - translate
        # Rotate coordinate system by cyclic permutation of axes
        points = points[:, (1, 2, 0)]
        L = L[np.newaxis, (1, 2, 0)][0]

    # TODO : refactor so boundary PSLG is built during above loop avoiding subsequent loops
    # reindex edge vertices
    # TODO : There must be a better way of reindexing the edge vertices that this...
    points_segments, edges_ax = zip(*[
        reindex_edges(points, points_ax[i], edges_ax[i]) for i in range(3)
    ])


    # Build lists of perimeter edges
    perim_edges = 3 * [4 * [None]]
    for i in range(3):
        # Need to adjust edge indices for perimeter segments
        v_count = points_ax[i].sum()
        # v_count = points.shape[0]
        for j in range(4):
            npoints_perim = perim_refined[i][j].shape[0]
            perim_edges[i][j] = np.array([[k, k+1] for k in range(npoints_perim-1)])
            # TODO : should be modifying vertex indices by running vertex count!
            perim_edges[i][j] += v_count
            v_count += npoints_perim

    # add holes
    pslg_holes = add_holes(segments)

    # Group together segment and perimeter points and edges for each axis
    # TODO : vertex indices for sphere vertices in edge array need to be
    #      : reindexed as a reduced set is passed to triangle.
    boundary_pslgs = []
    for i in range(3):
        pslg_points = np.vstack((
            points_segments[i][:,((i+1)%3,(i+2)%3)],
            np.vstack(perim_refined[i])[:,(1,2)]
        ))
        pslg_edges = np.vstack((edges_ax[i], np.vstack(perim_edges[i])))
        boundary_pslgs.append((pslg_points, pslg_edges, pslg_holes[i]))
    return boundary_pslgs


def triangulate_PSLGs(pslgs):
    triangulated_boundaries = []
    for i, (points, edges, holes) in enumerate(pslgs):
        # Set mesh info for triangulation
        mesh_data = triangle.MeshInfo()
        mesh_data.set_points(points)
        mesh_data.set_facets(edges)
        if len(holes):
            mesh_data.set_holes(holes)

        # Call triangle library to perform Delaunay triangulation
        # TODO : set max_volume based on geometry
        ds = np.mean(npl.norm(points[edges[:,0]] - points[edges[:,1]], axis=1))
        max_volume = ds**2
        min_angle = 20.

        mesh = triangle.build(
            mesh_data,
            max_volume=max_volume,
            min_angle=min_angle,
            allow_boundary_steiner=False
        )

        # Extract triangle vertices from triangulation adding back x coord
        points = np.column_stack((np.zeros(len(mesh.points)), np.array(mesh.points)))
        points = points[:,(-i%3,(1-i)%3,(2-i)%3)]
        tris = np.array(mesh.elements)
        # TODO : If holes are not required after triangulating boundaries then remove
        holes = np.column_stack((np.zeros(len(holes)), holes))
        holes = holes[:,(-i%3,(1-i)%3,(2-i)%3)]

        triangulated_boundaries.append((points, tris, holes))
    return triangulated_boundaries


def build_tet_mesh(segments, boundaries, Lx, Ly, Lz):

    def duplicate_lower_boundaries(lower_boundaries, Lx, Ly, Lz):
        L = np.array([Lx, Ly, Lz])
        upper_boundaries = []
        for i, (points, tris, holes) in enumerate(lower_boundaries):
            translate = np.array([L[j] if j==i else 0. for j in range(3)])
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
        for _, tris in segments[:1]:
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
    pprint(segments)
    pprint(boundaries)

    points = build_point_list(segments, boundaries)
    pprint(segments)
    pprint(boundaries)
    facets, markers = build_facet_list(segments, boundaries)
    holes = build_hole_list(segments)

    rad_edge = 1.4
    min_angle = 18.
    max_volume = None  # 0.00001
    #options = tet.Options('pq{}/{}Y'.format(rad_edge, min_angle))
    options = tet.Options('pq')

    mesh = tet.MeshInfo()
    mesh.set_points(points)
    #mesh.set_facets(facets)
    mesh.set_facets(facets, markers=markers.tolist())
    #mesh.set_holes(holes)

    return tet.build(mesh, options=options, max_volume=max_volume)
    # TODO : Improve/address conversion from meshpy data types to numpy
    # mesh = tet.build(mesh, options=options, max_volume=max_volume)
    # mesh.points = np.array(mesh.points)

if __name__ == '__main__':
    Lx, Ly, Lz, prefix = get_args(sys.argv)
    segments = load_segments_geomview(prefix)
    boundary_pslgs = build_boundary_PSLGs(segments, Lx, Ly, Lz)
    # plot_points_edges(*boundary_pslgs[1][:2])
    output_boundaries_poly(boundary_pslgs)
    boundaries = triangulate_PSLGs(boundary_pslgs)
    output_boundaries_geomview(boundaries)
    mesh = build_tet_mesh(segments, boundaries, Lx, Ly, Lz)
    #mesh.write_vtk('mesh.vtk')
