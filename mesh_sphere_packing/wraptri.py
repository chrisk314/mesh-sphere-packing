
import ctypes
import numpy as np
from numpy import ctypeslib as ctl

libname = 'libwraptri.so'
libdir = './src'
lib = ctl.load_library(libname, libdir)

lib.wrap_tri.argtypes = [
    ctypes.c_char_p,
    ctl.ndpointer(np.float64, flags='aligned, c_contiguous'),
    ctl.ndpointer(np.float64, flags='aligned, c_contiguous'),
    ctypes.c_int,
    ctypes.c_int,
    ctl.ndpointer(np.int32, flags='aligned, c_contiguous'),
    ctl.ndpointer(np.float64, flags='aligned, c_contiguous'),
    ctypes.c_int,
]


def triangulate(points, point_markers, options="p", point_attributes=None,\
        regions=None):
    lib.wrap_tri(
        ctypes.c_char_p(options.encode('utf-8')),
        points,
        point_attributes,
        points.shape[0],
        point_attributes.shape[1],
        point_markers,
        regions,
        regions.shape[0]
    )


if __name__ == '__main__':
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 10.0],
        [0.0, 10.0]
    ], dtype=np.float64)
    point_attributes = np.array([
        [0.0],
        [1.0],
        [11.0],
        [10.0]
    ], dtype=np.float64)
    point_markers = np.array([
        0, 2, 0, 0
    ], dtype=np.int32)
    regions = np.array([
        [0.5, 5.0, 7.0, 0.1]
    ], dtype=np.float64)
    options = "pczAevn"
    triangulate(
        points,
        point_markers,
        point_attributes=point_attributes,
        regions=regions,
        options=options
    )
