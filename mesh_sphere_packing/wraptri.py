
import ctypes
import os

import numpy as np
from numpy import ctypeslib as ctl

libwraptri = ctl.load_library(
    'libwraptri.so', os.path.join(os.path.dirname(__file__), 'src')
)


def wrapped_ndptr(*args, **kwargs):
    base = ctl.ndpointer(*args, **kwargs)

    def from_param(cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(
        base.__name__, (base,),
        {'from_param': classmethod(from_param)}
    )


Float64ArrayType = wrapped_ndptr(dtype=np.float64, flags='aligned, c_contiguous')
Int32ArrayType = wrapped_ndptr(dtype=np.int32, flags='aligned, c_contiguous')

libwraptri.wrap_tri.argtypes = [
    ctypes.c_char_p,
    Float64ArrayType,
    Float64ArrayType,
    ctypes.c_int,
    ctypes.c_int,
    Int32ArrayType,
    Float64ArrayType,
    ctypes.c_int,
    Float64ArrayType,
    ctypes.c_int,
]


def triangulate(points, point_markers, options="p", point_attributes=None,\
        holes=None, regions=None):

    libwraptri.wrap_tri(
        ctypes.c_char_p(options.encode('utf-8')),
        points,
        point_attributes,
        points.shape[0],
        point_attributes.shape[1] if np.any(point_attributes) else 0,
        point_markers,
        holes,
        holes.shape[0] if np.any(holes) else 0,
        regions,
        regions.shape[0] if np.any(regions) else 0
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
