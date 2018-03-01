
import ctypes
import os

import numpy as np
from numpy import ctypeslib as ctl

libwrapmfh5 = ctl.load_library(
    'libwrapmfh5.so', os.path.join(os.path.dirname(__file__), 'src')
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

libwrapmfh5.writeMeshMultiFlowH5.argtypes = [
    ctypes.c_char_p,
    Float64ArrayType,
    Int32ArrayType,
    Int32ArrayType,
    Int32ArrayType,
    Int32ArrayType,
    Int32ArrayType,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def write_mfh5(fname, mesh):

    points, elements, faces, markers, neighbours, adjacent_elements =\
        np.array(mesh.points), np.array(mesh.elements), np.array(mesh.faces),\
        np.array(mesh.face_markers), np.array(mesh.neighbors),\
        np.array(mesh.adjacent_elements)

    libwrapmfh5.writeMeshMultiFlowH5(
        ctypes.c_char_p(fname.encode('utf-8')),
        np.ascontiguousarray(points, np.float64),
        np.ascontiguousarray(elements, np.int32),
        np.ascontiguousarray(neighbours, np.int32),
        np.ascontiguousarray(faces, np.int32),
        np.ascontiguousarray(markers, np.int32),
        np.ascontiguousarray(adjacent_elements, np.int32),
        len(points),
        len(elements),
        len(faces),
    )
