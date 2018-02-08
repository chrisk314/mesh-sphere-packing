
from unittest.mock import Mock, patch

import numpy as np

from mesh_sphere_packing.splitsphere import Domain
from mesh_sphere_packing.build import build

domain = Domain([2., 2., 2.], [True, True, True])

particles = np.array([[0., 0., 0., 0., 0.5]])

config = Mock(
    particle_file=None,
    allow_overlaps=False,
    tetgen_rad_edge_ratio=1.4,
    tetgen_min_angle=18.,
    tetgen_max_volume=1.0e-03,
    segment_length=1.0e-01,
    output_format=[]
)


@patch('mesh_sphere_packing.tetmesh.redirect_tetgen_output', autospec=True)
def test_build(patch):
    build(domain, particles, config)
