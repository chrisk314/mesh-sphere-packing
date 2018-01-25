
from unittest.mock import Mock, PropertyMock

from mesh_sphere_packing.build import build

args = Mock(
    particle_center=[0., 0., 0.],
    particle_radius=.5,
    domain_dimensions=[2., 2., 2.]
)


def test_build():
    build(args)
