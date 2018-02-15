from distutils.core import setup
from setuptools import find_packages

setup(
    name='mesh_sphere_packing',
    version='0.3',
    description='Mesh generator for producing tetrahedral meshes with periodic '\
        'boundaries from packings of spheres.',
    packages=find_packages(),
    install_requires=[
        'h5py>=2.7.1',
        'meshpy>=2016.1.2',
        'numpy>=1.14.0',
        'pyvtk>=0.5.18',
        'pyyaml>=3.12',
        'scipy>=1.0.0',
    ],
    author='Chris Knight',
    author_email='chrisk314@gmail.com',
)
