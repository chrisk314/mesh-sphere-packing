from distutils.core import setup
from setuptools import find_packages

setup(
    name='mesh_sphere_packing',
    version='0.1',
    description='Mesh generator for producing tetrahedral meshes with periodic '\
        'boundaries from packings of spheres.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'meshpy'
    ],
    author='Chris Knight',
    author_email='chrisk314@gmail.com',
)
