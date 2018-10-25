# mesh-sphere-packing
MeshSpherePacking is a mesh generator for producing quality tetrahedral meshes with periodic
boundaries from packings of spheres for use in finite volume simulations. The code was produced 
during my PhD research to enable highly resolved CFD simulations of fluid flow in the interstitial
spaces of densely packed, idealized sand grain packings. In order to apply periodic boundary
conditions (PBCs) along a given axis, some CFD packages require congruence of all boundary triangles 
at either end of the domain. I didn't find an existing package to meet my (very bespoke!)
requirements, so this imaginatively named project was born! 

Below is a rendering of a fluid velocity 
field at low Reynolds number using an MSP mesh with over 600 particles and about 8 million
tetrahedral elements. It took around 5 minutes to build the mesh on a single HPC node and another 5
minutes to output the mesh in `.vtk` format.

<p align="center">
<img alt="fluid velocity field" src="https://user-images.githubusercontent.com/2366658/47520008-e55b8e00-d886-11e8-9054-57382cdbd516.png" width="500">
</p>

The mesh generator produces tetrahedral meshes of the interstitial, or void, space between
collections of spheres which reside in a cuboid shaped spatial domain. Periodic boundaries can be
applied along any axis. Periodic boundaries are achieved by identifying intersections of the
particle geometry with any of the domain boundaries and splitting the particles into "pieces" along
the intersection planes. For an axis along which periodic boundaries are in effect, intersections
occur at the lower boundary with "ghost" particles which are duplicated from the upper boundary. The
domain boundaries are triangulated (just the lower boundaries in the case of PBCs) using Shewchuk's
[Triangle](https://www.cs.cmu.edu/~quake/triangle.html) library, taking into account any holes
produced by the intersections. Where PBCs are applied the triangulated lower boundaries are then
duplicated at the corresponding upper boundary, hence achieving the required congruency of boundary
triangles across the periodic boundary. A Piecewise Linear Complex is then formed of all the sphere
pieces and boundaries which is triangulated using [TetGen](http://wias-berlin.de/software/tetgen/).
The wrapper library [MeshPy](https://mathema.tician.de/software/meshpy/) provides the Python
interface to Triangle and TetGen.

### Usage
To install simply clone and pip install like so
```
git@github.com:chrisk314/mesh-sphere-packing.git
cd mesh-sphere-packing
pip install .
export PATH=${PATH}:${PWD}/bin
```

Particle coordinates and radii must be specified along with the domain side lengths and periodic
boundary flags in a file like below

```
# particles.msp
2.0e-03 2.0e-03 2.0e-03
1 1 1
00 1.0e-03 2.0e-04 3.0e-04 5.0e-04
01 5.0e-04 1.0e-03 1.5e-04 3.0e-04
02 1.2e-03 1.2e-03 1.2e-03 5.0e-04
03 1.5e-03 1.0e-04 1.5e-03 3.0e-04
04 5.0e-04 1.9e-03 8.0e-04 2.0e-04 
05 1.0e-04 3.0e-04 1.8e-03 4.0e-04
06 8.0e-04 8.0e-04 7.0e-04 2.0e-04
07 1.3e-03 1.3e-03 2.0e-04 4.0e-04 
08 1.9e-03 1.5e-03 1.6e-03 3.0e-04
09 1.8e-03 6.0e-04 8.0e-04 4.0e-04
10 2.2e-04 1.5e-03 6.0e-04 2.0e-04
11 5.0e-04 1.5e-03 1.8e-03 3.0e-04
```

This example specifies 12 particles in a 2mm x 2mm x 2mm domain and periodic boundaries along the x,
y, and z axes. Here's an image of this particle configuration

<p align="center">
<img alt="12 spheres in a box" src="https://user-images.githubusercontent.com/2366658/47515711-7fb5d480-d87b-11e8-9419-696177404d36.png" width="500">
</p>

In order to control the sphere refinement process and the parameters used by TetGen during the 
tetrahedral mesh generation a .yaml config file should be provided as below

```YAML
# config.yml
allow_overlaps: false
duplicate_particles: [false, false, false]
output_format: [vtk]
particle_file: particles.msp
segment_length: 3.3e-05       # Controls sphere refinement
tetgen_max_volume: 5.0e-13    # Controls tetrahedron volume
tetgen_min_angle: 18.0        # Minimum allowed face angle for tetrahedra
tetgen_rad_edge_ratio: 1.4    # Target radius:edge ratio for tetrahedra
```

Default values are specified by the program but as the mesh quality is dependent on both the 
length scale of the input geometry and the characteristic edge length scale specified by
`segment_length`, these values should be set in a config file on a case by case basis.

The build process is initiated using a config file `config.yml` and a particle file `particles.msp`, 
such as those provided in the [example](./example) directory, like so
```bash
mspbuild -c config.yml -p particles.msp
```

This produces the below log output on stdout which is also written to `msp.log`,
additionally the stdout from TetGen is redirected to `tet.out`.
```
MSP-Build  INFO     25-10-18 17:08:02.839  :    Reading program inputs
MSP-Build  INFO     25-10-18 17:08:02.844  :    Starting mesh build process
MSP-Build  INFO     25-10-18 17:08:02.844  :    Splitting input particles
MSP-Build  INFO     25-10-18 17:08:03.293  :    Triangulating domain boundaries
MSP-Build  INFO     25-10-18 17:08:06.518  :    Building tetrahedral mesh
MSP-Build  INFO     25-10-18 17:08:06.519  :        -> building vertex list...
MSP-Build  INFO     25-10-18 17:08:07.616  :        -> calling TetGen (writing log to ./tet.log)
MSP-Build  INFO     25-10-18 17:08:14.799  :    Built mesh with 64944 points, 316378 tetrahedra, 659834 faces, and 408396 edges
MSP-Build  INFO     25-10-18 17:08:14.807  :    Completed mesh build
MSP-Build  INFO     25-10-18 17:08:14.807  :    Outputting mesh in formats: vtk
MSP-Build  INFO     25-10-18 17:08:23.414  :    Finished
```

Output was specified in `.vtk` format in the config file so an output file
`particles,a_5.00e-13,s_3.30e-05.vtk` containing the mesh geometry and topology is also created.
It's possible to specify multiple output formats. The currently supported output formats are:
Polygon [`.ply`](http://paulbourke.net/dataformats/ply/), Geomview
[`.msh`](http://gmsh.info/doc/texinfo/gmsh.html#File-formats), TetGen
[`.poly`](http://wias-berlin.de/software/tetgen/fformats.poly.html), legacy VTK
[`.vtk`](https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf), MultiFlow `.h5`. The
output `.vtk` mesh can be viewed in ParaView and is shown below

<p align="center">
<img alt="periodic tetrahedral mesh" src="https://user-images.githubusercontent.com/2366658/47516181-b9d3a600-d87c-11e8-99df-ef54bb42c88e.png" width="500">
</p>
