# Glass-Panel-Generator

Utility program for generating hand blown glass panels (bubbles and chords).

This project can generate a complete old glass panel model from two provided heighmaps.
The full model is composed of elevation data, bubble distribution and refractive index field:
- The surface elevation profile of the front and back face of the glass panel are defined by 2 floating point greyscale images (`.exr` files) and should be provided by the user. A few examples are provided in  the `./heighmaps/` folder, but you can create your own using whatever procedural noise or elevation data you like (or see `crown_heighfield_generator.py` for a simple crown glass generator).
- The discrete bubble distribution is represented by a voxel grid. Each voxel contains (at most) one bubble, and one bubble can overlap multiple cells. The voxel grid is then encoded in a `.bubble` file where each voxel contains a tuple `(pos.x, pos.y, pos.z, radius)` representing the coordinates (in panel space) and the radius of the bubble contained inside the voxel.
- The continuous Refractive Index Field (RIF) is represented by an another voxel grid encoded in a `.vol` file (see [mitsuba documentation](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_volumes.html#volume-gridvolume) for more details).

The bubble voxel grid is generated by the `bubble_generator.py` script.
The RIF is generated by the `RIF_generator.py` and  `filament_sdf.py` scripts.

![Zoom on a window made hand blown glass showing a few bubbles.](https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/Cordes_caustique.jpg)
*Zoom on a window made hand blown glass showing a few bubbles.*

# Usage
You can check the `generate_panels.py` script for a few usage examples.
These panels where used in most of the scenes in the thesis manuscript.

Given the panel shape and surface, the script modifies the provided heightmaps to account for cross interactions between the bubbles / chords and the surface (e.g. a bubble that is close to the surface will create a little bump on the surface). see `crosstalk.py` script (this may require some parameter tweaking though).

![https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/manchon_2_1024.jpg]
*Front heightmap before bubbles and RIF cross talk.*

![https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/heightmap_front.jpg]
*Front heightmap after bubbles and RIF cross talk.*

Accounting for all these effects is quite important to get a realistic looking glass and can be easily observed on real hand blown glass panels.

![Zoom on a window made hand blown glass showing a few chords.](https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/Manchon_Vernon.pngManchon_Vernon.jpg)
*Zoom on a window made hand blown glass showing a few chords. Chords are thin filaments of glass that have different optical properties than the main material. This results in continuous variations of the refractive index field inside the panel.*

![Render of a Tiffany lamp.](https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/Tiffany_ref.jpg)

*Render of a Tiffany lamp. The lamp glass facets are cut from the panel  `./panels/manchon_2_exagerated_e0.03_full*

![Zoom a glass facet of the Tiffany lamp.](https://github.com/Phd-Antic-Glass/Glass-Panel-Generator/blob/main/images/Tiffany_ref_zoom_A.jpg)

*Zoom on a glass facet showing bubbles and chords (pathtracing rendered using mitsuba3_old_glass renderer)*

# DEPENDENCIES:
Numpy, opencv, scipy, matplotlib, perlin_numpy, drjit

For installing opencv, you can use:
`pip3 install opencv-python`

For perlin_numpy:
`pip3 install git+https://github.com/pvigier/perlin-numpy`
(see https://github.com/pvigier/perlin-numpy)

For drjit, we use v0.4.6 (see [drjit](https://drjit.readthedocs.io/en/v0.4.6/))

