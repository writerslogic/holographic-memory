CRYSTAL LOGO WITH A CRISP WHITE HALO

The detailed crystal retains its fine dark ink, internal connections, node
dots and hatch accents. A narrow white keyline follows every stroke and
node, making the same artwork legible on white or black backgrounds.
The open facets and surrounding canvas are transparent.

All production artwork files in this package have transparent backgrounds.
The animation makes one continuous 360-degree turn every ten seconds.

FILES

crystal-detailed-loop.svg
Animated vector. Transparent background. Scales to different sizes and
repeats indefinitely. The halo is part of the artwork; no external CSS,
JavaScript, fonts or image files are required.

crystal-detailed-loop.webp
Lossless animated image with full alpha transparency. 768 x 768 pixels,
25 frames per second, 250 frames, infinite looping.

crystal-detailed-loop.apng
Lossless animated PNG with full alpha transparency. Same dimensions,
frame rate, duration and infinite looping as the WebP.

crystal-detailed-static.svg
Editable transparent vector still of the opening pose.

crystal-detailed-static.png
1536 x 1536 pixel transparent PNG of the opening pose.

SOURCE

source/build_crystal_halo.py
The complete editable geometry, hand-drawn strokes, halo, 3D motion and
SVG/WebP/PNG exporter. Requires Python, NumPy, SciPy, Matplotlib, and
Pillow with WebP support. Run --samples for design previews or --full
for the complete RGBA master, animated SVG, and animated WebP.

source/crystal-halo-geometry.json
Geometry and animation specifications.

source/build_additional_exports.py
Creates the transparent APNG and a separate comparison preview from the
RGBA master. Requires NumPy, Pillow and ffmpeg.

The white/black comparison preview shown in chat uses test backgrounds
only. Those backgrounds are not present in these production artwork files.
