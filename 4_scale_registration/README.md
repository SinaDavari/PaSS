# Stage 4 — Scale registration: mesh → metric `.blend`

Imports the cleaned mesh into a fresh Blender scene, scales it to metric
units, and saves the `.blend` file for the following stages.

## Why a measurement is required

Metric scale is **fundamentally unobservable** from monocular imagery — no
image-based reconstruction (nor any 3D model derived from it) can recover
absolute scale on its own. PASS therefore takes exactly **one** real-world
measurement as an input, supplied up front on the command line. There are no
interactive prompts; the pipeline stays end-to-end.

Two scale sources (mutually exclusive):

| Flag | Source | Role |
|---|---|---|
| `--site-height <m>` | total site height measured on-site with a tape | **deployment path** |
| `--lidar-ref <ply>` | Z extent of a LiDAR scan of the same site | evaluation-only convenience — LiDAR is never a deployment input |

The mesh is scaled **uniformly about the world origin**, which preserves the
bbox-corner-at-(0,0,0) convention established in stage 3. The applied factor
is recorded in a `.scale.json` sidecar for provenance.

The import also attaches a **vertex-color material** (Color Attribute →
Principled BSDF, roughness 1): an imported PLY has no material, so without
this the scene would render plain white in stage 6 even though the vertex
colors are present.

## Usage

```bash
# deployment: tape measurement (e.g. floor-to-ceiling 3.2 m)
blender --background --python scale_to_metric.py -- \
    --mesh recon/mysite_mesh.ply --site-height 3.2 --output scene/mysite.blend

# evaluation only: scale from the LiDAR reference
blender --background --python scale_to_metric.py -- \
    --mesh recon/mysite_mesh.ply --lidar-ref lidar/mysite_blk.ply \
    --output scene/mysite.blend
```

Tested with Blender 3.6 LTS and 4.x (the PLY importer rename is handled).
