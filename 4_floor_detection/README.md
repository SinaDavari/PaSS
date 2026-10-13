# Stage 4 — Floor detection + scene canonicalization

Imports the cleaned mesh into a fresh Blender scene, finds the floor, levels
the scene and puts the floor plane exactly at **Z = 0**. Runs in headless
Blender — no GUI, no interaction.

Leveling comes **before** metric scaling (stage 5): the scale factor is
derived from the site height measured normal to the floor, which is only well
defined once the scene is level.

## Method

| # | Step | Detail |
|---|---|---|
| 0 | Provisional scale | the mesh is still in reconstruction units, so the metric thresholds below would be meaningless; the factor `site height / raw Z extent` brings it to approximately metric units. Stage 5 supersedes it with the exact factor measured normal to the floor |
| 1 | Bottom crop | keep vertices in the bottom **⅓** of the Z range (assumes a roughly upright reconstruction — true for DA3 outputs of handheld video) |
| 2 | RANSAC plane | dominant plane in the crop, inlier band **±10 cm**; least-squares (SVD) refit on the inliers |
| 3 | Level | rotate the whole mesh so the floor normal aligns with **+Z** — also corrects small reconstruction tilt, which a bbox-based approach cannot |
| 4 | Drop | translate so the fitted **plane** (not min-Z — noise below the floor would otherwise float the real floor above zero) sits exactly at **Z = 0** |
| 5 | Re-center | translate X/Y so the **center of the floor footprint** sits at **(0, 0)** — the site is centered on the origin for BlendCon, with Z owned by the floor |

The floor inliers are stored on the mesh as the vertex group `FloorInliers`;
stage 5 builds the `StandableFloor` from exactly these points once the scene
is metric.

The import also attaches a **vertex-color material** (Color Attribute →
Principled BSDF, roughness 1): an imported PLY has no material, so without
this the scene would render plain white in stage 6 even though the vertex
colors are present.

## Usage

```bash
# deployment: the measured floor-to-ceiling height (e.g. 3.2 m) sets the provisional scale
blender --background --python detect_floor.py -- \
    --mesh recon/mysite_mesh.ply --site-height 3.2 --output scene/mysite_leveled.blend

# evaluation only: provisional scale from the LiDAR reference's Z extent
blender --background --python detect_floor.py -- \
    --mesh recon/mysite_mesh.ply --lidar-ref lidar/mysite_blk.ply \
    --output scene/mysite_leveled.blend
```

| Option | Default | |
|---|---|---|
| `--site-height` / `--lidar-ref` | required (one of) | provisional target height |
| `--crop-fraction` | 0.333 | bottom Z fraction searched for the floor |
| `--tolerance` | 0.10 m | RANSAC inlier band (±) |
| `--iterations` / `--seed` | 2000 / 0 | RANSAC settings (deterministic given a seed) |

Outputs: the leveled `.blend` (scene mesh with the `FloorInliers` vertex
group, floor at Z = 0) and a `.floor.json` report (provisional factor, plane
parameters, corrected tilt, inlier counts).

Tested with Blender 3.6 LTS and 4.x (the PLY importer rename is handled).
