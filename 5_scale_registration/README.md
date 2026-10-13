# Stage 5 — Metric-scale recovery + `StandableFloor`

Scales the leveled scene from stage 4 to metric units and produces the
**`StandableFloor`** object that the patched BlendCon (stage 6) uses to
constrain worker/camera placement. Runs in headless Blender.

## Why a measurement is required

Metric scale is **fundamentally unobservable** from monocular imagery — no
image-based reconstruction (nor any 3D model derived from it) can recover
absolute scale on its own. PASS therefore takes exactly **one** real-world
measurement as an input, supplied up front on the command line. There are no
interactive prompts; the pipeline stays end-to-end.

Two scale sources (mutually exclusive):

| Flag | Source | Role |
|---|---|---|
| `--site-height <m>` | floor-to-ceiling height measured on-site with a tape | **deployment path** |
| `--lidar-ref <ply>` | Z extent of a LiDAR scan of the same site | evaluation-only convenience — LiDAR is never a deployment input |

Because stage 4 leveled the scene and put the floor plane at Z = 0, the
reconstruction's height above the floor is simply its **maximum Z**. The
factor `s = target height / max Z` is applied **uniformly about the origin**,
which lies on the floor plane, so the floor stays at Z = 0. This exact factor
supersedes stage 4's provisional one; the `.scale.json` report records both
and their product (the total factor applied to the stage-3 mesh).

## StandableFloor

With the scene level and metric, the walkable region is extracted from the
floor inliers found in stage 4 (vertex group `FloorInliers`):

| # | Step | Detail |
|---|---|---|
| 1 | Rasterize | floor inliers → **10 cm** occupancy grid |
| 2 | Close | 1-cell morphological closing bridges point-sampling gaps |
| 3 | Inset | binary-erode by the **30 cm** inset, keeping workers and cameras away from walls and the ragged reconstruction boundary |
| 4 | Mesh | one quad per remaining cell as the `StandableFloor` mesh at Z = 0 |

## Usage

```bash
# deployment: tape measurement (e.g. floor-to-ceiling 3.2 m)
blender --background --python scale_to_metric.py -- \
    --blend scene/mysite_leveled.blend --site-height 3.2 --output scene/mysite.blend

# evaluation only: scale from the LiDAR reference
blender --background --python scale_to_metric.py -- \
    --blend scene/mysite_leveled.blend --lidar-ref lidar/mysite_blk.ply \
    --output scene/mysite.blend
```

| Option | Default | |
|---|---|---|
| `--inset` | 0.30 m | inward erosion of the standable area |
| `--cell-size` | 0.10 m | occupancy-grid resolution |
| `--tolerance` | 0.10 m | floor band used only if the stage-4 vertex group is missing |
| `--object` | largest mesh | scene mesh to scale |

Outputs: the metric `.blend` (scene mesh + `StandableFloor`) and a
`.scale.json` report (applied factors, standable area in m²).
