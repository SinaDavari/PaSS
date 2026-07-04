# Stage 5 — Floor detection + scene canonicalization

Automates the manual "make sure the floor sits at Z=0 on the X-Y plane" step
and produces the **`StandableFloor`** object that the patched BlendCon
(stage 6) uses to constrain worker/camera placement. Runs in headless Blender
— no GUI, no interaction.

## Method

| # | Step | Detail |
|---|---|---|
| 1 | Bottom crop | keep vertices in the bottom **⅓** of the Z range (assumes a roughly upright reconstruction — true for DA3 outputs of handheld video) |
| 2 | RANSAC plane | dominant plane in the crop, inlier band **±10 cm**; least-squares (SVD) refit on the inliers |
| 3 | Level | rotate the whole mesh so the floor normal aligns with **+Z** — also corrects small reconstruction tilt, which a bbox-based approach cannot |
| 4 | Drop | translate so the fitted **plane** (not min-Z — noise below the floor would otherwise float the real floor above zero) sits exactly at **Z = 0** |
| 5 | Re-center | translate X/Y so the **center of the floor footprint** sits at **(0, 0)** — the site is centered on the origin for BlendCon, with Z owned by the floor |
| 6 | StandableFloor | rasterize floor inliers into a **10 cm** occupancy grid, binary-erode by the **30 cm** inset, emit one quad per remaining cell as the `StandableFloor` mesh at Z = 0 |

The 30 cm inward inset (implemented as morphological erosion of the
footprint) keeps synthetic workers and cameras away from walls and from the
ragged reconstruction boundary.

## Usage

```bash
blender --background --python detect_floor.py -- \
    --blend scene/mysite.blend --output scene/mysite_floor.blend
```

| Option | Default | |
|---|---|---|
| `--crop-fraction` | 0.333 | bottom Z fraction searched for the floor |
| `--tolerance` | 0.10 m | RANSAC inlier band (±) |
| `--inset` | 0.30 m | inward erosion of the standable area |
| `--cell-size` | 0.10 m | occupancy-grid resolution |
| `--iterations` / `--seed` | 2000 / 0 | RANSAC settings (deterministic given a seed) |
| `--object` | largest mesh | scene mesh to canonicalize |

Outputs: the canonicalized `.blend` (scene mesh + `StandableFloor`) and a
`.floor.json` report (plane parameters, corrected tilt, inlier counts,
standable area in m²).
