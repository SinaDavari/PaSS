# Stage 3 — Postprocess: point cloud → clean colored mesh

Headless PyMeshLab automation of the manual MeshLab workflow used in the
paper. Input: the raw colored point cloud from stage 2. Output: a clean,
colored, watertight mesh whose bounding-box **min corner sits at (0, 0, 0)**
(BlendCon's scene convention; stage 5 later refines Z using the true floor
plane).

## Steps (with paper parameters)

| # | Operation | Parameters |
|---|---|---|
| 1 | Point-cloud outlier removal | MeshLab defaults (k=32, probability=0.8) |
| 2 | Normal estimation | neighbours **30**, smooth iterations **10** |
| 3 | Screened Poisson reconstruction | depth **12**, min samples **1**, interpolation weight **10** |
| 4 | Trim Poisson bulges | remove the bottom **5%** Poisson-density vertices — screened Poisson closes open scan boundaries with low-density "balloon" bubbles that would otherwise inflate the bbox and corrupt stage 4's height-based scaling (`--density-trim-pct`, 0 = off) |
| 5 | Vertex-color transfer | cloud → Poisson mesh |
| 6 | Remove isolated pieces | MeshLab default (diameter < 10% of bbox diagonal) |
| 7 | Anchor to origin | translate bbox min corner to (0, 0, 0), transform frozen |

## Usage

```bash
python clean_and_mesh.py --input recon/mysite_pointcloud.ply \
                         --output recon/mysite_mesh.ply
```

All parameters are exposed as flags (`--poisson-depth`, `--normal-neighbors`,
…) with the paper values as defaults.

Requires `pymeshlab >= 2023.12` (older releases work too — the script handles
the `Percentage`/`PercentageValue` API rename).
