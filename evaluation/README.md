# Evaluation — reconstruction quality vs. LiDAR reference

> **Not part of the deployment pipeline.** LiDAR scans (Leica BLK ARC) serve
> **only as evaluation references** for the paper's comparison of the three
> image-based reconstruction methods — Depth Anything 3 (deployment) vs.
> VGGT and π³. Both scripts run fully headless (Open3D + SciPy, no Blender).

## 1. `align_meshes.py` — register reconstruction to LiDAR

Replaces the manual Blender overlay with two-step ICP:

1. **7-DoF point-to-point ICP with scaling** (max corr. 1.0 m) — recovers the
   reconstruction's unknown scale together with the coarse pose;
2. **rigid point-to-plane refinement** (max corr. 0.2 m).

```bash
python align_meshes.py --pred recon/mysite_mesh.ply --gt lidar/mysite.ply \
    --out-matrix eval/mysite_T.json --out-aligned eval/mysite_aligned.ply
```

The JSON stores the total 4×4 matrix plus a translation / Euler-XYZ /
scale decomposition (handy for visual inspection in Blender) and the ICP
fitness/RMSE diagnostics.

## 2. `compute_metrics.py` — quality metrics

| Metric | Definition |
|---|---|
| Accuracy | mean NN distance, pred → gt |
| Completeness | mean NN distance, gt → pred |
| Chamfer (L1) | mean of Accuracy and Completeness |
| Normal Consistency | mean \|n_pred · n_gt\| over pred→gt matches |
| Precision / Recall / F-score | fraction of matches within `--threshold` (default **5 cm**) |

```bash
python compute_metrics.py --pred eval/mysite_aligned.ply --gt lidar/mysite.ply \
    --csv eval/recon_metrics.csv --tag mysite_da3
```

### Reproducibility note

By default the metrics are computed over the mesh **vertices** — the exact
protocol behind the published tables (a headless port of the Blender-terminal
script used for the paper). Vertex-based metrics depend on tessellation
density; `--sample N` switches to uniform surface sampling (N points per
surface), the density-independent variant. Use vertex mode to reproduce the
paper.
