# Evaluation — reconstruction quality vs. LiDAR reference

> **Not part of the deployment pipeline.** LiDAR scans (Leica BLK ARC) serve
> **only as evaluation references** for the paper's comparison of the three
> image-based reconstruction methods — Depth Anything 3 (deployment) vs.
> VGGT and π³.

## One-command evaluation (recommended): `evaluate_vs_lidar.py`

Evaluates a reconstruction against a LiDAR scan when both live in `.blend`
files (e.g. Blender scene files), replicating the paper's manual workflow
end-to-end — the original `.blend` files are **never modified**:

1. open the LiDAR ground-truth blend (headless Blender)
2. append every object from the reconstruction blend (automated copy-paste)
3. export the two main meshes as world-space point clouds
4. register reconstruction → LiDAR (see `align_meshes.py` below)
5. compute the metrics (see `compute_metrics.py` below)
6. save `<tag>_overlay.blend` — both meshes laid on top of each other —
   for visual inspection of the alignment

```bash
# run with a Python that has open3d + scipy; Blender is called internally
python evaluate_vs_lidar.py \
    --gt-blend lidar_scan.blend --pred-blend recon.blend \
    --gt-object <lidar mesh name> --pred-object <recon mesh name> \
    --workdir out/site_method --tag site_method
```

By default the mesh with the most vertices on each side is evaluated
(`--gt-object` / `--pred-object` override). Outputs in `--workdir`:
`<tag>_T.json` (transform + registration diagnostics), `metrics.csv`,
`<tag>_overlay.blend` (review file).

## Building blocks (standalone, for plain PLY inputs)

### `align_meshes.py` — register reconstruction to LiDAR

Automates the manual "lay the meshes on top of each other" step. Two
candidate initializations — identity, and FPFH-feature RANSAC global
registration — are each refined with **7-DoF point-to-point ICP with
scaling** followed by **rigid point-to-plane ICP**; the branch with the best
final fitness wins. Registration runs on voxel-downsampled clouds
(`--voxel`, default 5 cm); the resulting transform applies to full
resolution. Assumes roughly metric inputs.

```bash
python align_meshes.py --pred recon.ply --gt lidar.ply \
    --out-matrix T.json [--out-aligned recon_aligned.ply]
```

### `compute_metrics.py` — quality metrics

| Metric | Definition |
|---|---|
| Accuracy | mean NN distance, pred → gt |
| Completeness | mean NN distance, gt → pred |
| Chamfer (L1) | mean of Accuracy and Completeness |
| Normal Consistency | mean \|n_pred · n_gt\| over pred→gt matches |
| Precision / Recall / F-score | fraction of matches within `--threshold` (default **5 cm**) |

```bash
python compute_metrics.py --pred recon.ply --gt lidar.ply \
    [--matrix T.json] --csv metrics.csv --tag site_method
```

### Reproducibility note

By default the metrics are computed over the mesh **vertices** — the exact
protocol behind the published tables. Vertex-based metrics depend on
tessellation density; `--sample N` switches to uniform surface sampling
(N points per surface), the density-independent variant. Use vertex mode to
reproduce the paper.
