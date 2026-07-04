# Stage 2 — Image-based 3D reconstruction

> **Depth Anything 3 (DA3) is the deployment method of the PASS pipeline.**
> The VGGT and π³ (Pi3) wrappers in [`baselines/`](baselines/) exist **only**
> for the paper's reconstruction-quality comparison (see
> [`../evaluation/`](../evaluation/)) and are **not** part of the deployment
> pipeline.

All three wrappers share one output contract so downstream stages never care
which backend produced the cloud:

```
<output-dir>/<site-name>_pointcloud.ply     # colored point cloud
```

The DA3 wrapper additionally guarantees **Z-up orientation**: DA3's GLB
export uses the glTF convention (Y-up, oriented by the first camera), and the
wrapper rotates it to Z-up on extraction — residual tilt from the phone's
pitch is corrected by stage 5's RANSAC leveling.

## Deployment: Depth Anything 3

Install DA3 in its own environment (Apache-2.0):

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3
cd Depth-Anything-3 && pip install -e .
```

Run:

```bash
python run_da3.py --input frames/ --output-dir recon/ --site-name mysite
```

The wrapper calls the official `da3 auto` CLI with GLB export (DA3's native
point-cloud output with adaptive confidence filtering; plain `ply` is not a
supported `da3` export format despite older docstrings), extracts the point
cloud (camera wireframes skipped, vertex colors preserved), applies the
glTF→Z-up rotation, and writes the standard name. Useful flags:
`--num-max-points` (default 1,000,000), `--process-res`, `--keep-gltf-axes`;
anything else can be forwarded verbatim with `--da3-args ...`.

## Evaluation baselines (comparison only)

| Backend | Install from | License | Wrapper |
|---|---|---|---|
| VGGT | https://github.com/facebookresearch/vggt | VGGT (non-commercial) | `baselines/run_vggt.py --vggt-root <clone>` |
| π³ (Pi3) | https://github.com/yyfz/Pi3 | non-commercial | `baselines/run_pi3.py --pi3-root <clone>` |

Neither repo is vendored here (license + hygiene); the wrappers import them
from a local clone (`--*-root` flag or `$VGGT_ROOT` / `$PI3_ROOT`). Inference
follows each project's official example: Pi3 runs a single full pass with the
official confidence + depth-edge masking; VGGT exports world points with
percentile-based confidence culling (default: drop the bottom 5%).

Pin the upstream commits you used in your lab notebook / paper for exact
reproducibility — all three projects evolve quickly.
