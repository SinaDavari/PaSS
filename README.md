# PASS: Passive-sensing Site Synthesis

**Site-adaptive synthetic training data generation for construction worker
detection, from nothing but a smartphone video.**

Code repository for the paper *"PASS: Passive-sensing Site Synthesis"*
(submitted to *Automation in Construction*).

PASS turns a short walk-through video of a construction site into a fully
annotated, site-specific synthetic training dataset — and from it, a
site-adapted YOLOv10-s worker detector:

```
Short smartphone video
        │  1_capture            sharp, overlapping frame set
        ▼
Image-based 3D reconstruction (Depth Anything 3)
        │  2_reconstruction     colored point cloud (.ply)
        ▼
Point-cloud cleaning + Screened-Poisson meshing (PyMeshLab)
        │  3_postprocess        watertight colored mesh, corner at (0,0,0)
        ▼
Metric scale recovery (one on-site tape measurement)
        │  4_scale_registration metric .blend scene
        ▼
Floor detection + scene canonicalization (RANSAC, headless Blender)
        │  5_floor_detection    floor levelled to Z=0 + StandableFloor mesh
        ▼
Synthetic image generation (modified BlendCon)
        │  6_blendcon           rendered images + labels
        ▼
YOLO dataset assembly + detector training
           7_detection          site-adapted YOLOv10-s
```

## Repository layout

| Stage | Folder | What it does |
|---|---|---|
| 1 | [`1_capture/`](1_capture/) | Video → N sharpest, uniformly spread frames (blur-aware sampling) |
| 2 | [`2_reconstruction/`](2_reconstruction/) | Frames → colored point cloud. **Depth Anything 3 is the deployment method**; VGGT and π³ wrappers are provided for the paper's evaluation comparison only |
| 3 | [`3_postprocess/`](3_postprocess/) | Outlier removal → normals → Screened Poisson → colored mesh with bbox corner at the origin |
| 4 | [`4_scale_registration/`](4_scale_registration/) | Scale the mesh to metric units from a measured site height (or a LiDAR reference, evaluation only) and save as `.blend` |
| 5 | [`5_floor_detection/`](5_floor_detection/) | RANSAC floor plane → level scene, floor at Z=0 → inset → `StandableFloor` object |
| 6 | [`6_blendcon/`](6_blendcon/) | Our camera-placement patch for [BlendCon](https://github.com/Ali-Tohidifar/BlendCon) (patch + apply script, upstream not vendored) |
| 7 | [`7_detection/`](7_detection/) | BlendCon output → per-scenario 2D labels + merged YOLO dataset with a single `dataset.yaml` |
| — | [`evaluation/`](evaluation/) | **Study only, not part of the pipeline.** Mesh-vs-LiDAR alignment and Accuracy / Completeness / Chamfer / Normal-Consistency / F-score metrics |

## Models and data

- **Trained models** — all 29 trained YOLOv10-s detectors are available on
  Hugging Face:
  [`SinaDavari/pass-worker-detection-yolov10s`](https://huggingface.co/SinaDavari/pass-worker-detection-yolov10s).
- **Real test datasets** — the real, manually annotated test sets gathered
  from both construction sites and used to evaluate the models are available
  on Hugging Face:
  [`SinaDavari/pass-worker-detection-test-sets`](https://huggingface.co/datasets/SinaDavari/pass-worker-detection-test-sets).

## Design principles

- **Passive sensing only.** The deployment pipeline needs a smartphone video
  and a single tape measurement. LiDAR scans (Leica BLK ARC) are used **only
  as evaluation references**; never as a pipeline input.
- **One measurement, then end-to-end.** Metric scale is fundamentally
  unobservable from monocular imagery, so the user supplies one measured
  height (in meters) up front; everything else is automatic and
  non-interactive.
- **No heavy artifacts in this repo.** Trained models, dataset samples, and
  test sets live on Hugging Face:
  [`SinaDavari/pass-worker-detection-yolov10s`](https://huggingface.co/SinaDavari/pass-worker-detection-yolov10s)
  (29 YOLOv10-s checkpoints from the incremental-augmentation study).
- **Synthetic data samples**: 2,000 fully labeled sample images (500 per
  site × background source, with YOLO labels and segmentation masks) from the
  PASS-generated synthetic datasets are available on Hugging Face:
  [`SinaDavari/pass-synthetic-data-samples`](https://huggingface.co/datasets/SinaDavari/pass-synthetic-data-samples).

## Quickstart

```bash
# 0. environment (stages 1, 3, evaluation)
conda env create -f environment.yml && conda activate pass

# 1. frames from video
python 1_capture/sample_frames.py --video site.mp4 --output-dir frames/ --num-frames 50

# 2. 3D reconstruction (run inside the DA3 environment; needs the `da3` CLI
#    and trimesh — see 2_reconstruction/README.md for install)
python 2_reconstruction/run_da3.py --input frames/ --output-dir recon/ --site-name mysite

# 3. clean + mesh
python 3_postprocess/clean_and_mesh.py --input recon/mysite_pointcloud.ply --output recon/mysite_mesh.ply

# 4. metric scale (site height measured on-site with a tape, e.g. 3.2 m)
blender --background --python 4_scale_registration/scale_to_metric.py -- \
    --mesh recon/mysite_mesh.ply --site-height 3.2 --output scene/mysite.blend

# 5. floor detection + canonicalization
blender --background --python 5_floor_detection/detect_floor.py -- \
    --blend scene/mysite.blend --output scene/mysite_floor.blend

# 6. synthetic data generation (patched BlendCon; see 6_blendcon/README.md)
cd 6_blendcon && ./apply_patch.sh && cd BlendCon && blender --background --python DataGenerator.py

# 7. YOLO dataset assembly + training with https://github.com/THU-MIG/yolov10
python 7_detection/blendcon_to_yolo.py --runs blendcon/Dataset/* --out datasets/mysite_synth
```

Per-site parameters (paths, measured height, floor-detection settings) can be
kept in a config file — see [`configs/site_example.yaml`](configs/site_example.yaml).

## The study (paper experiments)

Two construction sites (small, large) were each reconstructed from smartphone
video (Depth Anything 3, with VGGT and π³ as comparisons) and scanned with a
Leica BLK ARC LiDAR. Reconstruction quality was measured against the LiDAR
reference (`evaluation/`). Synthetic images were generated on both the
image-based and the LiDAR meshes, and YOLOv10-s detectors were trained under
an incremental-augmentation protocol (real-only baseline + 2.5k→17.5k
synthetic increments per source), yielding 29 models — all public on
[Hugging Face](https://huggingface.co/SinaDavari/pass-worker-detection-yolov10s).

## Third-party code and licenses

This repo is MIT-licensed and vendors **no** third-party code. It interfaces
with, and you must install separately under their own licenses:

| Project | License | Role |
|---|---|---|
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Apache-2.0 | deployment reconstruction |
| [VGGT](https://github.com/facebookresearch/vggt) | VGGT license (non-commercial) | evaluation comparison |
| [π³ (Pi3)](https://github.com/yyfz/Pi3) | non-commercial | evaluation comparison |
| [BlendCon](https://github.com/Ali-Tohidifar/BlendCon) | MIT | synthetic image engine (we ship a patch only) |
| [YOLOv10](https://github.com/THU-MIG/yolov10) | AGPL-3.0 | downstream detector |

## Citation

See [`CITATION.cff`](CITATION.cff) — full reference will be added upon
acceptance.
