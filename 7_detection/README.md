# Stage 7 — Detection dataset assembly + training

Converts BlendCon output (stage 6) into a YOLO training-ready dataset in two
steps, handled by `blendcon_to_yolo.py`:

1. **Label derivation** — each BlendCon scenario folder (identified by its
   `Joint_Tracker.pickle`) gets a `2DBBs/` subfolder with one YOLO .txt per
   rendered frame. 2D boxes come from the pickle's per-frame `BB2D` entries:
   boxes completely outside the frame are skipped, the rest are clamped to
   the image bounds and normalized to `0 cx cy w h` (class 0 = worker) —
   the same protocol used for the paper's datasets.
2. **Dataset assembly** — images + labels from all scenarios are merged into
   one YOLO tree with a deterministic shuffled train/val split and a single
   `dataset.yaml`:

```
<out>/
├── images/{train,val}/<scenario>__<image>.jpg
├── labels/{train,val}/<scenario>__<image>.txt
└── dataset.yaml            # nc: 1, names: {0: worker}
```

## Usage

```bash
python blendcon_to_yolo.py \
    --runs path/to/Dataset/large1_5 [more roots or scenario folders ...] \
    --out datasets/mysite_synth --train-ratio 0.9 --seed 0 [--link]
```

`--runs` accepts BlendCon Dataset roots and/or individual scenario folders
(discovered recursively). `--link` symlinks instead of copying.
[`dataset_template.yaml`](dataset_template.yaml) documents the YAML layout
the script generates.

## Training (paper recipe)

Detectors are trained with [YOLOv10](https://github.com/THU-MIG/yolov10)
(AGPL-3.0, installed separately):

```bash
yolo detect train data=<out>/dataset.yaml model=yolov10s.yaml \
    epochs=800 batch=64 patience=25 imgsz=640 lr0=0.001 lrf=0.01
```

For the paper's incremental-augmentation protocol, merge the synthetic
images/labels with a real dataset by listing both in the training YAML (or
concatenating the folders) — see `exp_setup.py` in the paper's experiment
records.

The 29 trained checkpoints from the paper (real-only baseline + 2.5k→17.5k
synthetic increments × {3D-recon, LiDAR} × {small, large site}) are public at
[`SinaDavari/pass-worker-detection-yolov10s`](https://huggingface.co/SinaDavari/pass-worker-detection-yolov10s).
