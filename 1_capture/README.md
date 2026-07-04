# Stage 1 — Capture: video → overlapping frame set

Extracts a fixed number of **sharp, temporally uniform** frames from a short
smartphone walk-through video, dropping frames that are too blurry for
image-based 3D reconstruction (motion blur is the dominant failure mode of
handheld capture).

## Method

1. The video is split into `--num-frames` equal temporal windows
   (e.g. a 100 s @ 30 fps video with the default 50 windows → 60 candidate
   frames per window).
2. Every candidate frame is scored with the **variance of the Laplacian**
   (computed on a downscaled grayscale copy for speed and
   resolution-independence). Higher = sharper.
3. The sharpest frame of each window is kept; if even the best frame of a
   window scores below `--min-sharpness`, the window is **dropped** — a
   missing frame hurts reconstruction less than a blurry one.

Uniform windows guarantee the overlap between consecutive kept frames that
multi-view reconstruction needs; blur filtering guarantees per-frame quality.

## Usage

```bash
python sample_frames.py --video site.mp4 --output-dir frames/ --num-frames 50
```

| Option | Default | Meaning |
|---|---|---|
| `--num-frames` | 50 | target frame count / number of temporal windows |
| `--min-sharpness` | 60.0 | variance-of-Laplacian cutoff (tune per camera; inspect the printed scores) |
| `--stride` | 1 | score every Nth candidate within a window (speedup for long videos) |
| `--ext` | jpg | output format |

Output: `frames/frame_0000.jpg …`, ready for stage 2.
