#!/usr/bin/env python3
"""
PASS stage 1 — capture: video -> overlapping, blur-filtered frame set.

Splits the video into `--num-frames` uniform temporal windows, scores every
candidate frame in each window with the variance of the Laplacian (a standard
sharpness measure), and keeps the sharpest frame per window. Windows whose
best frame is still below `--min-sharpness` are dropped entirely, so the
output may contain fewer than `--num-frames` images (better to lose a frame
than to feed a blurry one to the reconstruction).

Example (100 s @ 30 fps video -> ~50 sharp frames):
    python sample_frames.py --video site.mp4 --output-dir frames/ --num-frames 50
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Frames are downscaled to this long-side size before scoring so that the
# sharpness measure is comparable across resolutions and fast to compute.
SCORE_LONG_SIDE = 960


def sharpness_score(frame_bgr: np.ndarray) -> float:
    """Variance of the Laplacian on a downscaled grayscale copy."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = SCORE_LONG_SIDE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sample_frames(video: Path, output_dir: Path, num_frames: int,
                  min_sharpness: float, stride: int, ext: str,
                  prefix: str) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if total <= 0:
        sys.exit(f"ERROR: video reports no frames: {video}")
    print(f"Video: {video.name} | {total} frames @ {fps:.1f} fps "
          f"(~{total / fps:.1f} s)" if fps else f"Video: {video.name} | {total} frames")

    # Uniform temporal windows; the sharpest candidate in each window wins.
    bounds = np.linspace(0, total, num_frames + 1, dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped = 0, 0
    frame_idx = 0
    for wi in range(num_frames):
        start, end = bounds[wi], bounds[wi + 1]
        if end <= start:
            continue

        best_score, best_frame, best_idx = -1.0, None, -1
        # Sequential read is much faster than per-frame seeking.
        while frame_idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame_idx - start) % stride == 0 and frame_idx >= start:
                score = sharpness_score(frame)
                if score > best_score:
                    best_score, best_frame, best_idx = score, frame, frame_idx
            frame_idx += 1

        if best_frame is None:
            dropped += 1
            continue
        if best_score < min_sharpness:
            print(f"  window {wi + 1:3d}: DROPPED (best sharpness "
                  f"{best_score:.1f} < {min_sharpness:.1f})")
            dropped += 1
            continue

        out = output_dir / f"{prefix}{kept:04d}.{ext}"
        cv2.imwrite(str(out), best_frame)
        kept += 1
        print(f"  window {wi + 1:3d}: frame {best_idx:6d} "
              f"(sharpness {best_score:7.1f}) -> {out.name}")

    cap.release()
    print(f"\nDone: {kept} frames kept, {dropped} windows dropped as blurry "
          f"-> {output_dir}")
    if kept < 3:
        print("WARNING: fewer than 3 frames kept — reconstruction will fail. "
              "Lower --min-sharpness or capture a steadier video.")
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path, help="Input video file")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write the sampled frames")
    p.add_argument("--num-frames", type=int, default=50,
                   help="Target number of frames / temporal windows (default 50)")
    p.add_argument("--min-sharpness", type=float, default=60.0,
                   help="Variance-of-Laplacian threshold; windows whose best "
                        "frame scores below this are dropped (default 60)")
    p.add_argument("--stride", type=int, default=1,
                   help="Score every Nth frame inside a window (default 1 = all)")
    p.add_argument("--ext", default="jpg", choices=["jpg", "png"],
                   help="Output image format (default jpg)")
    p.add_argument("--prefix", default="frame_",
                   help="Output filename prefix (default 'frame_')")
    args = p.parse_args()

    sample_frames(args.video, args.output_dir, args.num_frames,
                  args.min_sharpness, args.stride, args.ext, args.prefix)


if __name__ == "__main__":
    main()
