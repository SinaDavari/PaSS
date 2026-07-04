#!/usr/bin/env python3
"""
PASS — EVALUATION BASELINE ONLY (not part of the deployment pipeline).

Frames -> colored point cloud via VGGT, used in the paper solely to compare
reconstruction quality against Depth Anything 3 (the deployment method).
World points and per-point confidence are taken from the official model
(https://github.com/facebookresearch/vggt, VGGT license — non-commercial);
the lowest-confidence percentile of points is culled before export.

Requires the VGGT repo to be installed/cloned locally (not vendored here).
Point --vggt-root at the clone.

Example:
    python run_vggt.py --input frames/ --output-dir recon/ --site-name mysite \
        --vggt-root ~/src/vggt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Directory of frames from stage 1")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--site-name", required=True)
    p.add_argument("--vggt-root", type=Path,
                   default=Path(os.environ.get("VGGT_ROOT", "")),
                   help="Path to the VGGT repo clone (or set $VGGT_ROOT)")
    p.add_argument("--ckpt", default="facebook/VGGT-1B",
                   help="HF model id or local path (default facebook/VGGT-1B)")
    p.add_argument("--cull-percent", type=float, default=5.0,
                   help="Percent of lowest-confidence points to remove (default 5)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not args.vggt_root or not (args.vggt_root / "vggt").is_dir():
        sys.exit("ERROR: VGGT repo not found. Clone "
                 "https://github.com/facebookresearch/vggt and pass "
                 "--vggt-root (or set $VGGT_ROOT).")
    sys.path.insert(0, str(args.vggt_root))

    import torch  # noqa: E402
    import trimesh  # noqa: E402
    from vggt.models.vggt import VGGT  # noqa: E402
    from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading VGGT ({args.ckpt})...")
    model = VGGT.from_pretrained(args.ckpt).to(device).eval()

    image_paths = sorted(str(f) for f in args.input.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        sys.exit(f"ERROR: no images found in {args.input}")
    print(f"Loaded {len(image_paths)} frames. Running inference...")

    inputs = load_and_preprocess_images(image_paths).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = model(inputs)

    world_points = predictions["world_points"].detach().cpu().numpy()
    images = predictions["images"].detach().cpu().numpy()
    conf = predictions["world_points_conf"].detach().cpu().numpy()
    if world_points.ndim == 5:  # drop batch dim
        world_points, images, conf = world_points[0], images[0], conf[0]
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))

    vertices = world_points.reshape(-1, 3)
    colors = images.reshape(-1, 3)
    confidence = conf.reshape(-1)

    # Percentile-based confidence culling (removes floaters/sky noise).
    if args.cull_percent > 0:
        thr = np.percentile(confidence, args.cull_percent)
        mask = confidence > thr
        print(f"Culling bottom {args.cull_percent}% confidence "
              f"({(~mask).sum()} of {len(mask)} points)")
        vertices, colors = vertices[mask], colors[mask]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.site_name}_pointcloud.ply"
    pcd = trimesh.points.PointCloud(vertices,
                                    colors=(colors * 255).astype(np.uint8))
    pcd.export(str(out))
    print(f"\nPoint cloud ready: {out} ({len(vertices)} points)")


if __name__ == "__main__":
    main()
