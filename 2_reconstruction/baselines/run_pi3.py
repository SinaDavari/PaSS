#!/usr/bin/env python3
"""
PASS — EVALUATION BASELINE ONLY (not part of the deployment pipeline).

Frames -> colored point cloud via Pi3 (π³), used in the paper solely to
compare reconstruction quality against Depth Anything 3 (the deployment
method). Inference follows the official example.py of
https://github.com/yyfz/Pi3 (single pass, no chunking): confidence mask
(sigmoid > 0.1) combined with a depth-edge mask, then colored points written
to .ply.

Requires the Pi3 repo to be installed/cloned locally (its license is
non-commercial; it is not vendored here). Point --pi3-root at the clone.

Example:
    python run_pi3.py --input frames/ --output-dir recon/ --site-name mysite \
        --pi3-root ~/src/Pi3
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Directory of frames from stage 1")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--site-name", required=True)
    p.add_argument("--pi3-root", type=Path,
                   default=Path(os.environ.get("PI3_ROOT", "")),
                   help="Path to the Pi3 repo clone (or set $PI3_ROOT)")
    p.add_argument("--ckpt", default=None,
                   help="Local checkpoint; default downloads yyfz233/Pi3 from HF")
    p.add_argument("--interval", type=int, default=1,
                   help="Sample every Nth input image (default 1 = all)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not args.pi3_root or not (args.pi3_root / "pi3").is_dir():
        sys.exit("ERROR: Pi3 repo not found. Clone https://github.com/yyfz/Pi3 "
                 "and pass --pi3-root (or set $PI3_ROOT).")
    sys.path.insert(0, str(args.pi3_root))

    import torch  # noqa: E402  (after path setup, matching Pi3's env)
    from pi3.models.pi3 import Pi3  # noqa: E402
    from pi3.utils.basic import load_images_as_tensor, write_ply  # noqa: E402
    from pi3.utils.geometry import depth_edge  # noqa: E402

    device = torch.device(args.device)

    print("Loading Pi3 model...")
    if args.ckpt:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(args.ckpt))
        else:
            model.load_state_dict(
                torch.load(args.ckpt, map_location=device, weights_only=False))
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    imgs = load_images_as_tensor(str(args.input), interval=args.interval).to(device)
    print(f"Loaded {imgs.shape[0]} frames. Running inference...")

    dtype = (torch.bfloat16 if args.device.startswith("cuda")
             and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs[None])

    # Official masking: confidence + depth-edge suppression (as in example.py).
    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(res["local_points"][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{args.site_name}_pointcloud.ply"
    write_ply(res["points"][0][masks].cpu(),
              imgs.permute(0, 2, 3, 1)[masks], str(out))
    print(f"\nPoint cloud ready: {out}")


if __name__ == "__main__":
    main()
