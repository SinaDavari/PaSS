#!/usr/bin/env python3
"""
PASS stage 2 — reconstruction (DEPLOYMENT METHOD): frames -> colored point cloud
via Depth Anything 3 (DA3).

This is a thin wrapper around the official `da3` CLI
(https://github.com/ByteDance-Seed/Depth-Anything-3, Apache-2.0), which must be
installed in the current environment (`pip install -e .` in the DA3 repo).
It runs `da3 auto <frames> --export-format glb` (DA3's native point-cloud
export, with its adaptive confidence filtering and point cap), extracts the
point cloud from the GLB (largest non-wireframe geometry, vertex colors
preserved), and writes it under a standard name:

    <output-dir>/<site-name>_pointcloud.ply

so that every reconstruction backend (this one, and the evaluation-only
baselines in baselines/) exposes the exact same output contract to stage 3.

Example:
    python run_da3.py --input frames/ --output-dir recon/ --site-name mysite
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def newest(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def glb_to_ply(glb_path: Path, ply_path: Path, zup: bool = True) -> int:
    """Extract DA3's point cloud from the exported GLB and write a colored
    binary PLY. The GLB holds one large point-cloud geometry plus optional
    camera-wireframe (Path3D) nodes, which are skipped.

    DA3 exports in glTF convention (x-right, y-up, z-backward, oriented by
    the first camera); with zup=True the points are rotated to the pipeline's
    Z-up convention, (x, y, z) -> (x, -z, y), matching Blender's own glTF
    import. Residual tilt (the phone's pitch at recording start) is corrected
    later by stage 5's RANSAC leveling."""
    try:
        import trimesh
    except ImportError:
        sys.exit("ERROR: trimesh is required for the GLB->PLY conversion "
                 "(pip install trimesh, or run inside the DA3 environment).")

    scene = trimesh.load(str(glb_path))
    geom = None
    if isinstance(scene, trimesh.Scene):
        candidates = [g for g in scene.geometry.values()
                      if not isinstance(g, trimesh.path.Path3D)]
        if candidates:
            geom = max(candidates, key=lambda g: len(g.vertices))
    else:
        geom = scene
    if geom is None or len(geom.vertices) == 0:
        sys.exit(f"ERROR: no point geometry found in {glb_path}")

    vertices = geom.vertices
    if zup:
        vertices = vertices[:, [0, 2, 1]].copy()   # (x, y, z) -> (x, z, y)
        vertices[:, 1] *= -1.0                     #            -> (x, -z, y)

    colors = None
    if (hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors")
            and len(geom.visual.vertex_colors) > 0):
        colors = geom.visual.vertex_colors[:, :3]

    pcd = trimesh.points.PointCloud(vertices, colors=colors)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    pcd.export(str(ply_path))
    return len(vertices)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Directory of frames from stage 1 (or a video file)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for the exported point cloud")
    p.add_argument("--site-name", required=True,
                   help="Site identifier used for the output filename")
    p.add_argument("--model-dir", default=None,
                   help="DA3 model directory/id (default: DA3's own default)")
    p.add_argument("--process-res", type=int, default=None,
                   help="DA3 processing resolution (default: DA3's own default)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default cuda)")
    p.add_argument("--num-max-points", type=int, default=1_000_000,
                   help="Point cap for DA3's export (default 1,000,000)")
    p.add_argument("--keep-gltf-axes", action="store_true",
                   help="Skip the glTF(Y-up) -> pipeline(Z-up) rotation")
    p.add_argument("--da3-args", nargs=argparse.REMAINDER, default=[],
                   help="Everything after this flag is passed verbatim to "
                        "`da3 auto` (e.g. --da3-args --num-max-points 2000000)")
    args = p.parse_args()

    if shutil.which("da3") is None:
        sys.exit("ERROR: `da3` CLI not found. Install Depth Anything 3 first:\n"
                 "  git clone https://github.com/ByteDance-Seed/Depth-Anything-3\n"
                 "  cd Depth-Anything-3 && pip install -e .")

    export_dir = args.output_dir / "da3_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["da3", "auto", str(args.input),
           "--export-dir", str(export_dir),
           "--export-format", "glb",
           "--device", args.device,
           "--num-max-points", str(args.num_max_points),
           "--no-show-cameras",
           "--auto-cleanup"]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.process_res:
        cmd += ["--process-res", str(args.process_res)]
    cmd += args.da3_args

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    glb = newest(export_dir, "*.glb")
    if glb is None:
        sys.exit(f"ERROR: DA3 finished but no .glb found under {export_dir}")

    out = args.output_dir / f"{args.site_name}_pointcloud.ply"
    print(f"Extracting point cloud from {glb.name} ...")
    n_pts = glb_to_ply(glb, out, zup=not args.keep_gltf_axes)
    print(f"\nPoint cloud ready: {out} ({n_pts:,} points)")
    print("Next: 3_postprocess/clean_and_mesh.py")


if __name__ == "__main__":
    main()
