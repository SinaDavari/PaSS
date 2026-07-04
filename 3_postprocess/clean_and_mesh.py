#!/usr/bin/env python3
"""
PASS stage 3 — postprocess: raw point cloud -> clean, colored, origin-anchored mesh.

Automates (via the PyMeshLab API) the MeshLab workflow used in the paper:

  1. outlier removal on the point cloud (MeshLab defaults: k=32, prob=0.8)
  2. normal estimation           (neighbours 30, smooth iterations 10)
  3. Screened Poisson surface    (depth 12, min samples 1, interp. weight 10)
  4. trim low-density Poisson "balloon" bulges (bottom 5% density)
  5. vertex-color transfer from the cloud onto the new mesh
  6. remove isolated pieces      (MeshLab default: diameter < 10% of bbox diag)
  7. translate so the bounding-box MIN corner sits at (0, 0, 0)
     (BlendCon's scene convention; stage 5 later refines Z with the true floor)
     and export as binary PLY with vertex colors

Runs fully headless — no MeshLab GUI needed.

Example:
    python clean_and_mesh.py --input recon/mysite_pointcloud.ply \
                             --output recon/mysite_mesh.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pymeshlab


def pct(value: float):
    """Percentage value across pymeshlab API generations."""
    if hasattr(pymeshlab, "PercentageValue"):      # >= 2023.12
        return pymeshlab.PercentageValue(value)
    return pymeshlab.Percentage(value)             # older releases


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Raw point cloud .ply from stage 2")
    p.add_argument("--output", required=True, type=Path,
                   help="Cleaned mesh .ply to write")
    # Normal estimation (paper settings)
    p.add_argument("--normal-neighbors", type=int, default=30)
    p.add_argument("--normal-smooth-iters", type=int, default=10)
    # Screened Poisson (paper settings)
    p.add_argument("--poisson-depth", type=int, default=12)
    p.add_argument("--poisson-min-samples", type=float, default=1.0)
    p.add_argument("--poisson-interp-weight", type=float, default=10.0)
    # Outlier removal (MeshLab defaults)
    p.add_argument("--outlier-knearest", type=int, default=32)
    p.add_argument("--outlier-prob", type=float, default=0.8)
    # Isolated pieces (MeshLab default: 10% of bbox diagonal)
    p.add_argument("--isolated-diag-pct", type=float, default=10.0)
    # Poisson balloon-bulge trim (bottom N% of Poisson density; 0 = off)
    p.add_argument("--density-trim-pct", type=float, default=5.0)
    args = p.parse_args()

    if not args.input.is_file():
        sys.exit(f"ERROR: input not found: {args.input}")

    ms = pymeshlab.MeshSet()
    print(f"Loading {args.input} ...")
    ms.load_new_mesh(str(args.input))
    cloud_id = ms.current_mesh_id()
    print(f"  {ms.current_mesh().vertex_number():,} points")

    # -- 1. outlier removal on the cloud -------------------------------------
    print("1/7 Removing point-cloud outliers "
          f"(k={args.outlier_knearest}, prob={args.outlier_prob}) ...")
    try:                                          # pymeshlab >= 2023.12
        ms.compute_selection_point_cloud_outliers(
            propthreshold=args.outlier_prob,
            knearest=args.outlier_knearest)
    except pymeshlab.PyMeshLabException:          # older releases
        ms.compute_selection_point_cloud_outliers(
            propability=args.outlier_prob,        # (sic — old spelling)
            knearest=args.outlier_knearest)
    ms.meshing_remove_selected_vertices()
    print(f"  {ms.current_mesh().vertex_number():,} points remain")

    # -- 2. normals -----------------------------------------------------------
    print(f"2/7 Estimating normals (k={args.normal_neighbors}, "
          f"smooth iters={args.normal_smooth_iters}) ...")
    ms.compute_normal_for_point_clouds(
        k=args.normal_neighbors,
        smoothiter=args.normal_smooth_iters)

    # -- 3. screened Poisson ---------------------------------------------------
    print(f"3/7 Screened Poisson (depth={args.poisson_depth}, "
          f"min samples={args.poisson_min_samples}, "
          f"interp weight={args.poisson_interp_weight}) ...")
    ms.generate_surface_reconstruction_screened_poisson(
        depth=args.poisson_depth,
        samplespernode=args.poisson_min_samples,
        pointweight=args.poisson_interp_weight)
    mesh_id = ms.current_mesh_id()
    print(f"  mesh: {ms.current_mesh().vertex_number():,} verts, "
          f"{ms.current_mesh().face_number():,} faces")

    # -- 4. trim low-density Poisson bulges ------------------------------------
    # Screened Poisson closes open scan boundaries with large low-density
    # "balloon" bubbles that inflate the bbox (and would corrupt the
    # height-based scaling in stage 4). Poisson stores per-vertex density as
    # quality; drop the sparsest percentile.
    if args.density_trim_pct > 0:
        print(f"4/7 Trimming low-density Poisson bulges "
              f"(bottom {args.density_trim_pct}%) ...")
        quality = ms.current_mesh().vertex_scalar_array()
        thr = float(np.percentile(quality, args.density_trim_pct))
        ms.compute_selection_by_condition_per_vertex(condselect=f"(q < {thr})")
        ms.meshing_remove_selected_vertices()
        print(f"  mesh: {ms.current_mesh().vertex_number():,} verts")
    else:
        print("4/7 Density trim disabled")

    # -- 5. color transfer cloud -> mesh --------------------------------------
    print("5/7 Transferring vertex colors from the cloud ...")
    ms.transfer_attributes_per_vertex(
        sourcemesh=cloud_id, targetmesh=mesh_id, colortransfer=True)

    # -- 6. remove isolated pieces ---------------------------------------------
    print(f"6/7 Removing isolated pieces "
          f"(< {args.isolated_diag_pct}% of bbox diagonal) ...")
    ms.set_current_mesh(mesh_id)
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pct(args.isolated_diag_pct))
    ms.meshing_remove_unreferenced_vertices()
    print(f"  mesh: {ms.current_mesh().vertex_number():,} verts, "
          f"{ms.current_mesh().face_number():,} faces")

    # -- 7. anchor bbox min corner at the origin -------------------------------
    print("7/7 Translating bbox min corner to (0, 0, 0) ...")
    bbox_min = ms.current_mesh().bounding_box().min()
    ms.compute_matrix_from_translation(
        traslmethod="Set new Origin",            # (sic — pymeshlab spelling)
        neworigin=np.asarray(bbox_min, dtype=np.float64),
        freeze=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(args.output), binary=True,
                         save_vertex_color=True, save_vertex_normal=True)
    dims = ms.current_mesh().bounding_box()
    print(f"\nCleaned mesh ready: {args.output}")
    print(f"  extent: {dims.dim_x():.2f} x {dims.dim_y():.2f} x "
          f"{dims.dim_z():.2f} (reconstruction units — metric after stage 4)")
    print("Next: 4_scale_registration/scale_to_metric.py")


if __name__ == "__main__":
    main()
