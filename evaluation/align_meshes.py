#!/usr/bin/env python3
"""
PASS evaluation — align a reconstructed mesh to the LiDAR reference. Headless.

NOT part of the deployment pipeline: LiDAR scans (Leica BLK ARC) are used
only as evaluation references, to compare the three image-based
reconstruction methods (DA3 — deployment — vs. VGGT and π³).

Two-step registration (Open3D), replacing the manual Blender overlay:
  1. 7-DoF point-to-point ICP *with scaling* — recovers the unknown scale of
     the image-based reconstruction along with a coarse pose
     (max correspondence distance 1.0 m)
  2. rigid point-to-plane ICP refinement (max correspondence distance 0.2 m)

Total transform T = T_refine @ T_coarse is written as JSON (4x4 matrix plus
a translation / Euler-XYZ rotation / scale decomposition, convenient for
inspection in Blender), and optionally the aligned prediction as .ply for
compute_metrics.py.

Example:
    python align_meshes.py --pred recon/mysite_mesh.ply --gt lidar/mysite.ply \
        --out-matrix eval/mysite_T.json --out-aligned eval/mysite_aligned.ply
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_as_pcd(path: Path) -> o3d.geometry.PointCloud:
    """Load a mesh or point cloud as an Open3D point cloud (vertices)."""
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) > 0:
        return pcd
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0:
        sys.exit(f"ERROR: no vertices/points in {path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_normals():
        pcd.normals = mesh.vertex_normals
    return pcd


def decompose(matrix: np.ndarray):
    """4x4 affine -> (translation, Euler XYZ degrees, per-axis scale)."""
    translation = matrix[:3, 3]
    upper = matrix[:3, :3]
    scale = np.linalg.norm(upper, axis=0)
    rotation = upper / scale
    euler = R.from_matrix(rotation).as_euler("xyz", degrees=True)
    return translation, euler, scale


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, type=Path,
                   help="Image-based reconstruction (mesh or cloud)")
    p.add_argument("--gt", required=True, type=Path,
                   help="LiDAR reference (mesh or cloud)")
    p.add_argument("--out-matrix", required=True, type=Path,
                   help="JSON file for the 4x4 transform + decomposition")
    p.add_argument("--out-aligned", type=Path, default=None,
                   help="Optional .ply of the aligned prediction "
                        "(input to compute_metrics.py)")
    p.add_argument("--coarse-dist", type=float, default=1.0,
                   help="Max correspondence distance, step 1 (default 1.0 m)")
    p.add_argument("--refine-dist", type=float, default=0.2,
                   help="Max correspondence distance, step 2 (default 0.2 m)")
    args = p.parse_args()

    print("Loading data ...")
    source = load_as_pcd(args.pred)   # reconstruction
    target = load_as_pcd(args.gt)     # LiDAR
    print(f"  pred: {len(source.points):,} pts | gt: {len(target.points):,} pts")

    # Step 1: 7-DoF (similarity) ICP — recovers scale + coarse pose.
    print("Step 1/2: point-to-point ICP with scaling ...")
    coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=args.coarse_dist,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(with_scaling=True))
    print(f"  fitness {coarse.fitness:.4f} | inlier RMSE "
          f"{coarse.inlier_rmse:.4f} m")

    # Step 2: rigid point-to-plane refinement on the coarsely aligned source.
    source.transform(coarse.transformation)
    if not target.has_normals():
        target.estimate_normals()
    if not source.has_normals():
        source.estimate_normals()

    print("Step 2/2: point-to-plane ICP refinement ...")
    refine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=args.refine_dist,
        init=np.identity(4),
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPlane())
    print(f"  fitness {refine.fitness:.4f} | inlier RMSE "
          f"{refine.inlier_rmse:.4f} m")

    total = refine.transformation @ coarse.transformation
    trans, euler, scale = decompose(total)

    args.out_matrix.parent.mkdir(parents=True, exist_ok=True)
    args.out_matrix.write_text(json.dumps({
        "pred": str(args.pred),
        "gt": str(args.gt),
        "matrix_4x4": total.tolist(),
        "decomposition": {
            "translation_m": trans.tolist(),
            "rotation_euler_xyz_deg": euler.tolist(),
            "scale_xyz": scale.tolist(),
        },
        "icp": {
            "coarse": {"fitness": coarse.fitness,
                       "inlier_rmse": coarse.inlier_rmse,
                       "max_dist": args.coarse_dist},
            "refine": {"fitness": refine.fitness,
                       "inlier_rmse": refine.inlier_rmse,
                       "max_dist": args.refine_dist},
        },
    }, indent=2) + "\n")
    print(f"\nTransform written: {args.out_matrix}")
    print(f"  scale ~ {scale.mean():.5f} | translation "
          f"({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}) m")

    if args.out_aligned:
        # `source` already carries T_coarse; apply the refinement on top.
        source.transform(refine.transformation)
        args.out_aligned.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.out_aligned), source)
        print(f"Aligned prediction written: {args.out_aligned}")
        print("Next: compute_metrics.py --pred <aligned> --gt <lidar>")


if __name__ == "__main__":
    main()
