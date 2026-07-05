#!/usr/bin/env python3
"""
PASS evaluation — align a reconstructed mesh to the LiDAR reference. Headless.

NOT part of the deployment pipeline: LiDAR scans (Leica BLK ARC) are used
only as evaluation references, to compare the three image-based
reconstruction methods (DA3 — deployment — vs. VGGT and π³).

Registration (Open3D), replacing the manual Blender overlay. Two candidate
initializations are refined and the better final fit is kept automatically:
  a. identity — wins when both meshes already sit in similar frames
     (e.g. floor at Z=0, similar orientation)
  b. global FPFH-feature RANSAC — wins when the frames differ
Each branch then runs:
  1. 7-DoF point-to-point ICP *with scaling* — recovers the residual scale
     of the reconstruction along with the pose (max corr. distance 1.0 m)
  2. rigid point-to-plane ICP refinement (max corr. distance 0.2 m)
Winner = highest refinement fitness (tie: lowest inlier RMSE). Assumes
roughly metric inputs (within ~2x of true scale).

Registration runs on voxel-downsampled copies (--voxel, default 5 cm) for
robustness and speed; the final transform is applied to the full cloud.

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
                   help="Max correspondence distance, step 2 (default 1.0 m)")
    p.add_argument("--refine-dist", type=float, default=0.2,
                   help="Max correspondence distance, step 3 (default 0.2 m)")
    p.add_argument("--voxel", type=float, default=0.05,
                   help="Voxel size (m) for the downsampled clouds used in "
                        "registration (default 0.05); the final transform is "
                        "applied to the full-resolution cloud")
    p.add_argument("--no-global-init", action="store_true",
                   help="Skip the FPFH+RANSAC global initialization and "
                        "start ICP from identity (legacy behavior)")
    args = p.parse_args()

    print("Loading data ...")
    source_full = load_as_pcd(args.pred)   # reconstruction
    target_full = load_as_pcd(args.gt)     # LiDAR
    print(f"  pred: {len(source_full.points):,} pts | "
          f"gt: {len(target_full.points):,} pts")

    # Registration runs on voxel-downsampled clouds: robust, fast, and the
    # resulting transform is applied to the full-resolution prediction.
    v = args.voxel
    source = source_full.voxel_down_sample(v)
    target = target_full.voxel_down_sample(v)
    for pc in (source, target):
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=v * 2, max_nn=30))
    print(f"  downsampled @ {v} m: pred {len(source.points):,} | "
          f"gt {len(target.points):,}")

    # Candidate initializations. Identity wins when the meshes already sit
    # in similar frames (e.g. both floor-at-Z=0); FPFH+RANSAC wins when they
    # don't. Both branches are refined and the better final fit is kept —
    # this replaces the judgment call of the manual "lay the meshes on top
    # of each other" step.
    inits = {"identity": np.identity(4)}
    if not args.no_global_init:
        print("Global registration candidate (FPFH + RANSAC) ...")
        fpfh = {}
        for name, pc in (("src", source), ("tgt", target)):
            fpfh[name] = o3d.pipelines.registration.compute_fpfh_feature(
                pc, o3d.geometry.KDTreeSearchParamHybrid(radius=v * 5,
                                                         max_nn=100))
        ransac = o3d.pipelines.registration.\
            registration_ransac_based_on_feature_matching(
                source, target, fpfh["src"], fpfh["tgt"],
                mutual_filter=True,
                max_correspondence_distance=v * 3,
                estimation_method=o3d.pipelines.registration.
                TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.
                    CorrespondenceCheckerBasedOnDistance(v * 3),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    1_000_000, 0.9999))
        print(f"  RANSAC fitness {ransac.fitness:.4f} | inlier RMSE "
              f"{ransac.inlier_rmse:.4f} m")
        inits["ransac"] = ransac.transformation

    def run_icp_branch(init_mat):
        """7-DoF scale ICP + rigid point-to-plane refine from a given init."""
        coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=args.coarse_dist,
            init=init_mat,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(with_scaling=True))
        src2 = o3d.geometry.PointCloud(source)   # keep `source` pristine
        src2.transform(coarse.transformation)
        refine = o3d.pipelines.registration.registration_icp(
            src2, target, max_correspondence_distance=args.refine_dist,
            init=np.identity(4),
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane())
        return coarse, refine

    branches = {}
    for name, init_mat in inits.items():
        coarse, refine = run_icp_branch(init_mat)
        s = np.linalg.norm(
            (refine.transformation @ coarse.transformation)[:3, 0])
        print(f"Branch '{name}': coarse fitness {coarse.fitness:.4f} | "
              f"refine fitness {refine.fitness:.4f}, "
              f"RMSE {refine.inlier_rmse:.4f} m | scale ~ {s:.4f}")
        branches[name] = (coarse, refine)

    # Winner: best refinement fitness (tie-broken by lower inlier RMSE).
    best = max(branches,
               key=lambda k: (branches[k][1].fitness,
                              -branches[k][1].inlier_rmse))
    coarse, refine = branches[best]
    print(f"Selected init: '{best}'")

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
        "registration": {
            "voxel_m": args.voxel,
            "selected_init": best,
            "branches": {k: {"refine_fitness": b[1].fitness,
                             "refine_rmse": b[1].inlier_rmse}
                         for k, b in branches.items()},
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
        # Apply the total transform to the FULL-resolution prediction.
        source_full.transform(total)
        args.out_aligned.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.out_aligned), source_full)
        print(f"Aligned prediction written: {args.out_aligned}")
        print("Next: compute_metrics.py --pred <aligned> --gt <lidar>")


if __name__ == "__main__":
    main()
