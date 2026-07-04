#!/usr/bin/env python3
"""
PASS evaluation — reconstruction-vs-LiDAR quality metrics. Headless.

Computes, between an aligned image-based reconstruction (pred) and the LiDAR
reference (gt):

  Accuracy            mean nearest-neighbor distance, pred -> gt
  Completeness        mean nearest-neighbor distance, gt -> pred
  Chamfer (L1)        mean of Accuracy and Completeness
  Normal Consistency  mean |n_pred . n_gt| over pred->gt matches
  Precision / Recall / F-score  at --threshold (default 5 cm)

Headless port of the Blender-terminal script used for the paper
(lidar_layover.py): same definitions, mathutils KDTree replaced by
scipy.cKDTree. Default point sets are the mesh VERTICES, matching the paper's
numbers exactly; pass --sample N for the density-independent variant that
uniformly samples N points per surface (more standard, but different from the
published tables).

Run align_meshes.py first, or pass its JSON via --matrix to transform the
prediction on the fly.

Example:
    python compute_metrics.py --pred eval/mysite_aligned.ply \
        --gt lidar/mysite.ply --csv eval/recon_metrics.csv --tag mysite_da3
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_points_normals(path: Path, sample: int):
    """Points + unit normals from a mesh (vertices or uniform samples) or
    point cloud (normals estimated if missing)."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        if sample > 0:
            pcd = mesh.sample_points_uniformly(sample, use_triangle_normal=True)
        else:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            pcd.normals = mesh.vertex_normals
    else:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            sys.exit(f"ERROR: no geometry in {path}")
        if sample > 0 and len(pcd.points) > sample:
            pcd = pcd.random_down_sample(sample / len(pcd.points))
        if not pcd.has_normals():
            pcd.estimate_normals()
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return pts, nrm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, type=Path,
                   help="Aligned reconstruction (from align_meshes.py)")
    p.add_argument("--gt", required=True, type=Path, help="LiDAR reference")
    p.add_argument("--matrix", type=Path, default=None,
                   help="Optional align_meshes.py JSON — applied to pred "
                        "before computing metrics")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Distance threshold for precision/recall/F-score "
                        "(default 0.05 m)")
    p.add_argument("--sample", type=int, default=0,
                   help="0 = use vertices (paper protocol); N>0 = uniformly "
                        "sample N points per surface (density-independent)")
    p.add_argument("--csv", type=Path, default=None,
                   help="Append results as a row to this CSV")
    p.add_argument("--tag", default=None,
                   help="Row label for the CSV (default: pred filename)")
    args = p.parse_args()

    print("Loading geometry ...")
    pred_pts, pred_nrm = load_points_normals(args.pred, args.sample)
    gt_pts, gt_nrm = load_points_normals(args.gt, args.sample)

    if args.matrix:
        T = np.asarray(json.loads(args.matrix.read_text())["matrix_4x4"])
        pred_pts = pred_pts @ T[:3, :3].T + T[:3, 3]
        # normals: rotation part only, re-normalized (handles the ICP scale)
        pred_nrm = pred_nrm @ T[:3, :3].T
        pred_nrm /= np.maximum(np.linalg.norm(pred_nrm, axis=1,
                                              keepdims=True), 1e-12)
        print(f"Applied transform from {args.matrix}")
    print(f"  pred: {len(pred_pts):,} pts | gt: {len(gt_pts):,} pts "
          f"({'vertices' if args.sample == 0 else f'{args.sample} samples'})")

    # Accuracy (pred -> gt) + normal consistency over the same matches.
    print("Accuracy (pred -> gt) ...")
    dist_a, idx_a = cKDTree(gt_pts).query(pred_pts, workers=-1)
    accuracy = float(dist_a.mean())
    precision = float((dist_a < args.threshold).mean())
    normal_consistency = float(
        np.abs(np.einsum("ij,ij->i", pred_nrm, gt_nrm[idx_a])).mean())

    # Completeness (gt -> pred).
    print("Completeness (gt -> pred) ...")
    dist_c, _ = cKDTree(pred_pts).query(gt_pts, workers=-1)
    completeness = float(dist_c.mean())
    recall = float((dist_c < args.threshold).mean())

    chamfer = (accuracy + completeness) / 2
    f_score = (2 * precision * recall / (precision + recall)
               if precision + recall > 0 else 0.0)

    print("\n" + "=" * 42)
    print("            RESULTS")
    print("=" * 42)
    print(f"pred: {args.pred.name}   gt: {args.gt.name}")
    print(f"threshold: {args.threshold} m")
    print("-" * 42)
    print(f"Accuracy (mean error)      {accuracy:.5f} m")
    print(f"Completeness (mean error)  {completeness:.5f} m")
    print(f"Chamfer (L1)               {chamfer:.5f} m")
    print(f"Normal Consistency         {normal_consistency:.5f}")
    print("-" * 42)
    print(f"Precision @ {args.threshold:.2f}         {precision * 100:.2f} %")
    print(f"Recall    @ {args.threshold:.2f}         {recall * 100:.2f} %")
    print(f"F-score   @ {args.threshold:.2f}         {f_score * 100:.2f} %")
    print("=" * 42)

    if args.csv:
        row = {"tag": args.tag or args.pred.stem,
               "accuracy": round(accuracy, 5),
               "completeness": round(completeness, 5),
               "chamfer": round(chamfer, 5),
               "normal_consistency": round(normal_consistency, 5),
               "precision": round(precision, 5),
               "recall": round(recall, 5),
               "f_score": round(f_score, 5),
               "threshold": args.threshold,
               "mode": "vertices" if args.sample == 0 else f"sample{args.sample}"}
        write_header = not args.csv.exists()
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"Appended to {args.csv}")


if __name__ == "__main__":
    main()
