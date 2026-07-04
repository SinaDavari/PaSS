#!/usr/bin/env python3
"""
PASS stage 5 — floor detection + scene canonicalization.

Runs INSIDE Blender (headless):

    blender --background --python detect_floor.py -- \
        --blend scene/mysite.blend --output scene/mysite_floor.blend

Pipeline (fully automatic, replaces the manual "floor at Z=0" step):

  1. crop      : keep vertices in the bottom `--crop-fraction` of the Z range
                 (assumes the reconstruction is roughly upright, which holds
                 for DA3 outputs from handheld video)
  2. RANSAC    : fit the dominant plane (the floor) with inlier band
                 +/- `--tolerance` m; least-squares refit on the inliers
  3. level     : rotate the whole mesh so the floor normal aligns with +Z
                 (also fixes small reconstruction tilt)
  4. drop      : translate so the floor plane sits exactly at Z = 0
                 (using the plane — not min-Z — so noise below the floor
                 cannot float the real floor above zero)
  5. re-center : translate X/Y so the center of the floor footprint sits at
                 the origin (the site is centered around (0,0,0) for
                 BlendCon; Z is owned by the floor)
  6. footprint : rasterize floor inliers into an occupancy grid
                 (`--cell-size` m), morphologically erode by `--inset` m,
                 and build the "StandableFloor" mesh at Z = 0 — the region
                 where BlendCon may place workers and cameras

Outputs the canonicalized .blend (scene mesh + StandableFloor object) and a
JSON report with plane parameters, inlier counts and standable area.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

FLOOR_OBJECT_NAME = "StandableFloor"


# --------------------------------------------------------------------------- #
# args / blender helpers
# --------------------------------------------------------------------------- #

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="detect_floor.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, type=Path,
                   help="Metric .blend scene from stage 4")
    p.add_argument("--output", required=True, type=Path,
                   help="Canonicalized .blend to write")
    p.add_argument("--object", default=None,
                   help="Scene mesh object name (default: largest mesh)")
    p.add_argument("--crop-fraction", type=float, default=1.0 / 3.0,
                   help="Bottom fraction of the Z range searched for the "
                        "floor (default 1/3)")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="RANSAC inlier band around the plane, meters "
                        "(default 0.10 = +/-10 cm)")
    p.add_argument("--inset", type=float, default=0.30,
                   help="Inward inset of the standable area, meters "
                        "(default 0.30)")
    p.add_argument("--cell-size", type=float, default=0.10,
                   help="Occupancy-grid cell size, meters (default 0.10)")
    p.add_argument("--iterations", type=int, default=2000,
                   help="RANSAC iterations (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed")
    p.add_argument("--report", type=Path, default=None,
                   help="JSON report path (default: <output>.floor.json)")
    return p.parse_args(argv)


def world_vertices(obj) -> np.ndarray:
    """All vertices of `obj` in world space, as an (N, 3) float64 array."""
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    m = np.asarray(obj.matrix_world, dtype=np.float64)
    return co @ m[:3, :3].T + m[:3, 3]


def apply_object_transform(obj):
    """Freeze the object's current matrix_world into its mesh data."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


# --------------------------------------------------------------------------- #
# RANSAC plane fit
# --------------------------------------------------------------------------- #

def ransac_plane(points: np.ndarray, tolerance: float, iterations: int,
                 rng: np.random.Generator):
    """Return (unit normal, d) of the dominant plane n.x + d = 0, and the
    inlier mask. Normal is oriented upward (+Z hemisphere)."""
    n_pts = len(points)
    if n_pts < 3:
        sys.exit("ERROR: not enough points in the floor crop for RANSAC")

    best_inliers, best_count = None, -1
    for _ in range(iterations):
        i, j, k = rng.choice(n_pts, 3, replace=False)
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:            # degenerate (collinear) sample
            continue
        normal /= norm
        d = -normal.dot(points[i])
        dist = np.abs(points @ normal + d)
        count = int((dist <= tolerance).sum())
        if count > best_count:
            best_count = count
            best_inliers = dist <= tolerance

    if best_inliers is None or best_count < 3:
        sys.exit("ERROR: RANSAC found no plane — check the input scene")

    # Least-squares refit (SVD) on the inliers for a stable final plane.
    pts = points[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[2]
    if normal[2] < 0:               # orient upward
        normal = -normal
    d = -normal.dot(centroid)
    dist = np.abs(points @ normal + d)
    inliers = dist <= tolerance
    return normal, d, inliers


# --------------------------------------------------------------------------- #
# footprint: occupancy grid -> eroded -> mesh
# --------------------------------------------------------------------------- #

def _shift(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Grid shifted by (dx, dy), zero-padded (no scipy inside Blender)."""
    shifted = np.zeros_like(grid)
    xs = slice(max(dx, 0), grid.shape[0] + min(dx, 0))
    xd = slice(max(-dx, 0), grid.shape[0] + min(-dx, 0))
    ys = slice(max(dy, 0), grid.shape[1] + min(dy, 0))
    yd = slice(max(-dy, 0), grid.shape[1] + min(-dy, 0))
    shifted[xd, yd] = grid[xs, ys]
    return shifted


def binary_morph(grid: np.ndarray, radius_cells: int,
                 op: str) -> np.ndarray:
    """Square-structuring-element binary erosion/dilation via shifted
    slicing."""
    if radius_cells <= 0:
        return grid
    out = grid.copy()
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            if dx == 0 and dy == 0:
                continue
            if op == "erode":
                out &= _shift(grid, dx, dy)
            else:
                out |= _shift(grid, dx, dy)
    return out


def build_floor_mesh(occupied: np.ndarray, origin_xy: np.ndarray,
                     cell: float) -> bpy.types.Object:
    """One quad per occupied cell at Z=0, shared vertices, as a new object."""
    verts, faces = [], []
    vidx = {}                        # (gx, gy) grid-node -> vertex index

    def node(gx: int, gy: int) -> int:
        key = (gx, gy)
        if key not in vidx:
            vidx[key] = len(verts)
            verts.append((origin_xy[0] + gx * cell,
                          origin_xy[1] + gy * cell, 0.0))
        return vidx[key]

    for gx, gy in zip(*np.nonzero(occupied)):
        faces.append((node(gx, gy), node(gx + 1, gy),
                      node(gx + 1, gy + 1), node(gx, gy + 1)))

    mesh = bpy.data.meshes.new(FLOOR_OBJECT_NAME)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(FLOOR_OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    if not args.blend.is_file():
        sys.exit(f"ERROR: blend file not found: {args.blend}")
    rng = np.random.default_rng(args.seed)

    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))

    # Target mesh: named, or the mesh with the most vertices.
    meshes = [o for o in bpy.data.objects if o.type == "MESH"
              and o.name != FLOOR_OBJECT_NAME]
    if args.object:
        obj = bpy.data.objects.get(args.object)
        if obj is None or obj.type != "MESH":
            sys.exit(f"ERROR: mesh object not found: {args.object}")
    else:
        if not meshes:
            sys.exit("ERROR: no mesh objects in the scene")
        obj = max(meshes, key=lambda o: len(o.data.vertices))
    print(f"Scene mesh: '{obj.name}' ({len(obj.data.vertices):,} verts)")

    # Stale floor from a previous run would corrupt the crop — remove it.
    old = bpy.data.objects.get(FLOOR_OBJECT_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    # ---- 1. bottom crop ----------------------------------------------------
    pts = world_vertices(obj)
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    z_cut = z_min + (z_max - z_min) * args.crop_fraction
    crop = pts[pts[:, 2] <= z_cut]
    print(f"1/6 Bottom crop: {len(crop):,} / {len(pts):,} verts "
          f"(z <= {z_cut:.3f})")

    # ---- 2. RANSAC ----------------------------------------------------------
    normal, d, crop_inliers = ransac_plane(crop, args.tolerance,
                                           args.iterations, rng)
    tilt_deg = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    print(f"2/6 RANSAC floor plane: n=({normal[0]:.4f}, {normal[1]:.4f}, "
          f"{normal[2]:.4f}), d={d:.4f} | {int(crop_inliers.sum()):,} inliers"
          f" | tilt {tilt_deg:.2f} deg")

    # ---- 3. level: rotate n -> +Z -------------------------------------------
    rot = Vector(normal).rotation_difference(Vector((0, 0, 1))).to_matrix()
    obj.matrix_world = Matrix(rot.to_4x4()) @ obj.matrix_world
    apply_object_transform(obj)
    print("3/6 Levelled (floor normal aligned to +Z)")

    # ---- 4. floor to Z=0 ------------------------------------------------------
    pts = world_vertices(obj)
    z_cut = pts[:, 2].min() + (pts[:, 2].max() - pts[:, 2].min()) * args.crop_fraction
    crop = pts[pts[:, 2] <= z_cut]
    n2, d2, inl2 = ransac_plane(crop, args.tolerance, args.iterations, rng)
    floor_z = float(crop[inl2][:, 2].mean())
    obj.matrix_world = Matrix.Translation((0, 0, -floor_z)) @ obj.matrix_world
    apply_object_transform(obj)
    print(f"4/6 Floor plane dropped to Z=0 (was at Z={floor_z:.4f})")

    # ---- 5. XY re-center ----------------------------------------------------
    # Center the FLOOR footprint (not the whole-mesh bbox, which walls or
    # ceiling overhangs could skew) on the origin; Z stays owned by the floor.
    pts = world_vertices(obj)
    floor_pts = pts[np.abs(pts[:, 2]) <= args.tolerance]
    if len(floor_pts) < 3:
        sys.exit("ERROR: no floor inliers after leveling")
    center_xy = (floor_pts[:, :2].min(axis=0)
                 + floor_pts[:, :2].max(axis=0)) / 2.0
    obj.matrix_world = (Matrix.Translation((-center_xy[0], -center_xy[1], 0))
                        @ obj.matrix_world)
    apply_object_transform(obj)
    print(f"5/6 Re-centered: floor footprint center moved from "
          f"({center_xy[0]:.3f}, {center_xy[1]:.3f}) to (0, 0)")

    # ---- 6. StandableFloor -------------------------------------------------
    pts = world_vertices(obj)
    floor_pts = pts[np.abs(pts[:, 2]) <= args.tolerance]
    if len(floor_pts) < 3:
        sys.exit("ERROR: no floor inliers after canonicalization")

    cell = args.cell_size
    origin_xy = np.floor(floor_pts[:, :2].min(axis=0) / cell) * cell
    idx = np.floor((floor_pts[:, :2] - origin_xy) / cell).astype(int)
    grid = np.zeros(idx.max(axis=0) + 1, dtype=bool)
    grid[idx[:, 0], idx[:, 1]] = True

    # 1-cell dilation (morphological closing) bridges point-sampling gaps in
    # the raster; the erosion radius grows by the same 1 cell so the net
    # inset is preserved.
    closed = binary_morph(grid, 1, "dilate")
    radius = max(1, int(math.ceil(args.inset / cell))) + 1
    eroded = binary_morph(closed, radius, "erode")
    if not eroded.any():
        sys.exit(f"ERROR: standable area empty after {args.inset} m inset — "
                 f"reduce --inset or check the reconstruction scale")

    floor_obj = build_floor_mesh(eroded, origin_xy, cell)
    area = float(eroded.sum()) * cell * cell
    print(f"6/6 StandableFloor: {int(eroded.sum())} cells "
          f"({area:.2f} m^2 after {args.inset} m inset)")

    # ---- save ---------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"\nCanonicalized scene saved: {args.output}")

    report_path = args.report or args.output.with_suffix(".floor.json")
    report_path.write_text(json.dumps({
        "input": str(args.blend),
        "scene_object": obj.name,
        "floor_object": floor_obj.name,
        "ransac": {
            "normal_before_leveling": [round(float(x), 6) for x in normal],
            "tilt_corrected_deg": round(tilt_deg, 3),
            "inliers": int(crop_inliers.sum()),
            "crop_points": int(len(crop)),
            "tolerance_m": args.tolerance,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "standable_floor": {
            "cells": int(eroded.sum()),
            "cell_size_m": cell,
            "inset_m": args.inset,
            "area_m2": round(area, 3),
        },
    }, indent=2) + "\n")
    print(f"Report: {report_path}")
    print("Next: 6_blendcon (synthetic data generation)")


if __name__ == "__main__":
    main()
