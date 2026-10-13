#!/usr/bin/env python3
"""
PASS stage 5 — metric-scale recovery + StandableFloor.

Runs INSIDE Blender (headless):

    blender --background --python scale_to_metric.py -- \
        --blend scene/mysite_leveled.blend --site-height 3.2 \
        --output scene/mysite.blend

Metric scale is fundamentally unobservable from monocular imagery, so PASS
requires exactly one real-world measurement, provided up front as a CLI
argument (never an interactive prompt — the pipeline is non-interactive):

  --site-height <meters>   the site's floor-to-ceiling height measured
                           on-site with a tape (DEPLOYMENT path)
  --lidar-ref <mesh/cloud> derive the target height from a LiDAR scan's Z
                           extent instead (EVALUATION-ONLY convenience; LiDAR
                           is never a deployment input)

The scene arrives leveled from stage 4 with the floor plane at Z = 0, so the
height of the reconstruction above the floor is simply its maximum Z. The
scale factor is s = H_target / max Z and the mesh is scaled uniformly about
the origin, which lies on the floor plane, so the floor stays at Z = 0. This
exact factor supersedes stage 4's provisional one; the report records both
and their product (the total factor applied to the stage-3 mesh).

With the scene level and metric, the walkable region is extracted from the
floor inliers found in stage 4 (vertex group "FloorInliers"): rasterize into
a `--cell-size` occupancy grid, close 1 cell, erode by `--inset`, and emit
one quad per surviving cell as the "StandableFloor" mesh at Z = 0 — the
region where BlendCon may place workers and cameras.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

FLOOR_OBJECT_NAME = "StandableFloor"
FLOOR_GROUP_NAME = "FloorInliers"
PROVISIONAL_SCALE_PROP = "pass_provisional_scale"


# --------------------------------------------------------------------------- #
# args / blender helpers
# --------------------------------------------------------------------------- #

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="scale_to_metric.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, type=Path,
                   help="Leveled .blend scene from stage 4 (floor at Z=0)")
    p.add_argument("--output", required=True, type=Path,
                   help="Metric .blend to write (scene mesh + StandableFloor)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--site-height", type=float,
                     help="Measured floor-to-ceiling height in meters (deployment)")
    src.add_argument("--lidar-ref", type=Path,
                     help="LiDAR reference mesh/cloud; its Z extent becomes "
                          "the target height (evaluation only)")
    p.add_argument("--object", default=None,
                   help="Scene mesh object name (default: largest mesh)")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="Floor band |Z| <= tolerance (m) used only if the "
                        "stage-4 vertex group is missing (default 0.10)")
    p.add_argument("--inset", type=float, default=0.30,
                   help="Inward inset of the standable area, meters "
                        "(default 0.30)")
    p.add_argument("--cell-size", type=float, default=0.10,
                   help="Occupancy-grid cell size, meters (default 0.10)")
    p.add_argument("--report", type=Path, default=None,
                   help="JSON report path (default: <output>.scale.json)")
    return p.parse_args(argv)


def import_ply(path: Path):
    """PLY import across Blender versions; returns the new object."""
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "ply_import"):          # Blender >= 4.0
        bpy.ops.wm.ply_import(filepath=str(path))
    else:                                          # Blender 3.x
        bpy.ops.import_mesh.ply(filepath=str(path))
    new = set(bpy.data.objects) - before
    if not new:
        sys.exit(f"ERROR: import produced no object: {path}")
    return new.pop()


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


def floor_inlier_indices(obj, tolerance: float) -> np.ndarray:
    """Indices of the stage-4 floor inliers (vertex group), or a |Z| band."""
    group = obj.vertex_groups.get(FLOOR_GROUP_NAME)
    if group is not None:
        gi = group.index
        idx = [v.index for v in obj.data.vertices
               if any(g.group == gi for g in v.groups)]
        if len(idx) >= 3:
            return np.asarray(idx, dtype=int)
    print(f"WARNING: vertex group '{FLOOR_GROUP_NAME}' missing — using "
          f"|Z| <= {tolerance} m as the floor band")
    z = world_vertices(obj)[:, 2]
    return np.nonzero(np.abs(z) <= tolerance)[0]


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

    # Stale floor from a previous run would be left at the wrong scale.
    old = bpy.data.objects.get(FLOOR_OBJECT_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    # ---- 1. height above the floor plane -----------------------------------
    pts = world_vertices(obj)
    current_h = float(pts[:, 2].max())          # floor is at Z=0 (stage 4)
    if current_h <= 0 or not math.isfinite(current_h):
        sys.exit("ERROR: mesh has no extent above Z=0 — run stage 4 first")

    # Target height: tape measurement (deployment) or LiDAR Z extent (eval).
    if args.site_height is not None:
        target_h, source = args.site_height, "measured site height"
        if target_h <= 0:
            sys.exit("ERROR: --site-height must be positive (meters)")
    else:
        if not args.lidar_ref.is_file():
            sys.exit(f"ERROR: LiDAR reference not found: {args.lidar_ref}")
        print(f"Importing LiDAR reference {args.lidar_ref} ...")
        lidar = import_ply(args.lidar_ref)
        lz = world_vertices(lidar)[:, 2]
        target_h, source = float(lz.max() - lz.min()), "LiDAR reference Z extent"
        bpy.data.objects.remove(lidar, do_unlink=True)   # measured, never kept

    factor = target_h / current_h
    provisional = float(obj.get(PROVISIONAL_SCALE_PROP, 1.0))
    print(f"1/3 Height above floor: {current_h:.4f} (stage-4 units)")
    print(f"    Target height:      {target_h:.4f} m  [{source}]")
    print(f"    Scale factor:       {factor:.6f}  (provisional {provisional:.6f}, "
          f"total vs. stage-3 mesh {provisional * factor:.6f})")

    # ---- 2. uniform scale about the origin (on the floor plane) ------------
    obj.matrix_world = Matrix.Scale(factor, 4) @ obj.matrix_world
    apply_object_transform(obj)
    if PROVISIONAL_SCALE_PROP in obj:
        del obj[PROVISIONAL_SCALE_PROP]
    obj["pass_scale_factor_total"] = provisional * factor
    print("2/3 Scaled to metric; floor remains at Z=0")

    # ---- 3. StandableFloor -------------------------------------------------
    pts = world_vertices(obj)
    floor_pts = pts[floor_inlier_indices(obj, args.tolerance)]
    if len(floor_pts) < 3:
        sys.exit("ERROR: no floor inliers available")

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
    print(f"3/3 StandableFloor: {int(eroded.sum())} cells "
          f"({area:.2f} m^2 after {args.inset} m inset)")

    # ---- save ---------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"\nMetric scene saved: {args.output}")

    report_path = args.report or args.output.with_suffix(".scale.json")
    report_path.write_text(json.dumps({
        "input": str(args.blend),
        "scene_object": obj.name,
        "floor_object": floor_obj.name,
        "scale": {
            "source": source,
            "target_height_m": target_h,
            "height_above_floor_before": round(current_h, 6),
            "factor": factor,
            "provisional_factor_stage4": provisional,
            "total_factor_vs_stage3_mesh": provisional * factor,
        },
        "standable_floor": {
            "floor_points": int(len(floor_pts)),
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
