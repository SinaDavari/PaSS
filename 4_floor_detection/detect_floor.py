#!/usr/bin/env python3
"""
PASS stage 4 — floor detection + scene canonicalization.

Runs INSIDE Blender (headless):

    blender --background --python detect_floor.py -- \
        --mesh recon/mysite_mesh.ply --site-height 3.2 \
        --output scene/mysite_leveled.blend

Takes the cleaned mesh from stage 3, levels it and puts the floor at Z = 0.
Leveling comes BEFORE metric scaling (stage 5) because the scale factor is
derived from the site height measured normal to the floor, which is only
well defined once the scene is level.

Pipeline (fully automatic, replaces the manual "floor at Z=0" step):

  0. provisional scale : the mesh is still in reconstruction units, so the
                 fixed metric thresholds below would be meaningless. A
                 provisional factor H_target / (raw Z extent) brings it to
                 approximately metric units. Stage 5 supersedes it with the
                 exact factor measured normal to the floor.
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

The floor inliers are stored on the mesh as the vertex group "FloorInliers"
so that stage 5 can build the StandableFloor from exactly these points once
the scene is metric. Outputs the leveled .blend and a JSON report with the
plane parameters, corrected tilt, inlier counts and the provisional factor.
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
FLOOR_GROUP_NAME = "FloorInliers"
PROVISIONAL_SCALE_PROP = "pass_provisional_scale"


# --------------------------------------------------------------------------- #
# args / blender helpers
# --------------------------------------------------------------------------- #

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="detect_floor.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, type=Path,
                   help="Cleaned mesh .ply from stage 3")
    p.add_argument("--output", required=True, type=Path,
                   help="Leveled .blend to write (floor at Z=0)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--site-height", type=float,
                     help="Measured site height in meters, used for the "
                          "provisional scale (deployment)")
    src.add_argument("--lidar-ref", type=Path,
                     help="LiDAR reference mesh/cloud; its Z extent is the "
                          "provisional target height (evaluation only)")
    p.add_argument("--crop-fraction", type=float, default=1.0 / 3.0,
                   help="Bottom fraction of the Z range searched for the "
                        "floor (default 1/3)")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="RANSAC inlier band around the plane, meters "
                        "(default 0.10 = +/-10 cm)")
    p.add_argument("--iterations", type=int, default=2000,
                   help="RANSAC iterations (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="RANSAC RNG seed")
    p.add_argument("--report", type=Path, default=None,
                   help="JSON report path (default: <output>.floor.json)")
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


def add_vertex_color_material(obj):
    """Attach a material that renders the mesh's vertex colors.

    An imported PLY has no material, so Cycles/Eevee would render it plain
    white even though the vertex colors are present — the colors must be
    routed into a shader via a Color Attribute node."""
    if not obj.data.color_attributes:
        print("WARNING: mesh has no vertex colors — material not added")
        return
    attr_name = obj.data.color_attributes[0].name
    mat = bpy.data.materials.new("PASS_VertexColor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 1.0     # matte site surfaces
    vcol = nodes.new("ShaderNodeVertexColor")
    vcol.layer_name = attr_name
    mat.node_tree.links.new(bsdf.inputs["Base Color"], vcol.outputs["Color"])
    obj.data.materials.append(mat)
    print(f"Vertex-color material attached (attribute '{attr_name}')")


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
# main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    if not args.mesh.is_file():
        sys.exit(f"ERROR: mesh not found: {args.mesh}")
    rng = np.random.default_rng(args.seed)

    # Fresh empty scene; import the cleaned mesh and make its colors render.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    print(f"Importing {args.mesh} ...")
    obj = import_ply(args.mesh)
    obj.name = args.mesh.stem
    add_vertex_color_material(obj)
    print(f"Scene mesh: '{obj.name}' ({len(obj.data.vertices):,} verts)")

    # ---- 0. provisional scale ---------------------------------------------
    pts = world_vertices(obj)
    raw_extent = float(pts[:, 2].max() - pts[:, 2].min())
    if raw_extent <= 0 or not math.isfinite(raw_extent):
        sys.exit("ERROR: mesh has zero/invalid Z extent")
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
    provisional = target_h / raw_extent
    obj.matrix_world = Matrix.Scale(provisional, 4) @ obj.matrix_world
    apply_object_transform(obj)
    obj[PROVISIONAL_SCALE_PROP] = provisional
    print(f"0/5 Provisional scale {provisional:.6f} "
          f"(raw Z extent {raw_extent:.4f} -> {target_h:.4f} m, {source}); "
          f"superseded by stage 5")

    # ---- 1. bottom crop ----------------------------------------------------
    pts = world_vertices(obj)
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    z_cut = z_min + (z_max - z_min) * args.crop_fraction
    crop = pts[pts[:, 2] <= z_cut]
    print(f"1/5 Bottom crop: {len(crop):,} / {len(pts):,} verts "
          f"(z <= {z_cut:.3f})")

    # ---- 2. RANSAC ----------------------------------------------------------
    normal, d, crop_inliers = ransac_plane(crop, args.tolerance,
                                           args.iterations, rng)
    tilt_deg = math.degrees(math.acos(min(1.0, abs(normal[2]))))
    print(f"2/5 RANSAC floor plane: n=({normal[0]:.4f}, {normal[1]:.4f}, "
          f"{normal[2]:.4f}), d={d:.4f} | {int(crop_inliers.sum()):,} inliers"
          f" | tilt {tilt_deg:.2f} deg")

    # ---- 3. level: rotate n -> +Z -------------------------------------------
    rot = Vector(normal).rotation_difference(Vector((0, 0, 1))).to_matrix()
    obj.matrix_world = Matrix(rot.to_4x4()) @ obj.matrix_world
    apply_object_transform(obj)
    print("3/5 Levelled (floor normal aligned to +Z)")

    # ---- 4. floor to Z=0 ------------------------------------------------------
    pts = world_vertices(obj)
    z_cut = pts[:, 2].min() + (pts[:, 2].max() - pts[:, 2].min()) * args.crop_fraction
    crop_idx = np.nonzero(pts[:, 2] <= z_cut)[0]
    n2, d2, inl2 = ransac_plane(pts[crop_idx], args.tolerance, args.iterations, rng)
    floor_z = float(pts[crop_idx][inl2][:, 2].mean())
    obj.matrix_world = Matrix.Translation((0, 0, -floor_z)) @ obj.matrix_world
    apply_object_transform(obj)
    print(f"4/5 Floor plane dropped to Z=0 (was at Z={floor_z:.4f})")

    # ---- 5. XY re-center ----------------------------------------------------
    # Center the FLOOR footprint (not the whole-mesh bbox, which walls or
    # ceiling overhangs could skew) on the origin; Z stays owned by the floor.
    floor_idx = crop_idx[inl2]
    pts = world_vertices(obj)
    floor_pts = pts[floor_idx]
    if len(floor_pts) < 3:
        sys.exit("ERROR: no floor inliers after leveling")
    center_xy = (floor_pts[:, :2].min(axis=0)
                 + floor_pts[:, :2].max(axis=0)) / 2.0
    obj.matrix_world = (Matrix.Translation((-center_xy[0], -center_xy[1], 0))
                        @ obj.matrix_world)
    apply_object_transform(obj)
    print(f"5/5 Re-centered: floor footprint center moved from "
          f"({center_xy[0]:.3f}, {center_xy[1]:.3f}) to (0, 0)")

    # Remember the floor inliers for stage 5 (StandableFloor extraction).
    group = obj.vertex_groups.new(name=FLOOR_GROUP_NAME)
    group.add([int(i) for i in floor_idx], 1.0, "REPLACE")

    # ---- save ---------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"\nLeveled scene saved: {args.output}")

    report_path = args.report or args.output.with_suffix(".floor.json")
    report_path.write_text(json.dumps({
        "mesh": str(args.mesh),
        "scene_object": obj.name,
        "floor_vertex_group": FLOOR_GROUP_NAME,
        "provisional_scale": {
            "source": source,
            "target_height_m": target_h,
            "raw_z_extent": round(raw_extent, 6),
            "factor": provisional,
        },
        "ransac": {
            "normal_before_leveling": [round(float(x), 6) for x in normal],
            "tilt_corrected_deg": round(tilt_deg, 3),
            "inliers": int(len(floor_idx)),
            "crop_points": int(len(crop_idx)),
            "tolerance_m": args.tolerance,
            "iterations": args.iterations,
            "seed": args.seed,
        },
    }, indent=2) + "\n")
    print(f"Report: {report_path}")
    print("Next: 5_scale_registration/scale_to_metric.py")


if __name__ == "__main__":
    main()
