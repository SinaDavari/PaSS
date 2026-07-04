#!/usr/bin/env python3
"""
PASS stage 4 — scale registration: cleaned mesh -> metric .blend scene.

Runs INSIDE Blender (headless):

    blender --background --python scale_to_metric.py -- \
        --mesh recon/mysite_mesh.ply --site-height 3.2 --output scene/mysite.blend

Metric scale is fundamentally unobservable from monocular imagery, so PASS
requires exactly one real-world measurement, provided up front as a CLI
argument (never an interactive prompt — the pipeline is non-interactive):

  --site-height <meters>   the site's total height measured on-site with a
                           tape (DEPLOYMENT path)
  --lidar-ref <mesh/cloud> derive the target height from a LiDAR scan's Z
                           extent instead (EVALUATION-ONLY convenience; LiDAR
                           is never a deployment input)

The mesh is scaled uniformly about the world origin so its Z extent matches
the target height. Because stage 3 anchored the bbox min corner at (0,0,0),
scaling about the origin preserves that corner. The scaled scene is saved as
.blend and a JSON sidecar records the applied factor for provenance.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="scale_to_metric.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True, type=Path,
                   help="Cleaned mesh .ply from stage 3")
    p.add_argument("--output", required=True, type=Path,
                   help=".blend file to write")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--site-height", type=float,
                     help="Measured total site height in meters (deployment)")
    src.add_argument("--lidar-ref", type=Path,
                     help="LiDAR reference mesh/cloud; its Z extent becomes "
                          "the target height (evaluation only)")
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


def z_extent(obj) -> float:
    zs = [(obj.matrix_world @ v.co).z for v in obj.data.vertices]
    return max(zs) - min(zs)


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


def main():
    args = parse_args()
    if not args.mesh.is_file():
        sys.exit(f"ERROR: mesh not found: {args.mesh}")

    # Fresh empty scene.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"Importing {args.mesh} ...")
    obj = import_ply(args.mesh)
    obj.name = args.mesh.stem
    add_vertex_color_material(obj)
    current_h = z_extent(obj)
    if current_h <= 0 or not math.isfinite(current_h):
        sys.exit("ERROR: mesh has zero/invalid Z extent")

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
        target_h, source = z_extent(lidar), "LiDAR reference Z extent"
        # Reference is only measured, never kept in the scene.
        bpy.data.objects.remove(lidar, do_unlink=True)

    factor = target_h / current_h
    print(f"Current Z extent: {current_h:.4f} (reconstruction units)")
    print(f"Target height:    {target_h:.4f} m  [{source}]")
    print(f"Scale factor:     {factor:.6f}")

    # Uniform scale about the world origin (preserves the stage-3 corner),
    # then freeze the transform into the mesh data.
    obj.scale = (factor, factor, factor)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()))
    print(f"\nMetric scene saved: {args.output}")

    sidecar = args.output.with_suffix(".scale.json")
    sidecar.write_text(json.dumps({
        "mesh": str(args.mesh),
        "scale_source": source,
        "target_height_m": target_h,
        "scale_factor": factor,
    }, indent=2) + "\n")
    print(f"Provenance sidecar: {sidecar}")
    print("Next: 5_floor_detection/detect_floor.py")


if __name__ == "__main__":
    main()
