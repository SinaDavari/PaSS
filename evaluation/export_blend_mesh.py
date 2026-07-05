#!/usr/bin/env python3
"""
PASS evaluation helper — export a mesh from a .blend file to world-space PLY.

Runs INSIDE Blender (headless):

    blender --background --python export_blend_mesh.py -- \
        --blend scene.blend --output mesh.ply [--object NAME]

Evaluation scenes often live as .blend files (e.g. the BlendCon scene files),
which Open3D cannot read. This exports one mesh — by name, or the largest —
with its full world transform baked in (object-level scale/rotation/location
applied), so downstream ICP and metrics operate on the same world-space
coordinates that Blender displays.
"""

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="export_blend_mesh.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blend", required=True, type=Path, help="Input .blend")
    p.add_argument("--output", required=True, type=Path, help="Output .ply")
    p.add_argument("--object", default=None,
                   help="Mesh object name (default: mesh with most vertices)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if not args.blend.is_file():
        sys.exit(f"ERROR: blend not found: {args.blend}")
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if args.object:
        obj = bpy.data.objects.get(args.object)
        if obj is None or obj.type != "MESH":
            names = ", ".join(o.name for o in meshes)
            sys.exit(f"ERROR: mesh '{args.object}' not found. Meshes: {names}")
    else:
        if not meshes:
            sys.exit("ERROR: no mesh objects in the blend file")
        obj = max(meshes, key=lambda o: len(o.data.vertices))
    print(f"Exporting '{obj.name}' ({len(obj.data.vertices):,} verts) "
          f"from {args.blend.name}")

    # Bake the world transform into the mesh data so the PLY holds
    # world-space coordinates (object scale/rotation/location applied).
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = str(args.output.resolve())
    if hasattr(bpy.ops.wm, "ply_export"):          # Blender >= 3.6 (new exporter)
        bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True,
                              export_normals=True, apply_modifiers=True,
                              forward_axis="Y", up_axis="Z")
    else:                                          # legacy exporter
        bpy.ops.export_mesh.ply(filepath=out, use_selection=True,
                                use_normals=True, use_mesh_modifiers=True)

    # report world-space extent as a sanity check
    n = len(obj.data.vertices)
    co = np.empty(n * 3)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    ext = co.max(0) - co.min(0)
    print(f"Exported: {args.output}")
    print(f"  world extent: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}")


if __name__ == "__main__":
    main()
