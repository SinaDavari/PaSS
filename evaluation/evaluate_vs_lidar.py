#!/usr/bin/env python3
"""
PASS evaluation — one-command reconstruction-vs-LiDAR evaluation from .blend
files, replicating the paper's manual Blender workflow end-to-end:

  1. open the LiDAR ground-truth .blend (headless Blender)
  2. append every object from the reconstruction .blend into it
     (the manual "copy-paste all objects" step)
  3. export the two main meshes as world-space point clouds
  4. register the reconstruction to the LiDAR (align_meshes.py: identity and
     FPFH+RANSAC initializations, 7-DoF scale ICP + point-to-plane refine,
     best branch kept — the automated "lay them on top of each other" step)
  5. compute Accuracy / Completeness / Chamfer / Normal Consistency /
     F-score (compute_metrics.py, vertex protocol — lidar_layover.py port)
  6. save a merged .blend with the reconstruction laid over the LiDAR for
     visual review (originals are never modified)

Usage (regular Python with open3d+scipy; Blender is called as a subprocess):

    python evaluate_vs_lidar.py \
        --gt-blend lash-1311-enc-lidar.blend --pred-blend lash-enc-pi3.blend \
        --workdir out/small_site_pi3 --tag small_site_pi3 \
        [--gt-object NAME] [--pred-object NAME] [--csv metrics.csv]

By default the main mesh on each side is the one with the most vertices
(GT: among the LiDAR blend's own objects; pred: among the appended ones).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()


# =========================================================================== #
# Blender-side tasks (run via `blender --background --python thisfile -- ...`)
# =========================================================================== #

def blender_main():
    import bpy
    import numpy as np

    argv = sys.argv[sys.argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--blender-task", required=True,
                   choices=["merge_export", "apply_save"])
    p.add_argument("--gt-blend", required=True, type=Path)
    p.add_argument("--pred-blend", required=True, type=Path)
    p.add_argument("--gt-object", default=None)
    p.add_argument("--pred-object", default=None)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--matrix-json", type=Path, default=None)
    p.add_argument("--out-blend", type=Path, default=None)
    a = p.parse_args(argv)

    bpy.ops.wm.open_mainfile(filepath=str(a.gt_blend.resolve()))
    gt_meshes = [o for o in bpy.data.objects if o.type == "MESH"]

    # Append every object from the reconstruction blend (GUI copy-paste).
    with bpy.data.libraries.load(str(a.pred_blend.resolve())) as (src, dst):
        dst.objects = list(src.objects)
    appended = [o for o in dst.objects if o is not None]
    for o in appended:
        bpy.context.scene.collection.objects.link(o)
    appended_meshes = [o for o in appended if o.type == "MESH"]
    print(f"[blender] appended {len(appended)} objects "
          f"({len(appended_meshes)} meshes) from {a.pred_blend.name}")

    def pick(meshes, name, side):
        if name:
            obj = next((o for o in meshes if o.name == name
                        or o.name.startswith(name)), None)
            if obj is None:
                sys.exit(f"ERROR: {side} mesh '{name}' not found among: "
                         + ", ".join(o.name for o in meshes))
            return obj
        return max(meshes, key=lambda o: len(o.data.vertices))

    gt_obj = pick(gt_meshes, a.gt_object, "GT")
    pred_obj = pick(appended_meshes, a.pred_object, "pred")
    print(f"[blender] GT mesh: '{gt_obj.name}' "
          f"({len(gt_obj.data.vertices):,} verts) | pred mesh: "
          f"'{pred_obj.name}' ({len(pred_obj.data.vertices):,} verts)")

    if a.blender_task == "merge_export":
        # Export both main meshes as world-space PLY point clouds.
        a.workdir.mkdir(parents=True, exist_ok=True)
        for obj, fname in ((gt_obj, "gt.ply"), (pred_obj, "pred.ply")):
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.transform_apply(location=True, rotation=True,
                                           scale=True)
            out = str((a.workdir / fname).resolve())
            if hasattr(bpy.ops.wm, "ply_export"):
                bpy.ops.wm.ply_export(filepath=out,
                                      export_selected_objects=True,
                                      export_normals=True,
                                      apply_modifiers=True,
                                      forward_axis="Y", up_axis="Z")
            else:
                bpy.ops.export_mesh.ply(filepath=out, use_selection=True,
                                        use_normals=True,
                                        use_mesh_modifiers=True)
            print(f"[blender] exported {fname}")

    else:  # apply_save
        from mathutils import Matrix
        T = Matrix(json.loads(a.matrix_json.read_text())["matrix_4x4"])
        # Apply to parent-less appended objects; children follow their parents.
        roots = [o for o in appended if o.parent is None
                 or o.parent not in appended]
        for o in roots:
            o.matrix_world = T @ o.matrix_world
        print(f"[blender] transform applied to {len(roots)} root object(s)")
        a.out_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(a.out_blend.resolve()))
        print(f"[blender] overlay saved: {a.out_blend}")


# =========================================================================== #
# Driver (regular Python)
# =========================================================================== #

def run(cmd, what):
    print(f"\n=== {what} ===")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"ERROR: {what} failed (exit {r.returncode})")


def driver_main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-blend", required=True, type=Path,
                   help="LiDAR ground-truth .blend (never modified)")
    p.add_argument("--pred-blend", required=True, type=Path,
                   help="Reconstruction .blend (never modified)")
    p.add_argument("--gt-object", default=None,
                   help="GT mesh name (default: largest mesh in gt-blend)")
    p.add_argument("--pred-object", default=None,
                   help="Pred mesh name (default: largest appended mesh)")
    p.add_argument("--workdir", required=True, type=Path,
                   help="Directory for exports, transform, metrics, overlay")
    p.add_argument("--tag", required=True, help="Row label for the CSV")
    p.add_argument("--csv", type=Path, default=None,
                   help="Append metrics to this CSV "
                        "(default: <workdir>/metrics.csv)")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--blender-bin", default="blender")
    args = p.parse_args()

    for f in (args.gt_blend, args.pred_blend):
        if not f.is_file():
            sys.exit(f"ERROR: not found: {f}")
    args.workdir.mkdir(parents=True, exist_ok=True)
    csv = args.csv or (args.workdir / "metrics.csv")

    obj_flags = []
    if args.gt_object:
        obj_flags += ["--gt-object", args.gt_object]
    if args.pred_object:
        obj_flags += ["--pred-object", args.pred_object]

    # 1-3. merge + export world-space clouds
    run([args.blender_bin, "--background", "--python", str(SCRIPT), "--",
         "--blender-task", "merge_export",
         "--gt-blend", str(args.gt_blend), "--pred-blend", str(args.pred_blend),
         "--workdir", str(args.workdir)] + obj_flags,
        "Blender: merge + export")

    # 4. registration
    T_json = args.workdir / f"{args.tag}_T.json"
    run([sys.executable, str(SCRIPT.parent / "align_meshes.py"),
         "--pred", str(args.workdir / "pred.ply"),
         "--gt", str(args.workdir / "gt.ply"),
         "--out-matrix", str(T_json)],
        "Registration (align_meshes.py)")

    # 5. metrics
    run([sys.executable, str(SCRIPT.parent / "compute_metrics.py"),
         "--pred", str(args.workdir / "pred.ply"),
         "--gt", str(args.workdir / "gt.ply"),
         "--matrix", str(T_json),
         "--threshold", str(args.threshold),
         "--csv", str(csv), "--tag", args.tag],
        "Metrics (compute_metrics.py)")

    # 6. overlay blend for visual review
    overlay = args.workdir / f"{args.tag}_overlay.blend"
    run([args.blender_bin, "--background", "--python", str(SCRIPT), "--",
         "--blender-task", "apply_save",
         "--gt-blend", str(args.gt_blend), "--pred-blend", str(args.pred_blend),
         "--workdir", str(args.workdir),
         "--matrix-json", str(T_json), "--out-blend", str(overlay)] + obj_flags,
        "Blender: apply transform + save overlay")

    print(f"\nDone. Review overlay: {overlay}")
    print(f"Transform: {T_json}\nMetrics CSV: {csv}")


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401 — inside Blender?
        blender_main()
    except ImportError:
        driver_main()
