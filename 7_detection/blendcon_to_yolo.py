#!/usr/bin/env python3
"""
PASS stage 7 — BlendCon output -> YOLO-ready dataset.

Two things happen here:

1. **Label derivation** — every BlendCon scenario folder (a folder containing
   `Joint_Tracker.pickle`) gets a `2DBBs/` subfolder with one YOLO-format
   .txt per rendered image: `0 cx cy w h` (normalized), class 0 = worker.
   2D boxes are read from the pickle's per-frame `BB2D` entries, clamped to
   the image bounds; boxes completely outside the frame are skipped.

2. **Dataset assembly** — images + labels from ALL scenarios are collected
   into a single YOLO dataset with a deterministic shuffled train/val split
   and one `dataset.yaml`:

       <out>/
       ├── images/{train,val}/<scenario>__<image>.jpg
       ├── labels/{train,val}/<scenario>__<image>.txt
       └── dataset.yaml

Example (one or more BlendCon Dataset roots and/or scenario folders):
    python blendcon_to_yolo.py --runs blendcon_test/Dataset/large1_5 \
        --out datasets/small_site_synth --train-ratio 0.9 --seed 0
"""

import argparse
import pickle
import random
import shutil
import sys
from pathlib import Path

SKIP_DIRS = {"Depth Map", "Semantic Segmentation", "2DBBs", "Debug_Blend_Files"}


def find_scenarios(roots):
    """Scenario folders = folders containing Joint_Tracker.pickle."""
    scenarios = []
    for root in roots:
        if not root.exists():
            sys.exit(f"ERROR: path not found: {root}")
        if (root / "Joint_Tracker.pickle").is_file():
            scenarios.append(root)
        scenarios += sorted(p.parent for p in root.rglob("Joint_Tracker.pickle")
                            if p.parent != root)
    # de-duplicate, keep order
    seen, unique = set(), []
    for s in scenarios:
        if s.resolve() not in seen:
            seen.add(s.resolve())
            unique.append(s)
    return unique


def scenario_images(scenario: Path):
    """Rendered frames only (depth/segmentation/debug subfolders excluded)."""
    return sorted(p for p in scenario.glob("*.jpg")
                  if p.parent.name not in SKIP_DIRS)


def derive_labels(scenario: Path) -> dict:
    """Write 2DBBs/<image>.txt for every rendered frame; return stats.

    Box handling follows the paper's extraction script (2DBB_all.py):
    skip boxes fully outside the frame, clamp the rest, normalize to
    [cx, cy, w, h].
    """
    with open(scenario / "Joint_Tracker.pickle", "rb") as fh:
        capture = pickle.load(fh)

    label_dir = scenario / "2DBBs"
    label_dir.mkdir(exist_ok=True)

    workers = capture.get("workers_name_list", [])
    stats = {"images": 0, "boxes": 0, "skipped_boxes": 0}

    for img in scenario_images(scenario):
        frame_key = str(int(img.stem[-4:]))          # test0101 -> '101'
        lines = []
        entry = capture.get(frame_key)
        if entry is not None:
            render_size = entry["render_size"]
            for key in entry:
                if not any(name in key for name in workers):
                    continue
                (x1, y1), (x2, y2) = entry[key]["BB2D"]

                # Skip boxes completely outside the image.
                if (x1 >= render_size[0] or x2 < 0
                        or y1 >= render_size[1] or y2 < 0):
                    stats["skipped_boxes"] += 1
                    continue

                # Clamp to image bounds.
                sx, sy = max(0, x1), max(0, y1)
                ex = min(render_size[0], x2 + 1)
                ey = min(render_size[1], y2)

                cx = (sx + ex) / 2 / render_size[0]
                cy = (sy + ey) / 2 / render_size[1]
                w = (ex - sx) / render_size[0]
                h = (ey - sy) / render_size[1]
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["boxes"] += 1

        (label_dir / f"{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
    return stats


def assemble_dataset(scenarios, out: Path, train_ratio: float, seed: int,
                     link: bool):
    """Collect (image, label) pairs from all scenarios into one YOLO tree."""
    pairs = []
    for scenario in scenarios:
        for img in scenario_images(scenario):
            lbl = scenario / "2DBBs" / f"{img.stem}.txt"
            if lbl.is_file():
                pairs.append((scenario.name, img, lbl))

    if not pairs:
        sys.exit("ERROR: no image/label pairs found")

    random.Random(seed).shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    splits = {"train": pairs[:n_train], "val": pairs[n_train:]}

    put = (lambda src, dst: dst.symlink_to(src.resolve())) if link \
        else shutil.copy2
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for scen_name, img, lbl in items:
            stem = f"{scen_name}__{img.stem}"
            put(img, out / "images" / split / f"{stem}{img.suffix}")
            put(lbl, out / "labels" / split / f"{stem}.txt")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"# PASS synthetic worker-detection dataset "
        f"({len(scenarios)} BlendCon scenario(s))\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: worker\n")
    return splits, yaml_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="BlendCon Dataset root(s) and/or scenario folder(s)")
    p.add_argument("--out", type=Path, required=True,
                   help="YOLO dataset directory to create")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--link", action="store_true",
                   help="Symlink instead of copying images/labels")
    args = p.parse_args()

    scenarios = find_scenarios(args.runs)
    if not scenarios:
        sys.exit("ERROR: no scenario folders (with Joint_Tracker.pickle) found")
    print(f"Found {len(scenarios)} scenario(s)")

    total = {"images": 0, "boxes": 0, "skipped_boxes": 0}
    for scenario in scenarios:
        stats = derive_labels(scenario)
        print(f"  {scenario.name}: {stats['images']} images, "
              f"{stats['boxes']} boxes "
              f"({stats['skipped_boxes']} out-of-frame skipped)")
        for k in total:
            total[k] += stats[k]

    splits, yaml_path = assemble_dataset(scenarios, args.out,
                                         args.train_ratio, args.seed,
                                         args.link)
    print(f"\nDataset assembled: {args.out}")
    print(f"  train: {len(splits['train'])} images | "
          f"val: {len(splits['val'])} images | "
          f"boxes: {total['boxes']}")
    print(f"  YAML: {yaml_path}")
    print("\nTrain with:  yolo detect train data="
          f"{yaml_path} model=yolov10s.yaml "
          "epochs=800 batch=64 patience=25 imgsz=640 lr0=0.001 lrf=0.01")


if __name__ == "__main__":
    main()
