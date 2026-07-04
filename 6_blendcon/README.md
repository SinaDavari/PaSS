# Stage 6 — Synthetic image generation (patched BlendCon)

PASS generates annotated synthetic images with
**[BlendCon](https://github.com/Ali-Tohidifar/BlendCon)** by Ali Tohidifar
(MIT license) — a Blender-based synthetic data engine for construction
scenes. Full credit for the engine goes to the upstream project; PASS
contributes **only a camera/worker-placement modification**, shipped here as
a patch. **Upstream BlendCon is not vendored in this repo.**

## What our patch changes

Applied to upstream commit `fb67849`, `blendcon_pass.patch` modifies
`DataGenerator.py` and `config.yaml`:

- **Floor-aware placement** — workers and cameras are only placed at
  positions that lie within the `StandableFloor` mesh produced by stage 5,
  verified by downward ray-casting against the floor boundary. This is what
  makes BlendCon work on *reconstructed* scenes (whose walkable area is an
  irregular subset of the bounding box) rather than hand-modeled ones.
- **Randomized camera radius** — the camera distance is drawn from a
  Gaussian around `Camera_Radius` (σ = radius/3, floor-clamped at 1.5 m)
  instead of being fixed, increasing viewpoint diversity per scene.
- Config updates for the reconstructed-scene workflow (640×640 renders,
  worker count, lighting/placement randomization iterations).

## Usage

```bash
./apply_patch.sh              # clones upstream @ fb67849 and applies the patch
cd BlendCon
# edit config.yaml; place the stage-5 .blend (with StandableFloor) in Scenes/
blender --background --python DataGenerator.py
```

The working directory must contain (all shipped by the upstream clone via
git-LFS): `Avatars/` (rigged worker .blend files), `Horizon.blend`,
`Empty.blend`, and a `Scenes/` folder into which you place the stage-5
canonicalized `.blend`. Renders per scenario = the target avatar's keyframe
range stepped by `Framerate`; total scenarios =
`Number_of_Image_Sequences × Iterations_Avatar_Location_Randomization ×
Iterations_Lighting_Randomization`.

Outputs per run: rendered images, per-frame annotation data
(`Joint_Tracker.json` + pickle: 2D boxes, joints, camera pose), and optional
depth/segmentation maps — consumed by stage 7.

## Attribution

```
BlendCon — https://github.com/Ali-Tohidifar/BlendCon
Copyright (c) 2023 Ali Tohidifar, MIT License
```

If you use this stage, please cite the BlendCon paper (see upstream README)
alongside ours.
