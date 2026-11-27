# Two-set optical GLINT

This experimental method has exactly two independently optimized Gaussian sets:

* **General**: scene geometry, opacity, and SH radiance, shared by direct,
  transmitted, and reflected observations.
* **Optical**: surface geometry, opacity/coverage, SH diffuse radiance,
  conditional transparency `t`, and specularity `s`. Both secondary paths
  originate on this same surface. There are no separate transmission or
  reflection geometry banks.

The original three-set GLINT entry points and checkpoints are unchanged. The
internal legacy roles `reflection` and `interface` select compatible CUDA
parameter layouts; they do not create additional physical sets. Like the
upstream main branch, both sets use **2D Gaussian surfels**, not volumetric 3DGS.

## Selection and training

1. Warm up ordinary GS for 1,000 steps. Geometry uses the existing gsplat
   rasterizer and refinement, with RGB L1/SSIM and optional normal supervision.
2. Project the warmed-up Gaussian centers into up to 24 distinct training
   cameras. A candidate must have `mean(diffuse_albedo) < 0.05` in at least two
   distinct cameras and in at least 60% of its valid observations. Behind-camera,
   off-image, masked, sky, invalid-prior, and occluded points do not vote. The
   occlusion check uses rendered metric z-depth, not monocular relative depth.
   Bright basecolor is **not** mandatory, so mirrors are eligible.
3. **Move**, rather than duplicate, candidates to the optical set. Initialize
   `t=0.5`, `s=0.1`. Train both paths for 2,000 probation steps. Optical SH stays
   fixed during this trial so material improvement is compared to the warmed-up
   diffuse hypothesis. Geometry and material gradients remain enabled. Both
   topologies are temporarily frozen so candidates retain their identities.
4. In one bounded multi-view pass, compare the optical composition to an opaque
   composition using the **same optical surface color, coverage and geometry**.
   Confirm only if mean RGB-L1 improvement is at least 0.005 and this improvement
   is supported in at least 60% of at least two visible camera views. Repeated
   frames from one camera cannot manufacture confidence. Return failed and
   insufficiently observed candidates to the general set.
5. Continue joint optimization and resume GLINT densification/pruning on both
   sets. Newly split optical children inherit their parent's material. Adam
   moments are retained when points move; newly introduced materials start with
   fresh optimizer state.
   The optical refiner preserves learned transparency during opacity resets.

These are configurable initialization values, not claimed optimal thresholds.
There is one candidate/probation cycle per run. Checkpoints preserve the phase,
optimizer/refiner state, thresholds, and validation summary. A short run ending
in probation is not a confirmed model. Missing priors produce no candidates.

## Rendering

Rasterize general and optical sets. Let `a` be optical coverage, `C_g` the
direct general render, and `C_o` the optical surface's conditional diffuse
color. When the optical surface is in front, using Schlick Fresnel `F`:

```text
r = s + (1-s) F
d = (1-r) (1-t)
q = (1-r) t
C = (1-a) C_g + a (d C_o + r C_reflect + q C_transmit)
d + r + q = 1
```

When the ordinary depth layer is in front, let `A_g` be its coverage and `B`
the background color. Set `a_eff = (1-A_g) a` and compose
`C = C_g + a_eff (d C_o + r C_reflect + q C_transmit - B)`.
This preserves the ordinary foreground's premultiplied color and replaces
only the background visible through it. A fully opaque ordinary foreground
suppresses the optical contribution. The same rule is used for the opaque
comparison hypothesis during candidate confirmation.

Both `C_reflect` and `C_transmit` trace **the same general Gaussian object**;
OptiX shares its BVH between the two queries. Reflection uses the optical
normal, and transmission follows GLINT's optically thin straight ray. Origins
use detached optical depth and a small scene-scaled ray offset. Only pixels
with nonzero visible optical coverage are packed into the tracer; results are
scattered back while preserving autograd and refinement statistics. Empty
optical sets skip optical rasterization and tracing entirely.

This retains CUDA rasterization/OptiX and avoids a third geometry bank or a
second BVH for reflected radiance. Actual speed must still be benchmarked:
two raster passes and up to two ray queries remain. Candidate/confirmation
passes run only at phase boundaries, not every training step.

## Run

The uploaded scene is `/root/data/3d-front-t/scene_3`. Convenience commands on
the prepared server are `bash scripts/train_two_set_scene3.sh smoke` (160 steps,
reduced resolution, deliberately shortened probation) and
`bash scripts/train_two_set_scene3.sh train` (the full default schedule).
The smoke run checks execution, not convergence or reconstruction quality.

Prepared server runtime: Python 3.10, PyTorch 2.7.1+cu126, gsplat commit
`83a0dcd3f850cccdc7b92b50fd6a568e3260eae1`, and a private CUDA 12.9 toolkit at
`/root/cuda-12.9-glint`. OptiX was built with the system CUDA 12.1 to keep its
PTX compatible with driver 550.107.02; matching OptiX driver libraries live in
`.optix-runtime`. The activation helper selects these paths. The system driver
and system CUDA were not replaced. The dependency's two invalid CUDA event
creation calls were corrected using
[`patches/gsplat-cuda-event-api.patch`](patches/gsplat-cuda-event-api.patch);
the patch is already applied to `/root/glint-gsplat`. It changes host utilities,
not rasterization kernels. Installation logs are `.venv-*-install.log`.

Install the upstream dependencies using the README. On the prepared server,
use `/root/GLINT/.venv/bin/python` after installation verification succeeds.
Supply an actual scene directory and its corresponding experiment config:

```bash
cd /root/GLINT
source .runtime-env.sh  # prepared server: venv and private OptiX libraries
PYTHON=/root/GLINT/.venv/bin/python
$PYTHON -m glint.train_two_set \
  --config configs/exps/glint/3d-front-t/scene_3.yaml \
  --data-root /path/to/3d-front-t/scene_3 \
  --output-dir results/two_set/scene_3 \
  --warmup-steps 1000 --probation-steps 2000 \
  --albedo-threshold 0.05 --min-views 2 \
  --support-fraction 0.6 --min-improvement 0.005 \
  --max-steps 60000
```

Optional `--ply` overrides `sparse/0/points3D.ply`. Reflection-environment PLY is
not required: radiance support must be present in the shared general geometry.
Use `--environment-ply /path/to/scene/envs/points3D.ply` to merge optional
environment seeds into that same general set, without creating a third set.
`--max-points` applies separately to each input PLY.
`--scene-scale` controls position learning rate, refinement limits and ray
offset; it defaults to the experiment's `spatial_scale`. `--no-refinement` helps
isolate transport ablations. Each set is capped by `--max-gaussians`.

```bash
$PYTHON -m glint.train_two_set \
  --config configs/exps/glint/3d-front-t/scene_3.yaml \
  --data-root /path/to/3d-front-t/scene_3 \
  --output-dir results/two_set/scene_3 \
  --resume results/two_set/scene_3/checkpoints/latest.pt --max-steps 60000

$PYTHON -m glint.evaluate_two_set \
  --config configs/exps/glint/3d-front-t/scene_3.yaml \
  --data-root /path/to/3d-front-t/scene_3 \
  --checkpoint results/two_set/scene_3/checkpoints/latest.pt \
  --output-dir results/two_set/scene_3/eval --save-components

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PYTHON -m pytest -q \
  tests/test_optical_selection.py tests/test_two_set_*.py
```

Checkpoints use `glint-two-set-v1`; use the new evaluation entry point, not the
legacy three-set loader. Resume restores the saved method schedule/settings;
output location and final iteration count can change. RNG state is saved, but
the shuffled dataloader cursor is not, so resume is not bitwise identical.

## Scope and limitations

Validation on the prepared RTX 3090 server (2026-11-15): all **77 tests passed**,
including native CUDA/OptiX forward/backward and shared-BVH checks. A scene_3
smoke run at ratio 0.125 (120x68), 40,000 initial points, 50 warmup + 100
probation + 10 confirmed steps completed. Of 107 candidates, 55 were accepted;
52 returned to the general set (33 lacked enough visible views). Final counts
were 39,945 general and 55 optical. Checkpoint evaluation completed on all 25
validation cameras, and checkpoint resume completed through step 170.

The smoke run disables topology changes and is deliberately too short to
measure converged quality. Its 160-step checkpoint, images and metrics are at
`results/two_set/scene_3_smoke`; resumed output is at
`results/two_set/scene_3_smoke_resume`. These results do not establish improvement
over canonical GLINT or performance at the full training resolution.

This is an implemented research baseline, not a measured quality/speed gain.
It preserves the rasterizer/tracer and refinement machinery, but uses an
explicit new RGB/normal training objective; it does not silently carry over
three-set-specific losses, secondary reflection, or the entire original loss
schedule. Experiment YAML supplies dataset/view splits, while the new CLI
controls this method's training settings.

Foreground ordering uses two rendered depth layers. It preserves partial
foreground coverage for this two-layer model but approximates interleaved
layers; it is not exact
joint per-surfel depth sorting. Optical-to-optical bounces, nested glass, and
Snell refraction are not included. The method needs candidate surface geometry
in the initial point cloud: albedo projection alone cannot create a missing
glass surface, and content visible only in reflections may be absent from the
shared geometry. These are important dataset/ablation checks before evaluating
the representation against original GLINT.

## Direct-beta comparison run

Use `bash scripts/train_two_set_direct_beta_scene3.sh` for a fresh 60,000-step run.
Only the optical mixture changes: `wr=beta`, `wt=(1-beta)*t`,
`wd=(1-beta)*(1-t)`, with beta supplied by the existing sigmoid specular
parameter (initially 0.1) and t initially 0.5. Coverage, occlusion and the
shared general-set BVH remain unchanged. The original 1,000-step warmup,
2,000-step probation, refinement ending at 21,000 and SH degree increments
every 1,000 steps remain in place.

Checkpoints store `transport_mode` in metadata and configuration; files without
it retain the original `fresnel` behavior. Evaluation and resume recover the
saved mode. Direct beta removes the explicit Fresnel constraint, but does not
guarantee correct reflection decomposition or prevent beta collapsing to zero.

## 3D-FRONT-T baseline batch (2026-11-18)

The current batch uses the direct-beta training/model/transport code from
`7a99838`. It does not enable ray-origin gradients, plane regularization,
color sabotage, or normal propagation. Training is fresh, seed 42, ratio 0.5,
60,000 steps per scene, with the original SH schedule and normal weight 0.1.

The sequential order is **scene_1, scene_2, scene_4, scene_5**. Scene 3 is
explicitly excluded: its previous 60,000-step direct-beta result,
**35.5377531773 dB over 25 validation views**, is recorded as historical data,
not as a fresh measurement from this batch.

| Scene | Training views | Validation views | Action |
| --- | ---: | ---: | --- |
| 1 | 149 | 22 | Train |
| 2 | 158 | 23 | Train |
| 3 | 175 | 25 | Reuse historical result |
| 4 | 140 | 20 | Train |
| 5 | 157 | 23 | Train |

After installing the CUDA/OptiX environment described in the README, prepare
`/root/scene_{1,2,4,5}-only.zip` and corresponding `.sha256` files. Each archive
must contain a `scene_N/` root with the released dataset layout. From the repo:

```bash
PYTHONPATH=. python scripts/prepare_front_t_batch.py
python scripts/train_front_t_batch.py --dry-run
python scripts/train_front_t_batch.py
```

The preparation command verifies archive hashes, dataset paths, view splits,
and representative RGB/albedo/normal tensors. It generates the required
`experiments/front_t_preflight.json`. Run the batch under a process manager
such as supervisor for unattended operation. Its status is written atomically
to `results/front_t_direct_beta_baseline/status.json`; each scene has its own
training log, checkpoint directory, and `periodic_psnr/history.json`.

PSNR uses **all official validation views** every 5,000 steps. The evaluator
has low CPU priority and a 0.5-second pause between frames, but shares the GPU
and may cause brief contention. The batch retains the latest evaluated
checkpoint and final checkpoint; older evaluated checkpoint files are removed
after successful evaluation, while all metric JSON files and evaluation images
remain. A failed training/evaluation stops the queue. Restarting the batch skips
completed scenes and resumes the interrupted scene from its own checkpoint.

The active server uses Python 3.12, PyTorch 2.9.1+cu126, CUDA toolkit 12.6,
and an RTX 3090. On older CUDA toolkits, apply both patches under `patches/`
to the pinned gsplat checkout before building: `gsplat-cuda-event-api.patch`
and `gsplat-cuda121-ceil-div.patch`. The latter also supports CUDA 12.6; it
replaces unavailable `cuda::ceil_div` calls with equivalent nonnegative integer
division without changing the rendering model.

Validation on this server: renderer installation forward/backward check passed,
31 two-set training/transport/refinement/GPU tests passed, and both batch
orchestration tests passed. This validates execution, not new final quality
results for scenes 1, 2, 4, or 5.
