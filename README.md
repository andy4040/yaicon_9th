# 3D Gaussian Splatting for Transparent Scene Reconstruction

YAICON 9th project.<br>
Built on [GLINT](https://github.com/youngju-na/GLINT), our model represents a scene with **two Gaussian sets: general GS and optical GS**.

In glass scenes, reflections can be learned as objects behind the glass or as colors attached to its surface. We separate scene geometry and color from the surfaces that determine reflection and transmission paths.

## Model

GLINT separates interface, transmission, and reflection Gaussians. Our optical GS generates reflected and transmitted rays, both of which query the same general GS for color.

| Set | Role |
|:--|:--|
| General GS | Scene geometry and color, shared by direct views and reflected/transmitted rays |
| Optical GS | Shared surface positions and normals for reflection and transmission, with learned material coefficients |

### Confirmed Optical Gaussians

Confirmed optical GS in 3D-FRONT-T scenes 1–5. Each row shows **ground truth / optical GS overlay / optical GS alone**. Cyan indicates projected optical coverage.

<img src="docs/assets/confirmed-optical-gaussians.png" alt="Scenes 1–5: ground truth, optical GS overlay, and confirmed optical GS alone" width="100%">

### Optical Candidate Selection

All Gaussians start in the general set. We then project their centers into up to 24 training views and check diffuse albedo. After excluding observations outside the image or rejected by visibility checks, a Gaussian becomes an optical candidate if:

- At least two views satisfy `mean(diffuse_albedo) < 0.05`.
- These views account for at least 60% of valid observations.

Dark opaque objects can also have low albedo, so candidates are not accepted immediately. After a trial training period, we compare RGB L1 errors under opaque and optical rendering at the same locations. Candidates that improve reconstruction across views remain in the optical set; the rest return to the general set. The default improvement threshold is `0.005`.

This version makes a single acceptance decision, without roughness filtering or candidate reevaluation. See [optical_selection.py](glint/optical_selection.py) for the full criteria.

### Reflection and Transmission

We blend the optical surface color as:

$$
C_{\mathrm{opt}}=\beta L_r+(1-\beta)\left[tL_t+(1-t)C_{\mathrm{surface}}\right]
$$

Here, $L_r$ and $L_t$ are colors queried from the general GS along reflected and transmitted rays, and $C_{\mathrm{surface}}$ is the surface color. We learn the reflection coefficient $\beta$ and conditional transmission coefficient $t$. The three blending weights sum to one.

This mode is called `direct_beta` in the code. It uses the learned reflection coefficient directly, without a Fresnel multiplier. Reflection follows the surface normal, while transmission travels straight through. Optical coverage and foreground occlusion determine the final composition with the general GS image.

### Training Stages

| Iterations | Stage |
|:--|:--|
| 0–5,000 | Learn scene geometry and color with general GS |
| 5,000–10,000 | Train optical candidates; pause densification, pruning, and optical SH updates |
| 10,000 | Validate candidates and confirm optical GS |
| 10,000–30,000 | Joint training with densification and pruning |
| 30,000–60,000 | Fix Gaussian topology and continue parameter optimization |

Training uses RGB L1, SSIM, and normal prior losses. The normal loss weight is `0.1` for each set. SSIM contributes to training; candidate acceptance uses RGB L1 improvement.

## Results

### Ref-DL3DV

Results from eight Ref-DL3DV scenes trained for 60,000 iterations each. The overall score is the unweighted mean of the per-scene validation averages.

| Scene ID | Validation views | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:--|--:|--:|--:|--:|
| `b65e8683` | 47 | 31.127 | 0.9388 | 0.0518 |
| `b9df30d6` | 46 | 29.594 | 0.9154 | 0.0631 |
| `52410f02` | 48 | 35.804 | 0.9645 | 0.0472 |
| `6b42314a` | 41 | 30.334 | 0.9404 | 0.0623 |
| `543b6607` | 47 | 29.193 | 0.9266 | 0.0582 |
| `194defaa` | 40 | 26.466 | 0.8758 | 0.1058 |
| `3712b8fd` | 51 | 27.520 | 0.8952 | 0.0999 |
| `5454b71d` | 38 | 35.222 | 0.9807 | 0.0152 |
| **Mean** | **358 total** | **30.657** | **0.9297** | **0.0629** |

Scene IDs are shortened to eight characters. SSIM and LPIPS were computed from saved PNGs, using AlexNet for LPIPS.

### Comparison with GLINT

| Model | Ref-DL3DV PSNR ↑ | SSIM ↑ | 3D-FRONT-T PSNR ↑ | SSIM ↑ |
|:--|--:|--:|--:|--:|
| GLINT | 30.21 | 0.92 | **34.50** | **0.96** |
| Ours | **30.66** | **0.93** | 34.08 | 0.95 |

Results presented at YAICON. GLINT scores are reported in the paper; ours come from project experiments, rather than a rerun under identical conditions. See [SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md) for the code baseline and experiment records.

### Limitations

Separating reflection from transmission remains difficult. Lower RGB error does not prevent reflected content from being learned through the transmission path, and inaccurate normals lead to incorrect reflection directions. Surfaces missed during initial selection also need a way to enter the optical set later. Straight-through transmission does not model refraction through thick glass.

## Usage

### Installation

Requires Linux, an NVIDIA GPU, and the CUDA toolkit. Python 3.11 or 3.12 is recommended.

Rendering: GLINT's gsplat fork + OptiX. Training: `TwoSetTrainer`.

```bash
git clone --recursive https://github.com/andy4040/yaicon_9th.git
cd yaicon_9th
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

export CUDA_HOME=/usr/local/cuda
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
BUILD_2DGS=1 NUM_CHANNELS=3,4,6,7 BUILD_EXPERIMENTAL=0 \
  pip install --no-build-isolation \
  'git+https://github.com/youngju-na/gsplat.git@83a0dcd3f850cccdc7b92b50fd6a568e3260eae1'
pip install -r requirements.txt
pip install --no-deps -e .
python -m glint.install_optix
python -m glint.verify_installation
```

Set `CUDA_HOME` to your CUDA installation path. The loader reads EasyVolCap camera and configuration formats, but training uses our own loop.

### Data

Download Ref-DL3DV from the [official GLINT repository](https://github.com/youngju-na/GLINT) to your GPU server. Keep the original directory structure, including RGB images, cameras, `sparse/0/points3D.ply`, and DiffusionRenderer priors.

```text
/path/to/ref-dl3dv/
├── b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d/
├── b9df30d6e6078880acc88acb01872c65d337f84b9dba44a23fa29c9861d7e23b/
└── ...
```

Full scene IDs and train/validation splits are in [configs/datasets/ref-dl3dv](configs/datasets/ref-dl3dv). The launcher uses an image scale of `0.5` and evaluates all configured validation views. Datasets and trained checkpoints are not included.

### Training

```bash
# Preview commands
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set --dry-run

# Train all eight scenes and evaluate each after training
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set

# Resume from checkpoints
python scripts/train_ref_dl3dv_two_set.py \
  --data-root /path/to/ref-dl3dv --output-root results/ref_dl3dv_two_set --resume
```

Add `--scenes FULL_SCENE_ID` to run a single scene. Commands are saved in `launch_commands.json`. Every 5,000 iterations, the checkpoint overwrites `checkpoints/latest.pt`; copy it separately to keep intermediate results.

Use `glint.train_two_set` for our model. The legacy `train_ref_dl3dv_6b42.sh` calls the original GLINT trainer, `glint.train`.

### Evaluation and Rendering

```bash
SCENE=b65e86833c1ae29714ce881bb9d14d3ed1256a08ab944fd9e75d6b29c674346d
python -m glint.evaluate_two_set \
  --config configs/exps/glint/ref-dl3dv/${SCENE}.yaml \
  --data-root /path/to/ref-dl3dv/${SCENE} \
  --checkpoint results/ref_dl3dv_two_set/${SCENE}/checkpoints/latest.pt \
  --output-dir results/ref_dl3dv_two_set/${SCENE}/eval \
  --ratio 0.5 --split val --save-components
```

The evaluator saves PSNR and L1 in `metrics.json`, along with final RGB, ground truth, general GS, optical coverage, and reflection/transmission contributions. These contributions include material weights and coverage. This command does not compute SSIM or LPIPS.

## Code

| File | Purpose |
|:--|:--|
| [two_set_model.py](glint/two_set_model.py) | Gaussian set management and checkpoints |
| [optical_selection.py](glint/optical_selection.py) | Optical candidate selection and validation |
| [two_set_transport.py](glint/two_set_transport.py) | Reflection/transmission tracing and composition |
| [train_two_set.py](glint/train_two_set.py) | Training stages, losses, and optimizers |
| [two_set_refinement.py](glint/two_set_refinement.py) | Optical GS densification and pruning |
| [evaluate_two_set.py](glint/evaluate_two_set.py) | Evaluation and component rendering |

## Acknowledgments

Built on [GLINT](https://github.com/youngju-na/GLINT), using its gsplat fork for rasterization and diff-surfel-tracing with NVIDIA OptiX for ray tracing. See the [upstream README](docs/UPSTREAM_README.md), [LICENSE](LICENSE), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
