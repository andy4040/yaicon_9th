# Source provenance

This repository restores the two-set direct-beta baseline from
[andy4040/yai5, commit 3f26fc5b3d54ce59310b4595b2e92fa70388bad5](https://github.com/andy4040/yai5/commit/3f26fc5b3d54ce59310b4595b2e92fa70388bad5).

The `glint/` sources and dataset configurations are preserved from that commit.
The local November 23 source backup has the same `glint/` implementation after
line-ending normalization. The restored model uses gsplat and OptiX with its own
TwoSetTrainer, not the later native EasyVolCap runner.

## Repository audit

| Repository | Commit | Finding |
| --- | --- | --- |
| yai5 | `3f26fc5` | Two-set general/optical model; direct-beta transport; Ref-DL3DV scene configurations. Chosen baseline. |
| yai4 | `7a99838` | Earlier two-set baseline. Its `glint/` directory differs from yai5 only in evaluation and HTML-viewer updates. |
| glintyai | `fa3259c` | Earlier shared-reflection/material-first experiments; no `two_set_model.py`. |

All three include Ref-DL3DV configurations. That alone does not identify a
completed training run. The legacy `train_ref_dl3dv_6b42.sh` invokes `glint.train`,
whereas the two-set model uses `glint.train_two_set`.

## Additions in this restoration

- A Korean project README based on the supplied YAICON presentation, with previously rendered confirmed optical-GS visualizations from 3D-FRONT-T.
- `scripts/train_ref_dl3dv_two_set.py`: a portable launcher with explicit
  direct-beta mode, 5k/10k/30k/60k schedule, half-resolution images, normal prior
  weight 0.1, and validation over each configured validation split.
- Documentation of previously completed project metrics. These are historical
  reported results, not a new reproduction run performed during restoration.
- Source attribution and Linux-friendly line endings.

The historical experiment server is no longer accessible. This repository is a
baseline restoration; it does not distribute that server's checkpoints or logs.
The presentation's 3D-FRONT-T result belongs to a separate project experiment.

## Dependencies

- gsplat: `83a0dcd3f850cccdc7b92b50fd6a568e3260eae1`
- diff-surfel-tracing: `ef6f24be9174eee037469f7e221cd72f6c0222cb`
- nested OptiX headers: retained through the pinned recursive submodule

The original licenses and author notices are preserved.
