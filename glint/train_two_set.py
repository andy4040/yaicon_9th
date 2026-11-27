"""Train ordinary GS plus shared optical surfaces with a short probation phase.

Run with ``python -m glint.train_two_set --help``. Original GLINT is unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .two_set_model import TwoSetModel, make_optimizers, transfer_optimizer_state, load_two_set, checkpoint_transport_mode
from .optical_selection import OpticalSelectionConfig, OpticalEvidence, select_optical_candidates
from .two_set_transport import render_two_set


class TwoSetTrainer:
    def __init__(self, model, dataset, tracer, args, *, selection_config=None):
        from .refinement import GlintRefiner, RefinementConfig
        from .two_set_refinement import OpticalRefiner
        self.model, self.dataset, self.tracer, self.args = model, dataset, tracer, args
        self.model.transport_mode = checkpoint_transport_mode({"transport_mode":
            getattr(args, "transport_mode", model.transport_mode)})
        self.args.transport_mode = self.model.transport_mode
        self.device = model.background.device
        self.selection_config = selection_config or OpticalSelectionConfig()
        self.optimizers = make_optimizers(model, args.scene_scale)
        self.phase = "warmup"
        self.step = 0
        self.validation_summary = {}
        self.refinement_config = RefinementConfig(
            stop=0 if args.no_refinement else args.refine_stop,
            max_gaussians=args.max_gaussians, min_gaussians=min(1024, args.max_gaussians),
            max_scale_ratio=0.1,
        )
        self._refiner_class = GlintRefiner
        self._optical_refiner_class = OpticalRefiner
        self._reset_refiners()

    def _reset_refiners(self):
        self.refiners = {
            name: (self._optical_refiner_class if name == "optical" else self._refiner_class)(getattr(self.model, name), self.optimizers[name],
                                     self.refinement_config, scene_scale=self.args.scene_scale)
            for name in ("general", "optical")
        }

    def _selection_frames(self):
        # Bounded one-time pass, not an extra multi-view render every iteration.
        entries = getattr(self.dataset, "entries", None)
        if entries is not None:
            first_per_camera = {}
            for index, (camera_name, _) in enumerate(entries):
                first_per_camera.setdefault(camera_name, index)
            available = list(first_per_camera.values())
        else:
            available = list(range(len(self.dataset)))
        indices = torch.linspace(0, len(available)-1,
                                 min(len(available), self.args.selection_views)).long().tolist()
        for index in indices:
            yield self.dataset[available[index]].to(self.device)

    @torch.no_grad()
    def promote(self):
        # Rendered z-depth avoids giving hidden background points the material
        # label of the optical surface in front of them.
        def frames():
            for frame in self._selection_frames():
                output = render_two_set(frame.camera, self.model, self.tracer, optical_enabled=False)
                depths.clear()
                depths[frame.camera_name] = output.general.surf_depth.permute(1, 2, 0)
                yield frame
        depths = {}
        mask, diagnostic = select_optical_candidates(
            self.model.general.get_xyz, frames(), self.selection_config, visibility_depths=depths)
        if bool(mask.all()):
            # A scene entirely labelled optical has no radiance support. Fail
            # explicitly instead of silently inventing a third environment set.
            raise ValueError("All Gaussians selected as optical; adjust priors/threshold or add ordinary scene points")
        count = int(mask.sum())
        old = self.optimizers
        self.model.promote(mask)
        self.optimizers = make_optimizers(self.model, self.args.scene_scale)
        transfer_optimizer_state(old, self.optimizers, mask, promotion=True)
        self._reset_refiners()
        self.phase = "probation" if count else "confirmed"
        self.validation_summary = {"candidates": count}
        print(f"[optical] candidates={count} general={len(self.model.general.get_xyz)}", flush=True)

    @torch.no_grad()
    def confirm(self):
        evidence = OpticalEvidence(len(self.model.optical.get_xyz), self.selection_config, device=self.device)
        for frame in self._selection_frames():
            output = render_two_set(frame.camera, self.model, self.tracer,
                                    ray_epsilon=self.args.ray_epsilon * self.args.scene_scale)
            evidence.update(self.model.optical.get_xyz, frame,
                            ordinary_rgb=output.opaque_rgb,
                            optical_rgb=output.render.permute(1, 2, 0),
                            optical_gate=output.optical_gate,
                            visibility_depth=output.optical.surf_depth.permute(1, 2, 0))
        accepted, decided = evidence.decision()
        accepted &= decided
        self.validation_summary.update(accepted=int(accepted.sum()),
                                       insufficient_views=int((~decided).sum()),
                                       returned_to_general=int((~accepted).sum()))
        old = self.optimizers
        self.model.reject(accepted)
        self.optimizers = make_optimizers(self.model, self.args.scene_scale)
        transfer_optimizer_state(old, self.optimizers, accepted, promotion=False)
        self._reset_refiners()
        self.phase = "confirmed"
        print(f"[optical] {self.validation_summary}", flush=True)

    def loss(self, output, frame):
        from .losses import ssim_loss
        target = frame.rgb.permute(2, 0, 1)
        l1 = (output.render-target).abs().mean()
        loss = 1.2*l1 + 0.4*ssim_loss(output.render[None], target[None])
        # Geometry supervision stays on the visible surface, in the same
        # camera-space convention as GLINT's DiffusionRenderer normal prior.
        if frame.normal is not None and self.args.normal_weight > 0:
            for name in ("general", "optical"):
                raster = getattr(output, name)
                if raster is None:
                    continue
                normal = F.normalize(raster.rend_normal.permute(1, 2, 0) @ frame.camera.R.T, dim=-1)
                decoded = frame.normal * 2.0 - 1.0
                prior = F.normalize(torch.nan_to_num(decoded), dim=-1)
                visible = (1-output.optical_gate if name == "general" else output.optical_gate).detach()[..., 0]
                valid = (decoded.norm(dim=-1) > 0.1) & torch.isfinite(decoded).all(-1)
                weight = visible * valid
                normal_error = 1-(normal*prior).sum(-1)
                loss = loss + self.args.normal_weight*(normal_error*weight).sum()/weight.sum().clamp_min(1)
        return loss, l1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format="glint-two-set-v1", model=self.model.state_dict(),
                       transport_mode=self.model.transport_mode,
                       step=self.step, phase=self.phase, config=vars(self.args),
                       selection_config=asdict(self.selection_config), validation=self.validation_summary,
                       optimizers={r: {k: o.state_dict() for k, o in opts.items()}
                                   for r, opts in self.optimizers.items()},
                       refiners={r: ref.state_dict() for r, ref in self.refiners.items()},
                       rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        # Keep checkpoints safe-loadable: argparse Paths must become strings.
        payload["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in payload["config"].items()}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def restore(self, payload):
        # Schedule and topology settings must follow the saved run, even when
        # CLI defaults differ. Output location and final step can be overridden.
        for key, value in payload["config"].items():
            if key not in {"output_dir", "max_steps", "resume", "data_root", "config", "num_workers"}:
                setattr(self.args, key, value)
        self.model.transport_mode = checkpoint_transport_mode(payload)
        self.args.transport_mode = self.model.transport_mode
        self.selection_config = OpticalSelectionConfig(**payload["selection_config"])
        from dataclasses import replace
        self.refinement_config = replace(self.refinement_config,
            stop=0 if self.args.no_refinement else self.args.refine_stop,
            max_gaussians=self.args.max_gaussians)
        self._reset_refiners()
        self.step, self.phase = int(payload["step"]), payload["phase"]
        self.validation_summary = payload.get("validation", {})
        for role, opts in payload["optimizers"].items():
            for key, state in opts.items():
                self.optimizers[role][key].load_state_dict(state)
        for role, state in payload["refiners"].items():
            self.refiners[role].load_state_dict(state)
        torch.set_rng_state(payload["rng"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])

    def train(self):
        from .dataset import make_glint_dataloader
        loader = make_glint_dataloader(self.dataset, batch_size=1, shuffle=True,
                                      num_workers=self.args.num_workers)
        iterator = iter(loader)
        started = time.perf_counter()
        self.model.train()
        for step in range(self.step, self.args.max_steps):
            if self.phase == "warmup" and step >= self.args.warmup_steps:
                self.promote()
            if self.phase == "probation" and step >= self.args.warmup_steps+self.args.probation_steps:
                self.confirm()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            frame = batch.samples[0].to(self.device)
            for role, opts in self.optimizers.items():
                gaussian = getattr(self.model, role)
                degree = min(int(math.sqrt(gaussian.get_features.shape[1]))-1, step//1000)
                gaussian.active_sh_degree.fill_(degree)
                for key, opt in opts.items():
                    opt.zero_grad(set_to_none=True)
                    if key == "means":
                        progress = min(step/30000, 1.0)
                        opt.param_groups[0]["lr"] = self.args.scene_scale*1.6e-4*(0.01**progress)
            output = render_two_set(frame.camera, self.model, self.tracer,
                                    ray_epsilon=self.args.ray_epsilon*self.args.scene_scale)
            for role, ref in self.refiners.items():
                raster = getattr(output, role)
                if raster is not None:
                    ref.retain_interface_gradient(SimpleNamespace(interface=raster))
            loss, l1 = self.loss(output, frame)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            for role, ref in self.refiners.items():
                raster = getattr(output, role)
                if raster is not None:
                    ref.accumulate_interface(SimpleNamespace(interface=raster))
            self.refiners["general"].accumulate_trace(output.transmission_trace)
            self.refiners["general"].accumulate_trace(output.reflection_trace)
            for role, opts in self.optimizers.items():
                for key, opt in opts.items():
                    # Keep the warmed-up diffuse hypothesis fixed during the
                    # trial, so optical improvement is not a moving SH target.
                    if self.phase == "probation" and role == "optical" and key in {"sh0", "shN"}:
                        continue
                    opt.step()
            # Freeze both topologies during probation so decisions retain
            # their identity; resume the existing GLINT refiner afterwards.
            if self.phase != "probation":
                for role, ref in self.refiners.items():
                    if len(getattr(self.model, role).get_xyz):
                        ref.step(step)
            else:
                # Freezing topology must not disable scale/quaternion safety
                # bounds while candidate geometry is still being optimized.
                for role, ref in self.refiners.items():
                    if len(getattr(self.model, role).get_xyz):
                        ref.stabilize_covariance()
            self.step = step+1
            if self.step % self.args.log_every == 0:
                record = dict(step=self.step, phase=self.phase, loss=float(loss.detach()),
                              l1=float(l1.detach()), elapsed_seconds=time.perf_counter()-started,
                              general=len(self.model.general.get_xyz), optical=len(self.model.optical.get_xyz))
                print(json.dumps(record), flush=True)
                with (Path(self.args.output_dir)/"train.jsonl").open("a") as log:
                    log.write(json.dumps(record)+"\n")
            if self.args.save_every and self.step % self.args.save_every == 0:
                self.save(Path(self.args.output_dir)/"checkpoints"/"latest.pt")
        self.save(Path(self.args.output_dir)/"checkpoints"/"latest.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--environment-ply", type=Path,
                        help="Optional environment points merged into the same general set")
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--transport-mode", choices=("fresnel", "direct_beta"), default="fresnel")
    for name, default in (("max-steps",60000),("warmup-steps",1000),("probation-steps",2000),
                          ("selection-views",24),("min-views",2),("max-points",100000),
                          ("max-gaussians",1000000),("refine-stop",21000),
                          ("log-every",10),("save-every",1000),("num-workers",4),("seed",42)):
        parser.add_argument("--"+name, type=int, default=default)
    parser.add_argument("--albedo-threshold", type=float, default=0.05)
    parser.add_argument("--support-fraction", type=float, default=0.6)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    parser.add_argument("--normal-weight", type=float, default=0.1)
    parser.add_argument("--scene-scale", type=float,
                        help="Defaults to model_cfg.sampler_cfg.spatial_scale from the experiment")
    parser.add_argument("--ray-epsilon", type=float, default=1e-4)
    parser.add_argument("--no-refinement", action="store_true")
    args = parser.parse_args()
    if args.scene_scale is None:
        from .dataset import load_easyvolcap_config
        experiment = load_easyvolcap_config(args.config)
        args.scene_scale = float(experiment.get("model_cfg", {}).get("sampler_cfg", {}).get("spatial_scale", 1.0))
    for key in ("max_steps", "probation_steps", "selection_views", "min_views", "max_points", "max_gaussians", "log_every"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.warmup_steps < 0 or args.num_workers < 0 or args.save_every < 0 or args.refine_stop < 0:
        parser.error("step counts and num_workers must be non-negative")
    if args.scene_scale <= 0 or args.ray_epsilon <= 0 or (args.ratio is not None and args.ratio <= 0):
        parser.error("scene-scale, ray-epsilon and ratio must be positive")
    if args.selection_views < args.min_views:
        parser.error("selection-views must be at least min-views")
    from .dataset import GlintDataset
    from .optix_backend import OptixSurfelTracer
    overrides = {"load_neighbors": False}
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.ratio is not None:
        overrides["ratio"] = args.ratio
    dataset = GlintDataset.from_config(args.config, split="train", **overrides)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA/gsplat/OptiX; CPU is for unit tests only")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, payload = (load_two_set(args.resume, device="cuda") if args.resume else
                      (TwoSetModel.from_ply(args.ply or dataset.data_root/"sparse/0/points3D.ply",
                                            max_points=args.max_points, seed=args.seed,
                                            environment_ply=args.environment_ply,
                                            transport_mode=args.transport_mode), None))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = OpticalSelectionConfig(albedo_threshold=args.albedo_threshold, min_views=args.min_views,
                                    support_fraction=args.support_fraction, min_improvement=args.min_improvement)
    trainer = TwoSetTrainer(model, dataset, OptixSurfelTracer(), args, selection_config=config)
    if payload is not None:
        trainer.restore(payload)
    trainer.train()


if __name__ == "__main__":
    main()
