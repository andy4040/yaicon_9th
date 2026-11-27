"""Two *physical*, disjoint Gaussian sets for shared optical transport."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .checkpoint import GlintGaussianSet
from .model import create_gaussian_set, load_colmap_ply


def checkpoint_transport_mode(payload):
    mode = payload.get("transport_mode", payload.get("config", {}).get("transport_mode", "fresnel"))
    if mode not in {"fresnel", "direct_beta"}:
        raise ValueError(f"Unknown two-set transport mode: {mode}")
    return mode


def _subset(source, mask, role):
    state = {"set.active_sh_degree": source.active_sh_degree}
    for name in GlintGaussianSet._BASE_PARAMETER_NAMES:
        state[f"set.{name}"] = getattr(source, name)[mask]
    if role == "interface":
        opacity = state["set._opacity"]
        # Non-saturated seeds allow both optical branches to learn immediately.
        state["set._specular"] = torch.full_like(opacity, 0.1).logit()
        state["set._transmission_coeff"] = torch.full_like(opacity, 0.5).logit()
    return GlintGaussianSet(state, "set", role=role, trainable=True)


class TwoSetModel(nn.Module):
    """General radiance and optical surface geometry/material, never three sets.

    Internal legacy roles select existing CUDA feature layouts only. No
    transmission/reflection geometry copies or optimizers are constructed.
    """
    def __init__(self, general, optical, background=None, *, transport_mode="fresnel"):
        super().__init__()
        self.transport_mode = checkpoint_transport_mode({"transport_mode": transport_mode})
        self.general = general
        self.optical = optical
        self.register_buffer("background", general.get_xyz.new_zeros(3) if background is None else background)

    @classmethod
    def from_ply(cls, path, *, device="cuda", sh_degree=3, max_points=None, seed=42,
                 environment_ply=None, transport_mode="fresnel"):
        xyz, rgb = load_colmap_ply(path, max_points=max_points, seed=seed)
        if environment_ply is not None:
            env_xyz, env_rgb = load_colmap_ply(environment_ply, max_points=max_points, seed=seed)
            xyz, rgb = torch.cat((xyz, env_xyz)), torch.cat((rgb, env_rgb))
        general = create_gaussian_set(xyz, rgb, role="reflection", device=device, sh_degree=sh_degree)
        empty = torch.zeros(len(xyz), dtype=torch.bool, device=device)
        return cls(general, _subset(general, empty, "interface"), transport_mode=transport_mode)

    @classmethod
    def from_state(cls, state, *, device="cpu", transport_mode="fresnel"):
        general = GlintGaussianSet(state, "general", role="reflection", trainable=True)
        optical = GlintGaussianSet(state, "optical", role="interface", trainable=True)
        return cls(general, optical, state["background"].clone(), transport_mode=transport_mode).to(device)

    @torch.no_grad()
    def promote(self, mask):
        """Move candidates, without copying their geometry into a third set."""
        if len(self.optical.get_xyz):
            raise ValueError("Promotion requires an empty optical set")
        if mask.shape != (len(self.general.get_xyz),) or mask.dtype != torch.bool:
            raise ValueError("Expected one boolean candidate decision per general Gaussian")
        if bool(mask.all()):
            raise ValueError("At least one general Gaussian must remain for radiance transport")
        self.optical = _subset(self.general, mask, "interface")
        self.general = _subset(self.general, ~mask, "reflection")

    @torch.no_grad()
    def reject(self, accepted):
        """Return failed/provisional optical surfaces to the general set."""
        if accepted.shape != (len(self.optical.get_xyz),) or accepted.dtype != torch.bool:
            raise ValueError("Expected one boolean acceptance decision per optical Gaussian")
        for name in GlintGaussianSet._BASE_PARAMETER_NAMES:
            value = torch.cat((getattr(self.general, name), getattr(self.optical, name)[~accepted]))
            setattr(self.general, name, nn.Parameter(value))
        for name in self.optical.parameter_map():
            raw = self.optical._STANDARD_NAMES[name]
            setattr(self.optical, raw, nn.Parameter(getattr(self.optical, raw)[accepted].clone()))


def make_optimizers(model, scene_scale=1.0):
    rates = dict(means=1.6e-4 * scene_scale, sh0=2.5e-3, shN=1.25e-4,
                 scales=5e-3, quats=1e-3, opacities=5e-2, specular=1e-2, transparency=5e-2)
    return {name: {key: torch.optim.Adam([value], lr=rates[key], eps=1e-15)
                   for key, value in getattr(model, name).parameter_map().items()}
            for name in ("general", "optical")}


def transfer_optimizer_state(old, new, masks, *, promotion):
    """Retain Adam history when partitioning/merging; new material starts fresh."""
    for role, optimizers in new.items():
        for key, optimizer in optimizers.items():
            if promotion:
                sources = [("general", masks if role == "optical" else ~masks)]
            elif role == "general":
                sources = [("general", None), ("optical", ~masks)]
            else:
                sources = [("optical", masks)]
            histories = []
            for source_role, selection in sources:
                source_opt = old[source_role].get(key)
                if source_opt is None:
                    histories = []
                    break
                parameter = source_opt.param_groups[0]["params"][0]
                history = source_opt.state.get(parameter, {})
                histories.append((parameter, history, selection))
            if not histories or not any(h for _, h, _ in histories):
                continue
            target = optimizer.param_groups[0]["params"][0]
            state = {}
            for moment in ("exp_avg", "exp_avg_sq"):
                parts = []
                for parameter, history, selection in histories:
                    value = history.get(moment, torch.zeros_like(parameter))
                    parts.append(value if selection is None else value[selection])
                state[moment] = torch.cat(parts).clone()
            steps = [h["step"] for _, h, _ in histories if "step" in h]
            state["step"] = max(steps, key=lambda x: float(x)).clone()
            optimizer.state[target] = state


def load_two_set(path: str | Path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "glint-two-set-v1":
        raise ValueError("Expected a glint-two-set-v1 checkpoint, not a legacy three-set checkpoint")
    return TwoSetModel.from_state(payload["model"], device=device,
                                  transport_mode=checkpoint_transport_mode(payload)), payload
