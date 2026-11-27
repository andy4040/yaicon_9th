"""Cheap, detached multi-view evidence for the two-set optical model.

Albedo proposes candidates; independent RGB evidence confirms them.  Camera
views, rather than training iterations or repeated frames, are counted.  All
image arguments use HWC layout and visibility depth is metric camera z-depth
(never the monocular, scale-ambiguous DiffusionRenderer depth prior).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OpticalSelectionConfig:
    albedo_threshold: float = 0.05
    min_views: int = 2
    support_fraction: float = 0.6
    min_improvement: float = 0.005
    visibility_relative_tolerance: float = 0.05
    visibility_absolute_tolerance: float = 0.01
    min_optical_gate: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.albedo_threshold <= 1:
            raise ValueError("albedo_threshold must be in (0, 1]")
        if self.min_views < 2:
            raise ValueError("min_views must be at least two independent cameras")
        if not 0 < self.support_fraction <= 1:
            raise ValueError("support_fraction must be in (0, 1]")
        if self.min_improvement <= 0:
            raise ValueError("min_improvement must be positive to reject dark opaque surfaces")
        if self.visibility_relative_tolerance < 0 or self.visibility_absolute_tolerance < 0:
            raise ValueError("visibility tolerances must be non-negative")
        if not 0 <= self.min_optical_gate <= 1:
            raise ValueError("min_optical_gate must be in [0, 1]")


def _view_name(frame: Any) -> str:
    name = getattr(frame, "camera_name", None) or getattr(frame.camera, "name", None)
    if not name:
        raise ValueError("Optical evidence needs a stable camera_name")
    return str(name)


def _project(xyz: Tensor, camera: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be [N, 3]")
    rotation = camera.R.to(xyz)
    translation = camera.T.to(xyz).reshape(1, 3)
    points = xyz @ rotation.T + translation
    pixels = points @ camera.K.to(xyz).T
    depth = points[:, 2]
    uv = pixels[:, :2] / pixels[:, 2:3].clamp_min(1e-8)
    height, width = int(camera.image_height), int(camera.image_width)
    valid = (
        torch.isfinite(xyz).all(-1)
        & torch.isfinite(uv).all(-1)
        & (depth > getattr(camera, "znear", 1e-4))
        & (depth < getattr(camera, "zfar", 1e6))
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    # GLINT rays pass through pixel centers (i + .5, j + .5).
    safe = torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    x = safe[:, 0].floor().clamp(0, width - 1).long()
    y = safe[:, 1].floor().clamp(0, height - 1).long()
    return x, y, depth, valid


def _sample(image: Tensor, x: Tensor, y: Tensor, camera: Any) -> Tensor:
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[:2] != (camera.image_height, camera.image_width):
        raise ValueError("Image must have HWC layout matching its camera")
    return image.to(device=x.device)[y, x]


def _visible(
    xyz: Tensor, frame: Any, config: OpticalSelectionConfig,
    visibility_depth: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    x, y, depth, valid = _project(xyz, frame.camera)
    mask = getattr(frame, "mask", None)
    if mask is not None:
        sampled = _sample(mask, x, y, frame.camera)
        valid &= torch.isfinite(sampled).all(-1) & (sampled.mean(-1) > 0.5)
    sky = getattr(frame, "sky_mask", None)
    if sky is not None:
        sampled = _sample(sky, x, y, frame.camera)
        valid &= torch.isfinite(sampled).all(-1) & (sampled.mean(-1) < 0.5)
    if visibility_depth is not None:
        sampled = _sample(visibility_depth, x, y, frame.camera).reshape(-1)
        tolerance = config.visibility_absolute_tolerance + config.visibility_relative_tolerance * sampled.abs()
        valid &= torch.isfinite(sampled) & (sampled > 0) & (depth <= sampled + tolerance)
    return x, y, valid


@torch.no_grad()
def select_optical_candidates(
    xyz: Tensor,
    frames: Iterable[Any],
    config: OpticalSelectionConfig = OpticalSelectionConfig(),
    *,
    visibility_depths: Mapping[str, Tensor] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Propose low-albedo points supported by a few independent cameras.

    A bright basecolor is deliberately not required: low diffuse albedo also
    occurs on mirrors. This function never confirms optical membership.
    """
    valid_views = torch.zeros(len(xyz), dtype=torch.int32, device=xyz.device)
    low_views = torch.zeros_like(valid_views)
    albedo_sum = torch.zeros(len(xyz), device=xyz.device, dtype=xyz.dtype)
    seen: set[str] = set()
    for frame in frames:
        albedo = getattr(frame, "diffuse_albedo", None)
        if albedo is None:
            continue
        name = _view_name(frame)
        if name in seen:
            continue
        seen.add(name)
        visibility_depth = None if visibility_depths is None else visibility_depths.get(name)
        x, y, valid = _visible(xyz, frame, config, visibility_depth)
        value = _sample(albedo, x, y, frame.camera).mean(-1)
        valid &= torch.isfinite(value) & (value >= 0) & (value <= 1)
        low = valid & (value < config.albedo_threshold)
        valid_views += valid.to(valid_views.dtype)
        low_views += low.to(low_views.dtype)
        albedo_sum += torch.where(valid, value, 0)
    fraction = low_views / valid_views.clamp_min(1)
    candidate = (low_views >= config.min_views) & (fraction >= config.support_fraction)
    return candidate, {
        "valid_views": valid_views,
        "low_albedo_views": low_views,
        "support_fraction": fraction,
        "mean_albedo": albedo_sum / valid_views.clamp_min(1),
    }


class OpticalEvidence(nn.Module):
    """Accumulate a small RGB hypothesis check during candidate probation.

    Pass the opaque and optical hypotheses at the same pixels.  The optical
    image should include the same foreground/background compositing as the
    ordinary hypothesis. Only pixels with meaningful optical coverage count.
    Calls reuse training renders and perform no rendering or backward pass.
    A repeated camera cannot manufacture multi-view confidence. Call reset()
    to start a fresh validation window after additional model training.
    """

    def __init__(
        self, count: int, config: OpticalSelectionConfig = OpticalSelectionConfig(),
        *, device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if count < 0:
            raise ValueError("count must be non-negative")
        self.config = config
        self.register_buffer("valid_views", torch.zeros(count, dtype=torch.int32, device=device))
        self.register_buffer("supporting_views", torch.zeros(count, dtype=torch.int32, device=device))
        self.register_buffer("improvement_sum", torch.zeros(count, device=device))
        self.seen_cameras: set[str] = set()

    def get_extra_state(self) -> dict[str, Any]:
        return {"seen_cameras": sorted(self.seen_cameras)}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.seen_cameras = set(state["seen_cameras"])

    @torch.no_grad()
    def reset(self) -> None:
        self.valid_views.zero_()
        self.supporting_views.zero_()
        self.improvement_sum.zero_()
        self.seen_cameras.clear()

    @torch.no_grad()
    def update(
        self, xyz: Tensor, frame: Any, *, ordinary_rgb: Tensor,
        optical_rgb: Tensor, optical_gate: Tensor | None = None,
        visibility_depth: Tensor | None = None,
    ) -> bool:
        if len(xyz) != len(self.valid_views):
            raise ValueError("Point count changed during optical probation; reset evidence after topology changes")
        name = _view_name(frame)
        if name in self.seen_cameras:
            return False
        if xyz.device != self.valid_views.device:
            raise ValueError("Evidence and points must be on the same device")
        self.seen_cameras.add(name)
        x, y, valid = _visible(xyz, frame, self.config, visibility_depth)
        target = _sample(frame.rgb, x, y, frame.camera)
        ordinary = _sample(ordinary_rgb, x, y, frame.camera)
        optical = _sample(optical_rgb, x, y, frame.camera)
        if target.shape[-1] != 3 or ordinary.shape != target.shape or optical.shape != target.shape:
            raise ValueError("RGB hypotheses and target must have three channels")
        improvement = (ordinary - target).abs().mean(-1) - (optical - target).abs().mean(-1)
        valid &= torch.isfinite(improvement)
        if optical_gate is not None:
            gate = _sample(optical_gate, x, y, frame.camera).reshape(-1)
            valid &= torch.isfinite(gate) & (gate >= self.config.min_optical_gate)
        self.valid_views += valid.to(self.valid_views.dtype)
        self.supporting_views += (valid & (improvement >= self.config.min_improvement)).to(self.supporting_views.dtype)
        self.improvement_sum += torch.where(valid, improvement, 0)
        return True

    def decision(self) -> tuple[Tensor, Tensor]:
        """Return (accepted, decided); under-observed candidates stay pending."""
        decided = self.valid_views >= self.config.min_views
        fraction = self.supporting_views / self.valid_views.clamp_min(1)
        mean = self.improvement_sum / self.valid_views.clamp_min(1)
        accepted = decided & (fraction >= self.config.support_fraction) & (mean >= self.config.min_improvement)
        return accepted, decided
