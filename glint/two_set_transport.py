"""Two-set, single-interaction radiance transport using GLINT's CUDA backends.

The general set supplies radiance for *both* secondary paths.  Optical surfels
carry the interaction geometry, diffuse color, specularity and transparency.
This retains GLINT's optically-thin (straight transmission) approximation.
Two rasterized depth layers approximate foreground occlusion; overlapping,
interleaved semitransparent layers and repeated optical bounces are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


class RenderOutput(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


@dataclass
class _Rays:
    origins: Tensor
    directions: Tensor


def _rasterize(camera: Any, gaussians: Any, background: Tensor) -> Any:
    # Keep the material/composition tests independent of optional CUDA builds.
    from glint.renderer import AttrDict, render_glint_camera

    return render_glint_camera(
        camera, gaussians, AttrDict(compute_cov3D_python=False, depth_ratio=0.0),
        background, device=str(camera.K.device),
    )


def _camera_rays(camera: Any) -> _Rays:
    y, x = torch.meshgrid(
        torch.arange(camera.image_height, device=camera.K.device, dtype=camera.K.dtype) + 0.5,
        torch.arange(camera.image_width, device=camera.K.device, dtype=camera.K.dtype) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    camera_directions = torch.einsum("ij,hwj->hwi", torch.linalg.inv(camera.K), pixels)
    directions = torch.einsum("ij,hwj->hwi", camera.R.T, camera_directions)
    center = (-camera.R.T @ camera.T.reshape(3, 1)).reshape(1, 1, 3)
    return _Rays(center.expand_as(directions), directions)


def _hwc(value: Tensor) -> Tensor:
    return value.permute(1, 2, 0)


def _trace_active(
    tracer: Any, general: Any, origins: Tensor, directions: Tensor,
    active: Tensor, background: Tensor, epsilon: float,
) -> Any:
    """Trace only optical pixels, retaining native per-Gaussian gradient data."""
    selected = torch.where(active.reshape(-1))[0]
    sparse_directions = directions.reshape(-1, 3)[selected].unsqueeze(0)
    sparse_origins = origins.reshape(-1, 3)[selected].unsqueeze(0)
    sparse_origins = sparse_origins + epsilon * F.normalize(sparse_directions, dim=-1)
    native = tracer.trace(
        general, _Rays(sparse_origins, sparse_directions), background=background,
    )
    result = RenderOutput(vars(native)) if not isinstance(native, dict) else RenderOutput(native)
    for name in ("rgb", "depth", "alpha", "normal", "specular", "transparency"):
        value = getattr(native, name, None)
        if value is not None:
            channels = value.shape[-1]
            dense = value.new_zeros((active.numel(), channels))
            result[name] = dense.index_copy(0, selected, value.reshape(-1, channels)).reshape(
                *active.shape[:2], channels,
            )
    result.active_pixels = selected
    return result


def render_two_set(
    camera: Any, model: Any, tracer: Any, *, optical_enabled: bool = True,
    f0: float = 0.04, ray_epsilon: float = 1e-4,
) -> RenderOutput:
    """Rasterize two sets and trace reflection/transmission into one shared set.

    ``optical_enabled=False`` renders optical surfels as ordinary opaque diffuse
    surfaces; it does not remove them. ``opaque_rgb`` is that competing surface
    hypothesis and is appropriate for lightweight candidate validation.
    Image fields use HWC, except ``render`` and native raster outputs (CHW).
    """
    if not 0.0 <= f0 <= 1.0 or ray_epsilon < 0.0:
        raise ValueError("f0 must be in [0, 1] and ray_epsilon non-negative")
    mode = getattr(model, "transport_mode", "fresnel")
    if mode not in {"fresnel", "direct_beta"}:
        raise ValueError(f"Unknown transport mode: {mode}")
    height, width = camera.image_height, camera.image_width
    background = model.background.to(device=camera.K.device, dtype=camera.K.dtype)
    direct = background.reshape(1, 1, 3).expand(height, width, 3)
    zero = direct.new_zeros((height, width, 1))
    general = None
    if len(model.general.get_xyz):
        general = _rasterize(camera, model.general, background)
        direct = _hwc(general.render)
    output = RenderOutput(
        general=general, optical=None, direct_rgb=direct, opaque_rgb=direct,
        optical_surface_rgb=torch.zeros_like(direct), optical_rgb=torch.zeros_like(direct),
        optical_gate=zero, transmission_trace=None, reflection_trace=None,
        diffuse_weight=torch.ones_like(zero), reflection_weight=zero,
        transmission_weight=zero, energy_sum=torch.ones_like(zero),
        render=direct.permute(2, 0, 1),
    )
    if not len(model.optical.get_xyz):
        return output

    optical = _rasterize(camera, model.optical, torch.zeros_like(background))
    coverage = _hwc(optical.rend_alpha).clamp(0.0, 1.0)
    depth = _hwc(optical.surf_depth)
    normal = F.normalize(_hwc(optical.rend_normal), dim=-1)
    valid = (depth > 0) & torch.isfinite(depth)
    gate = coverage * valid.to(coverage.dtype)
    replacement_rgb = direct
    if general is not None:
        general_depth = _hwc(general.surf_depth).detach()
        foreground = (general_depth > 0) & (general_depth + ray_epsilon < depth.detach())
        # The depth ordering is discrete; preserve gradients to coverage itself.
        gate = gate * torch.where(
            foreground, 1.0 - _hwc(general.rend_alpha).clamp(0.0, 1.0),
            torch.ones_like(gate),
        )
        # If ordinary geometry is in front, its premultiplied contribution
        # must remain intact. The optical layer replaces only the background
        # visible through that foreground, not the foreground RGB itself.
        replacement_rgb = torch.where(foreground, background.reshape(1, 1, 3), direct)
    # RGB and specularity are alpha-premultiplied raster features. Differentiate
    # through this normalization so applying the gate below does not count the
    # opacity derivative twice. Transparency already has GLINT's isolated
    # material gradient and conditional normalization in the rasterizer.
    denominator = coverage.clamp_min(1e-6)
    surface_rgb = _hwc(optical.render) / denominator
    specularity = (_hwc(optical.specular) / denominator).clamp(0.0, 1.0)
    transparency = _hwc(optical.conditional_transparency).clamp(0.0, 1.0)
    opaque = direct + gate * (surface_rgb - replacement_rgb)
    output.update(
        optical=optical, optical_surface_rgb=surface_rgb, optical_rgb=surface_rgb,
        optical_gate=gate, opaque_rgb=opaque, render=opaque.permute(2, 0, 1),
        material_transparency=transparency, material_specularity=specularity,
    )
    if not optical_enabled or not bool((gate.detach() > 1e-6).any()):
        return output

    rays = _camera_rays(camera)
    cosine = (F.normalize(rays.directions, dim=-1) * normal).sum(-1, keepdim=True).abs()
    fresnel = f0 + (1.0 - f0) * (1.0 - cosine.clamp(0.0, 1.0)).pow(5)
    # direct_beta uses the learned sigmoid coefficient directly, with no
    # Fresnel floor or angular multiplier. Ray directions remain geometric.
    reflection_weight = (specularity if mode == "direct_beta" else
                         specularity + (1.0 - specularity) * fresnel)
    transmission_weight = (1.0 - reflection_weight) * transparency
    diffuse_weight = (1.0 - reflection_weight) * (1.0 - transparency)
    origins = rays.origins + rays.directions * depth.detach()
    reflected = rays.directions - 2.0 * (rays.directions * normal).sum(-1, keepdim=True) * normal
    transmission_rgb = reflection_rgb = background.reshape(1, 1, 3).expand_as(direct)
    if len(model.general.get_xyz):
        if tracer is None:
            raise ValueError("Visible optical surfaces require a ray tracer")
        active = gate.detach() > 1e-6
        trans_trace = _trace_active(
            tracer, model.general, origins, rays.directions, active, background, ray_epsilon,
        )
        refl_trace = _trace_active(
            tracer, model.general, origins, reflected, active, background, ray_epsilon,
        )
        transmission_rgb, reflection_rgb = trans_trace.rgb, refl_trace.rgb
        output.update(transmission_trace=trans_trace, reflection_trace=refl_trace)
    optical_rgb = (
        diffuse_weight * surface_rgb
        + transmission_weight * transmission_rgb
        + reflection_weight * reflection_rgb
    )
    rendered = direct + gate * (optical_rgb - replacement_rgb)
    output.update(
        render=rendered.permute(2, 0, 1), optical_rgb=optical_rgb,
        diffuse_weight=diffuse_weight, reflection_weight=reflection_weight,
        transmission_weight=transmission_weight, fresnel=fresnel,
        energy_sum=diffuse_weight + reflection_weight + transmission_weight,
    )
    return output
