"""CPU composition/gradient tests; CUDA kernels are independently provided by GLINT."""

from types import SimpleNamespace

import pytest
import torch

import glint.two_set_transport as transport


def _fixture(monkeypatch, *, optical_alpha=0.8, optical_depth=2.0,
             general_alpha=0.8, general_depth=4.0, width=2):
    camera = SimpleNamespace(
        K=torch.tensor([[100.0, 0.0, width / 2], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]),
        R=torch.eye(3), T=torch.zeros(3, 1), image_width=width, image_height=1,
    )
    general_color = torch.tensor(0.6, requires_grad=True)
    optical_color = torch.tensor(0.1, requires_grad=True)
    specularity = torch.tensor(0.2, requires_grad=True)
    transparency = torch.tensor(0.6, requires_grad=True)
    model = SimpleNamespace(
        general=SimpleNamespace(get_xyz=torch.zeros(4, 3)),
        optical=SimpleNamespace(get_xyz=torch.zeros(2, 3)), background=torch.zeros(3),
    )

    def rasterize(camera, gaussians, background):
        is_optical = gaussians is model.optical
        alpha = optical_alpha if is_optical else general_alpha
        depth = optical_depth if is_optical else general_depth
        color = optical_color if is_optical else general_color
        output = transport.RenderOutput(
            render=color.expand(3, 1, width) * (alpha if is_optical else 1.0),
            rend_alpha=torch.full((1, 1, width), alpha),
            surf_depth=torch.full((1, 1, width), depth),
            rend_normal=torch.tensor([0., 0., -1.]).reshape(3, 1, 1).expand(3, 1, width),
        )
        if is_optical:
            output.specular = specularity.expand(1, 1, width) * alpha
            output.conditional_transparency = transparency.expand(1, 1, width)
        return output

    monkeypatch.setattr(transport, "_rasterize", rasterize)
    calls = []

    def trace(gaussians, rays, *, background):
        calls.append((gaussians, rays))
        # Return distinct colors for the two paths and retain a gradient to
        # the shared general radiance parameter.
        rgb = general_color * (0.5 if len(calls) % 2 else 1.0)
        shape = rays.origins.shape
        return SimpleNamespace(
            rgb=rgb.expand(shape), depth=torch.ones(*shape[:2], 1),
            alpha=torch.ones(*shape[:2], 1), normal=torch.zeros(shape),
            specular=None, transparency=None, weight_accumulate=None,
            backend_data=None,
        )

    return camera, model, SimpleNamespace(trace=trace), calls, (
        general_color, optical_color, specularity, transparency,
    )


@pytest.mark.parametrize("mode", ["fresnel", "direct_beta"])
def test_shared_general_radiance_and_material_gradients(monkeypatch, mode):
    camera, model, tracer, calls, parameters = _fixture(monkeypatch)
    model.transport_mode = mode
    result = transport.render_two_set(camera, model, tracer)
    assert len(calls) == 2
    assert all(target is model.general for target, _ in calls)
    torch.testing.assert_close(result.energy_sum, torch.ones_like(result.energy_sum))
    assert calls[0][1].directions[..., 2].min() > 0
    assert calls[1][1].directions[..., 2].max() < 0
    assert result.transmission_trace.rgb.shape == (1, 2, 3)
    # Surface color must be de-premultiplied before using the optical gate.
    torch.testing.assert_close(result.optical_surface_rgb, torch.full((1, 2, 3), 0.1))
    torch.testing.assert_close(result.opaque_rgb, torch.full((1, 2, 3), 0.2))
    result.render.sum().backward()
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("beta,t", [(0., 0.), (0., 1.), (1., .6), (.2, .6)])
def test_direct_beta_weights_and_fresnel_independence(monkeypatch, beta, t):
    camera, model, tracer, _, parameters = _fixture(monkeypatch)
    model.transport_mode = "direct_beta"
    with torch.no_grad():
        parameters[2].fill_(beta)
        parameters[3].fill_(t)
    result = transport.render_two_set(camera, model, tracer, f0=0.)
    other = transport.render_two_set(camera, model, tracer, f0=1.)
    for weight, expected in ((result.reflection_weight, beta),
                             (result.transmission_weight, (1-beta)*t),
                             (result.diffuse_weight, (1-beta)*(1-t))):
        torch.testing.assert_close(weight, torch.full_like(weight, expected))
    torch.testing.assert_close(result.energy_sum, torch.ones_like(result.energy_sum))
    torch.testing.assert_close(result.render, other.render)


@pytest.mark.parametrize("options", [
    {"optical_alpha": 0.0},
    {"optical_depth": 0.0},
    {"general_alpha": 1.0, "general_depth": 1.0},
])
def test_absent_or_occluded_optical_surface_leaves_general_unchanged(monkeypatch, options):
    camera, model, tracer, calls, _ = _fixture(monkeypatch, **options)
    result = transport.render_two_set(camera, model, tracer)
    assert not calls
    torch.testing.assert_close(result.render.permute(1, 2, 0), result.direct_rgb)
    assert result.optical_gate.count_nonzero() == 0


def test_opaque_mode_keeps_surface_without_tracing(monkeypatch):
    camera, model, tracer, calls, _ = _fixture(monkeypatch)
    result = transport.render_two_set(camera, model, tracer, optical_enabled=False)
    assert not calls
    torch.testing.assert_close(result.render.permute(1, 2, 0), result.opaque_rgb)
    assert not torch.equal(result.opaque_rgb, result.direct_rgb)


def test_no_optical_points(monkeypatch):
    camera, model, tracer, calls, _ = _fixture(monkeypatch)
    model.optical.get_xyz = torch.empty(0, 3)
    result = transport.render_two_set(camera, model, tracer)
    assert not calls
    assert result.optical is None
    torch.testing.assert_close(result.render.permute(1, 2, 0), result.direct_rgb)


def test_empty_general_uses_background_for_both_paths(monkeypatch):
    camera, model, tracer, calls, _ = _fixture(monkeypatch)
    model.general.get_xyz = torch.empty(0, 3)
    model.background[:] = 0.7
    result = transport.render_two_set(camera, model, tracer)
    assert not calls
    assert result.general is None
    expected = (1 - result.optical_gate) * 0.7 + result.optical_gate * (
        result.diffuse_weight * 0.1
        + (result.reflection_weight + result.transmission_weight) * 0.7
    )
    torch.testing.assert_close(result.render.permute(1, 2, 0), expected.expand(1, 2, 3))


def test_sparse_rays_skip_uncovered_pixels(monkeypatch):
    camera, model, tracer, calls, _ = _fixture(monkeypatch, width=3)
    original = transport._rasterize

    def rasterize(camera, gaussians, background):
        result = original(camera, gaussians, background)
        if gaussians is model.optical:
            result.rend_alpha[..., 1] = 0
        return result

    monkeypatch.setattr(transport, "_rasterize", rasterize)
    result = transport.render_two_set(camera, model, tracer)
    assert all(rays.origins.shape == (1, 2, 3) for _, rays in calls)
    torch.testing.assert_close(result.render[:, :, 1], result.general.render[:, :, 1])
    assert result.transmission_trace.active_pixels.tolist() == [0, 2]


def test_missing_tracer_errors_only_for_visible_optical(monkeypatch):
    camera, model, _, _, _ = _fixture(monkeypatch)
    with pytest.raises(ValueError, match="ray tracer"):
        transport.render_two_set(camera, model, None)


def test_premultiplied_color_opacity_gradient_is_counted_once(monkeypatch):
    camera, model, tracer, _, _ = _fixture(monkeypatch)
    alpha = torch.tensor(0.8, requires_grad=True)
    original = transport._rasterize

    def rasterize(camera, gaussians, background):
        result = original(camera, gaussians, background)
        if gaussians is model.optical:
            result.render = torch.full((3, 1, 2), 0.1) * alpha
            result.rend_alpha = alpha.expand(1, 1, 2)
        return result

    monkeypatch.setattr(transport, "_rasterize", rasterize)
    output = transport.render_two_set(camera, model, tracer, optical_enabled=False)
    output.render.sum().backward()
    torch.testing.assert_close(alpha.grad, torch.tensor(6 * (0.1 - 0.6)))


def test_partial_foreground_contribution_is_not_attenuated_twice(monkeypatch):
    camera, model, tracer, _, _ = _fixture(
        monkeypatch, general_alpha=0.5, general_depth=1.0, optical_alpha=1.0,
    )
    model.background[:] = 0.2
    original = transport._rasterize

    def rasterize(camera, gaussians, background):
        result = original(camera, gaussians, background)
        if gaussians is model.general:
            # Half opaque ordinary foreground, with background in the remainder.
            result.render = torch.full((3, 1, 2), 0.5 * 0.6 + 0.5 * 0.2)
        return result

    monkeypatch.setattr(transport, "_rasterize", rasterize)
    output = transport.render_two_set(camera, model, tracer, optical_enabled=False)
    expected = 0.5 * 0.6 + 0.5 * 0.1
    torch.testing.assert_close(output.render, torch.full_like(output.render, expected))


@pytest.mark.parametrize("kwargs", [{"f0": -0.1}, {"f0": 1.1}, {"ray_epsilon": -1.0}])
def test_reject_invalid_transport_settings(monkeypatch, kwargs):
    camera, model, tracer, _, _ = _fixture(monkeypatch)
    with pytest.raises(ValueError):
        transport.render_two_set(camera, model, tracer, **kwargs)
