"""Native CUDA/OptiX smoke test, skipped when the optional backend is absent."""
import importlib.util

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("gsplat") is None
    or importlib.util.find_spec("diff_surfel_tracing") is None,
    reason="requires CUDA, GLINT gsplat and the compiled OptiX extension",
)


def _gaussians(xyz, colors, role):
    from glint.checkpoint import GlintGaussianSet

    xyz = torch.tensor(xyz, device="cuda", dtype=torch.float32)
    colors = torch.tensor(colors, device="cuda", dtype=torch.float32)
    n = len(xyz)
    state = {
        "g._xyz": xyz,
        "g._features_dc": ((colors - 0.5) / 0.28209479177387814).reshape(n, 1, 3),
        "g._features_rest": xyz.new_zeros(n, 0, 3),
        "g._scaling": xyz.new_full((n, 2), 0.0),
        "g._rotation": xyz.new_tensor([[1., 0., 0., 0.]]).expand(n, 4).clone(),
        "g._opacity": xyz.new_full((n, 1), 0.8).logit(),
        "g._specular": xyz.new_full((n, 1), 0.15).logit(),
        "g._transmission_coeff": xyz.new_full((n, 1), 0.6).logit(),
        "g.active_sh_degree": torch.zeros(1, device="cuda", dtype=torch.long),
    }
    return GlintGaussianSet(state, "g", role=role, trainable=True)


@pytest.mark.parametrize("mode", ["fresnel", "direct_beta"])
def test_native_two_set_shared_bvh_forward_backward_multiple_steps(mode):
    from glint.camera import GlintCamera
    from glint.optix_backend import OptixSurfelTracer
    from glint.two_set_model import TwoSetModel
    from glint.two_set_transport import render_two_set

    camera = GlintCamera(
        K=torch.tensor([[16., 0., 8.], [0., 16., 8.], [0., 0., 1.]], device="cuda"),
        R=torch.eye(3, device="cuda"), T=torch.zeros(3, 1, device="cuda"),
        image_width=16, image_height=16,
    )
    general = _gaussians(
        [[0., 0., 4.], [0., 0., -2.]],
        [[0.2, 0.7, 0.3], [0.8, 0.2, 0.7]], "reflection",
    )
    optical = _gaussians([[0., 0., 2.]], [[0.1, 0.1, 0.1]], "interface")
    model = TwoSetModel(general, optical, transport_mode=mode)
    tracer = OptixSurfelTracer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        output = render_two_set(camera, model, tracer)
        assert output.render.shape == (3, 16, 16)
        assert torch.isfinite(output.render).all()
        torch.testing.assert_close(output.energy_sum, torch.ones_like(output.energy_sum))
        assert output.transmission_trace is not None and output.reflection_trace is not None
        # Both transport calls share the same ordinary-scene acceleration structure.
        assert len(tracer._states) == 1 and id(model.general) in tracer._states
        loss = (output.render - 0.4).square().mean()
        loss.backward()
        for parameter in (general._features_dc, optical._specular, optical._transmission_coeff):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        losses.append(float(loss.detach()))
        optimizer.step()
    assert all(value >= 0 for value in losses)
