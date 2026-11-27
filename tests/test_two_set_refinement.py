import pytest
import torch

pytest.importorskip("gsplat", reason="Actual GLINT refiner requires gsplat strategy helpers")

from glint.model import create_gaussian_set
from glint.refinement import GlintRefiner, RefinementConfig
from glint.two_set_refinement import OpticalRefiner


@pytest.mark.parametrize("refiner_type,keep_material", [(GlintRefiner, False), (OpticalRefiner, True)])
def test_opacity_reset_preserves_only_two_set_optical_material(refiner_type, keep_material):
    torch.manual_seed(7)
    optical = create_gaussian_set(torch.randn(8, 3), torch.rand(8, 3),
                                  role="interface", device="cpu", init_transparency=.8,
                                  init_opacity=.5)
    optimizers = {name: torch.optim.Adam([p], lr=.001)
                  for name, p in optical.parameter_map().items()}
    for name in ("opacities", "transparency"):
        parameter = optical.parameter_map()[name]
        parameter.grad = torch.ones_like(parameter)
        optimizers[name].step()
        optimizers[name].zero_grad(set_to_none=True)
    material = optical._transmission_coeff
    material_value = material.detach().clone()
    material_history = {k: v.clone() for k, v in optimizers["transparency"].state[material].items()}
    old_opacity = optical._opacity
    # Reset at step 3 without a topology update, so the test isolates the real
    # upstream scheduled reset and exercises actual Adam replacement behavior.
    config = RefinementConfig(start=0, stop=10, every=100, reset_every=3)
    refiner_type(optical, optimizers, config, scene_scale=1.).step(3)
    assert optical._opacity is not old_opacity
    assert optical.get_opacity.max() <= .010001
    opacity_state = optimizers["opacities"].state[optical._opacity]
    assert torch.count_nonzero(opacity_state["exp_avg"]) == 0
    assert torch.count_nonzero(opacity_state["exp_avg_sq"]) == 0
    if keep_material:
        assert optical._transmission_coeff is material
        torch.testing.assert_close(optical._transmission_coeff, material_value)
        for key, value in material_history.items():
            torch.testing.assert_close(optimizers["transparency"].state[material][key], value)
    else:
        assert optical._transmission_coeff is not material
        assert optical.get_transmission_coeff.max() <= .010001
