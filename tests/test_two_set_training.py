"""Two-set ownership, optimizer continuity, and a CPU trainer lifecycle.

The lifecycle substitutes rendering/refinement only; selection, promotion,
material optimization, confirmation, checkpoint and resume run real code.
"""
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from glint.checkpoint import GlintGaussianSet
from glint.two_set_model import TwoSetModel, load_two_set, make_optimizers, transfer_optimizer_state
from glint.optical_selection import OpticalSelectionConfig
from glint.train_two_set import TwoSetTrainer


def gaussian(count=5, role="reflection"):
    state = {
        "set._xyz": torch.stack((torch.arange(count) + 1.5, torch.full((count,), 1.5), torch.ones(count)), -1),
        "set._features_dc": torch.zeros(count, 1, 3),
        "set._features_rest": torch.zeros(count, 3, 3),
        "set._scaling": torch.full((count, 2), -3.),
        "set._rotation": torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
        "set._opacity": torch.zeros(count, 1),
        "set.active_sh_degree": torch.tensor([0]),
    }
    if role == "interface":
        state["set._specular"] = torch.zeros(count, 1)
        state["set._transmission_coeff"] = torch.zeros(count, 1)
    return GlintGaussianSet(state, "set", role=role, trainable=True)


def model():
    return TwoSetModel(gaussian(), gaussian(0, "interface"))


def seed_adam(optimizers):
    for opts in optimizers.values():
        for optimizer in opts.values():
            p = optimizer.param_groups[0]["params"][0]
            p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape_as(p) + 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def state_for(optimizers, role, key):
    opt = optimizers[role][key]
    return opt.state[opt.param_groups[0]["params"][0]]


def test_promotion_rejection_preserves_disjoint_geometry_and_adam_history():
    m = model()
    old = make_optimizers(m)
    seed_adam(old)
    before = m.general.get_xyz.detach().clone()
    history = {key: {name: value.clone() for name, value in state_for(old, "general", key).items()}
               for key in old["general"]}
    mask = torch.tensor([True, False, True, False, False])
    m.promote(mask)
    promoted = make_optimizers(m)
    transfer_optimizer_state(old, promoted, mask, promotion=True)
    assert set(dict(m.named_children())) == {"general", "optical"}
    torch.testing.assert_close(m.general.get_xyz, before[~mask])
    torch.testing.assert_close(m.optical.get_xyz, before[mask])
    for key, saved in history.items():
        for role, index in (("general", ~mask), ("optical", mask)):
            got = state_for(promoted, role, key)
            for moment in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(got[moment], saved[moment][index])
            torch.testing.assert_close(got["step"], saved["step"])
    assert not promoted["optical"]["specular"].state
    assert not promoted["optical"]["transparency"].state
    accepted = torch.tensor([True, False])
    m.reject(accepted)
    merged = make_optimizers(m)
    transfer_optimizer_state(promoted, merged, accepted, promotion=False)
    expected_general = torch.tensor([1, 3, 4, 2])
    torch.testing.assert_close(m.general.get_xyz, before[expected_general])
    torch.testing.assert_close(m.optical.get_xyz, before[[0]])
    for key, saved in history.items():
        torch.testing.assert_close(state_for(merged, "general", key)["exp_avg"], saved["exp_avg"][expected_general])
    # A subsequent optimizer update must use the new parameters and state sizes.
    seed_adam(merged)


def test_model_roundtrip_empty_and_nonempty_optical(tmp_path):
    for nonempty in (False, True):
        m = model()
        if nonempty:
            m.promote(torch.tensor([True, False, False, False, False]))
        path = tmp_path / f"model-{nonempty}.pt"
        torch.save({"format": "glint-two-set-v1", "model": m.state_dict()}, path)
        restored, _ = load_two_set(path)
        assert restored.transport_mode == "fresnel"
        assert set(restored.state_dict()) == set(m.state_dict())
        for key, value in m.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
    legacy = tmp_path / "legacy.pt"
    torch.save({"model": {}}, legacy)
    with pytest.raises(ValueError, match="legacy"):
        load_two_set(legacy)


def test_promotion_rejects_all_points_and_wrong_mask():
    m = model()
    with pytest.raises(ValueError, match="general"):
        m.promote(torch.ones(5, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        m.promote(torch.ones(5))






class Frame(SimpleNamespace):
    def to(self, device):
        assert torch.device(device).type == "cpu"
        return self


def frames():
    result = []
    for name in ("a", "b"):
        camera = SimpleNamespace(K=torch.eye(3), R=torch.eye(3), T=torch.zeros(3),
                                 image_height=4, image_width=8, name=name)
        albedo = torch.ones(4, 8, 3)
        albedo[:, 1:3] = 0.01
        result.append(Frame(camera=camera, camera_name=name, diffuse_albedo=albedo,
                            rgb=torch.ones(4, 8, 3), normal=None, sky_mask=None, mask=None))
    return result


@pytest.fixture
def fake_backends(monkeypatch):
    @dataclass
    class Config:
        stop: int
        max_gaussians: int
        min_gaussians: int
        max_scale_ratio: float

    class Refiner:
        def __init__(self, gaussian, optimizers, config, *, scene_scale):
            self.steps = 0
        def retain_interface_gradient(self, output): pass
        def accumulate_interface(self, output): pass
        def accumulate_trace(self, trace): pass
        def stabilize_covariance(self): pass
        def step(self, step): self.steps += 1
        def state_dict(self): return {"steps": self.steps}
        def load_state_dict(self, state): self.steps = state["steps"]

    refinement = ModuleType("glint.refinement")
    refinement.GlintRefiner, refinement.RefinementConfig = Refiner, Config
    monkeypatch.setitem(sys.modules, "glint.refinement", refinement)
    optical_refinement = ModuleType("glint.two_set_refinement")
    optical_refinement.OpticalRefiner = Refiner
    monkeypatch.setitem(sys.modules, "glint.two_set_refinement", optical_refinement)
    dataset = ModuleType("glint.dataset")
    dataset.make_glint_dataloader = lambda data, **kwargs: [SimpleNamespace(samples=[f]) for f in data]
    monkeypatch.setitem(sys.modules, "glint.dataset", dataset)

    def render(camera, m, tracer, **kwargs):
        h, w = camera.image_height, camera.image_width
        # A material-dependent, differentiable optical explanation succeeds at
        # x=2 but fails at x=1; candidate confirmation must split the two.
        signal = m.general._features_dc.mean() * .01
        if len(m.optical.get_xyz):
            signal = signal + m.optical.get_transmission_coeff.mean() * .8
        rgb = torch.ones(h, w, 3) * signal
        good = torch.ones(h, w, 1)
        good[:, 1] = 0
        rgb = rgb * good
        raster = SimpleNamespace(surf_depth=torch.ones(1, h, w))
        return SimpleNamespace(
            render=rgb.permute(2, 0, 1), opaque_rgb=torch.zeros_like(rgb),
            optical_gate=torch.ones(h, w, 1) if len(m.optical.get_xyz) else torch.zeros(h, w, 1),
            general=raster, optical=raster if len(m.optical.get_xyz) else None,
            reflection_trace=None, transmission_trace=None,
        )
    monkeypatch.setattr("glint.train_two_set.render_two_set", render)


def args(tmp_path, **overrides):
    values = dict(scene_scale=1., no_refinement=True, refine_stop=100, max_gaussians=100,
                  selection_views=2, ray_epsilon=1e-4, normal_weight=0.,
                  max_steps=4, warmup_steps=1, probation_steps=1, num_workers=0,
                  log_every=1, output_dir=tmp_path, save_every=1)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", ["fresnel", "direct_beta"])
def test_trainer_real_selection_confirmation_checkpoint_and_resume(fake_backends, tmp_path, mode):
    trainer = TwoSetTrainer(model(), frames(), None, args(tmp_path, transport_mode=mode))
    trainer.train()
    assert trainer.step == 4 and trainer.phase == "confirmed"
    assert trainer.validation_summary["candidates"] == 2
    assert trainer.validation_summary["accepted"] == 1
    assert trainer.validation_summary["returned_to_general"] == 1
    assert len(trainer.model.general.get_xyz) == 4
    assert len(trainer.model.optical.get_xyz) == 1
    path = tmp_path / "checkpoints/latest.pt"
    restored_model, payload = load_two_set(path)
    assert restored_model.transport_mode == payload["transport_mode"] == mode
    resumed = TwoSetTrainer(restored_model, frames(), None, args(tmp_path, max_steps=5, warmup_steps=99))
    resumed.restore(payload)
    assert resumed.model.transport_mode == resumed.args.transport_mode == mode
    assert resumed.args.warmup_steps == 1
    assert resumed.step == 4 and resumed.phase == "confirmed"
    resumed.train()
    assert resumed.step == 5 and resumed.phase == "confirmed"
    assert resumed.validation_summary == trainer.validation_summary


def test_provisional_checkpoint_resumes_probation_then_confirms(fake_backends, tmp_path):
    trainer = TwoSetTrainer(model(), frames(), None, args(tmp_path, max_steps=2, probation_steps=2))
    trainer.train()
    assert trainer.phase == "probation" and len(trainer.model.optical.get_xyz) == 2
    restored_model, payload = load_two_set(tmp_path / "checkpoints/latest.pt")
    resumed = TwoSetTrainer(restored_model, frames(), None, args(tmp_path, max_steps=4))
    resumed.restore(payload)
    assert resumed.args.probation_steps == 2
    resumed.train()
    assert resumed.phase == "confirmed" and resumed.validation_summary["accepted"] == 1


def test_selection_budget_counts_cameras_not_video_frames(fake_backends, tmp_path):
    class Dataset(list):
        pass
    a, b = frames()
    data = Dataset([a, a, a, b, b])
    data.entries = [("a", "0"), ("a", "1"), ("a", "2"), ("b", "0"), ("b", "1")]
    trainer = TwoSetTrainer(model(), data, None, args(tmp_path))
    assert [f.camera_name for f in trainer._selection_frames()] == ["a", "b"]


def test_invalid_normal_prior_does_not_poison_loss(fake_backends, tmp_path):
    trainer = TwoSetTrainer(model(), frames(), None, args(tmp_path, normal_weight=.1))
    frame = frames()[0]
    frame.normal = torch.full_like(frame.rgb, float("nan"))
    raster = SimpleNamespace(rend_normal=torch.ones(3, 4, 8))
    output = SimpleNamespace(render=frame.rgb.permute(2, 0, 1),
                             general=raster, optical=None,
                             optical_gate=torch.zeros(4, 8, 1))
    loss, _ = trainer.loss(output, frame)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, torch.tensor(0.), atol=1e-5, rtol=0)
