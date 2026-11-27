from types import SimpleNamespace

import pytest
import torch

from glint.optical_selection import OpticalEvidence, OpticalSelectionConfig, select_optical_candidates


def frame(name="a", albedo=0.01):
    camera = SimpleNamespace(
        K=torch.tensor([[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]]),
        R=torch.eye(3), T=torch.zeros(3), image_height=4, image_width=4,
        name=name, znear=1e-4, zfar=100.,
    )
    return SimpleNamespace(
        camera=camera, camera_name=name, diffuse_albedo=torch.full((4, 4, 3), albedo),
        rgb=torch.ones(4, 4, 3), sky_mask=None, mask=None,
    )


def test_candidate_needs_independent_views_and_low_albedo():
    xyz = torch.tensor([[0., 0., 2.], [0., 0., -2.], [100., 0., 2.]])
    selected, data = select_optical_candidates(xyz, [frame(), frame()])
    assert not selected.any()
    assert data["valid_views"].tolist() == [1, 0, 0]
    selected, _ = select_optical_candidates(xyz, [frame("a"), frame("b")])
    assert selected.tolist() == [True, False, False]
    selected, _ = select_optical_candidates(xyz, [frame("a"), frame("b", .5)])
    assert not selected.any()


def test_sky_missing_invalid_and_occluded_priors_are_excluded():
    xyz = torch.tensor([[0., 0., 2.]])
    a, b = frame("a"), frame("b")
    a.sky_mask = torch.ones(4, 4, 1)
    selected, _ = select_optical_candidates(xyz, [a, b])
    assert not selected.any()
    a.sky_mask = None
    selected, _ = select_optical_candidates(xyz, [a, b], visibility_depths={"a": torch.ones(4, 4, 1)})
    assert not selected.any()
    a.diffuse_albedo = None
    b.diffuse_albedo[:] = float("nan")
    selected, data = select_optical_candidates(xyz, [a, b])
    assert not selected.any()
    assert data["valid_views"].item() == 0


def test_validation_requires_improvement_and_independent_cameras():
    xyz = torch.tensor([[0., 0., 2.]])
    evidence = OpticalEvidence(1)
    ordinary, optical = torch.zeros(4, 4, 3), torch.ones(4, 4, 3)
    for _ in range(3):
        evidence.update(xyz, frame("a"), ordinary_rgb=ordinary, optical_rgb=optical)
    accepted, decided = evidence.decision()
    assert not accepted.any() and not decided.any()
    evidence.update(xyz, frame("b"), ordinary_rgb=ordinary, optical_rgb=optical)
    accepted, decided = evidence.decision()
    assert accepted.all() and decided.all()


def test_dark_opaque_equal_or_worse_hypothesis_is_not_confirmed():
    xyz = torch.tensor([[0., 0., 2.]])
    evidence = OpticalEvidence(1)
    for name in ("a", "b"):
        f = frame(name)
        f.rgb.zero_()
        evidence.update(xyz, f, ordinary_rgb=torch.zeros(4, 4, 3), optical_rgb=torch.zeros(4, 4, 3))
    accepted, decided = evidence.decision()
    assert not accepted.any() and decided.all()


def test_zero_coverage_is_pending_and_checkpoint_preserves_seen_views():
    xyz = torch.tensor([[0., 0., 2.]])
    evidence = OpticalEvidence(1)
    evidence.update(xyz, frame(), ordinary_rgb=torch.zeros(4, 4, 3), optical_rgb=torch.ones(4, 4, 3), optical_gate=torch.zeros(4, 4, 1))
    copy = OpticalEvidence(1)
    copy.load_state_dict(evidence.state_dict())
    assert copy.seen_cameras == {"a"}
    assert not copy.decision()[1].any()
    copy.reset()
    assert not copy.seen_cameras


def test_empty_sets_and_bad_configuration():
    selected, _ = select_optical_candidates(torch.empty(0, 3), [frame("a"), frame("b")])
    assert selected.shape == (0,)
    evidence = OpticalEvidence(0)
    assert evidence.decision()[0].shape == (0,)
    with pytest.raises(ValueError):
        OpticalSelectionConfig(min_views=1)
    with pytest.raises(ValueError):
        OpticalSelectionConfig(min_improvement=0)
