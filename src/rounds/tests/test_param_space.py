"""ParamSpace tests, against the real generator (numpy/scipy only, no torch) -- fast because
compositions here are tiny (a handful of markers/types), not because anything is mocked."""
import numpy as np
import pytest

from gen_dataset import panel_structure, generate_world
from parameter import DEFAULT_PIN
import config as cfg
from param_space import ParamSpace, PRIOR_VAR_UNIFORM01

ARCH_PIN = (r"\.STIFF$", r"\.ANISO$", r"^ARCH\.CONTACT$")


def _tiny_space(struct_seed=1, n_markers=4, n_types=3, with_artifacts=True):
    struct = panel_structure(n_markers, n_types, 0.6, struct_seed=struct_seed)
    ref = generate_world(struct, rng=np.random.default_rng(0), with_artifacts=with_artifacts)
    return struct, ParamSpace.from_reference_world(ref, pin=DEFAULT_PIN + ARCH_PIN, keep=cfg.KEEP)


def test_prior_var_is_exactly_uniform01():
    _, ps = _tiny_space()
    for k in range(len(ps)):
        assert ps.prior_var(k) == pytest.approx(1.0 / 12.0)
    assert PRIOR_VAR_UNIFORM01 == pytest.approx(1.0 / 12.0)


def test_round_trip_normalize_denormalize():
    struct, ps = _tiny_space()
    rng = np.random.default_rng(123)
    u = ps.sample_prior(rng)
    assert u.shape == (len(ps),)
    assert np.all((u >= 0.0) & (u <= 1.0))

    world = generate_world(struct, rng=np.random.default_rng(0), with_artifacts=True)
    ps.set_vector(world, u)
    u2 = ps.normalize(world)
    assert np.allclose(u, u2, atol=1e-6)


def test_denormalize_respects_bounds():
    _, ps = _tiny_space()
    raw_lo = ps.denormalize(np.zeros(len(ps)))
    raw_hi = ps.denormalize(np.ones(len(ps)))
    assert np.allclose(raw_lo, ps.lo, atol=1e-6)
    assert np.allclose(raw_hi, ps.hi, atol=1e-6)


def test_validity_mask_phase1_is_all_true():
    _, ps = _tiny_space()
    mask = ps.validity_mask()
    assert mask.dtype == bool
    assert mask.shape == (len(ps),)
    assert mask.all()
    # accepts (and ignores) a composition argument, per the Phase-2-forward-compatible signature
    assert ps.validity_mask(composition={"anything": 1}).all()


def test_log_prior_in_and_out_of_range():
    _, ps = _tiny_space()
    u = np.full(len(ps), 0.5)
    assert ps.log_prior(u) == 0.0
    u_bad = u.copy()
    u_bad[0] = 1.5
    assert ps.log_prior(u_bad) == -np.inf
    assert ps.log_prior_marginal(0, 0.3) == 0.0
    assert ps.log_prior_marginal(0, -0.1) == -np.inf


def test_group_strips_instance_index_not_fixed_names():
    _, ps = _tiny_space()
    for path, group in zip(ps.paths, ps.groups):
        if "DAPI" in path:
            assert "DAPI" in group
        if "PANEL.Markers.M" in path:
            assert "PANEL.Markers.*" in group


def test_serialisation_round_trip(tmp_path):
    _, ps = _tiny_space()
    p = tmp_path / "param_space.json"
    ps.save(str(p))
    ps2 = ParamSpace.load(str(p))
    assert ps2.paths == ps.paths
    assert ps2.groups == ps.groups
    assert np.allclose(ps2.lo, ps.lo)
    assert np.allclose(ps2.hi, ps.hi)
    assert ps2.tf == ps.tf


def test_different_compositions_give_different_layouts():
    _, ps_a = _tiny_space(struct_seed=1, n_markers=4, n_types=3)
    _, ps_b = _tiny_space(struct_seed=2, n_markers=6, n_types=5)
    assert ps_a.paths != ps_b.paths
    # but group identities should still be drawn from the same finite vocabulary of roles
    roles_a = set(ps_a.groups)
    roles_b = set(ps_b.groups)
    assert roles_a & roles_b
