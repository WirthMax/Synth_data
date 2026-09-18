"""Proposal/importance-weight tests. Pure math against a small synthetic ParamSpace -- no
renderer involved, since none of this touches rendering."""
import numpy as np
import pytest

from param_space import ParamSpace
from proposal import (
    Proposal,
    RoundRecord,
    effective_sample_size,
    importance_weights,
    log_density,
    piecewise_constant_log_density,
    piecewise_constant_sample,
    sample_theta,
)


def _tiny_param_space(n=4):
    paths = [f"p{i}" for i in range(n)]
    return ParamSpace(paths=paths, groups=paths, lo=[0.0] * n, hi=[1.0] * n, tf=["linear"] * n)


def test_piecewise_constant_sample_always_in_unit_interval():
    rng = np.random.default_rng(0)
    bw = [0.0, 0.7, 0.3, 0.0, 0.0]
    xs = np.array([piecewise_constant_sample(rng, bw) for _ in range(2000)])
    assert np.all((xs >= 0.0) & (xs <= 1.0))
    # zero-weight bins [0, 0.2) and [0.6, 1.0) should essentially never be hit
    assert np.mean((xs >= 0.0) & (xs < 0.2)) < 0.01
    assert np.mean((xs >= 0.6) & (xs < 1.0)) < 0.01


def test_piecewise_constant_log_density_integrates_to_one():
    bw = [0.1, 0.4, 0.2, 0.2, 0.1]
    b = len(bw)
    # midpoint rule over the 5 bins reproduces the exact integral for a piecewise-constant fn
    total = sum(np.exp(piecewise_constant_log_density((i + 0.5) / b, bw)) * (1.0 / b)
                for i in range(b))
    assert total == pytest.approx(1.0, abs=1e-9)
    assert piecewise_constant_log_density(0.05, bw) == pytest.approx(np.log(0.1 * b))


def test_piecewise_constant_marginal_matches_bin_weights():
    rng = np.random.default_rng(1)
    bw = np.array([0.1, 0.1, 0.7, 0.1])
    xs = np.array([piecewise_constant_sample(rng, bw) for _ in range(40000)])
    b = len(bw)
    counts = np.array([np.mean((xs >= i / b) & (xs < (i + 1) / b)) for i in range(b)])
    assert np.allclose(counts, bw, atol=0.02)


def test_sample_theta_all_dims_in_unit_interval_and_component_recorded():
    ps = _tiny_param_space(5)
    proposal = Proposal.targeted(0, picks=[("p2", [0.0, 0.0, 1.0, 0.0, 0.0])],
                                 alpha_prior=0.5, bins=5, param_space=ps)
    for i in range(200):
        u, comp_id = sample_theta(ps, proposal, np.random.default_rng(1000 + i))
        assert u.shape == (5,)
        assert np.all((u >= 0.0) & (u <= 1.0))
        assert comp_id in (0, 1)


def test_targeted_reshape_only_moves_the_targeted_dimension_marginal():
    ps = _tiny_param_space(3)
    # target p1 hard into the top bin; p0/p2 must still look like the prior (uniform)
    proposal = Proposal.targeted(0, picks=[("p1", [0.0, 0.0, 0.0, 0.0, 1.0])],
                                 alpha_prior=0.5, bins=5, param_space=ps)
    us = np.array([sample_theta(ps, proposal, np.random.default_rng(i))[0]
                   for i in range(20000)])
    # p1 (index 1): half the mass should sit in [0.8, 1.0) (the reshape share) plus the prior's
    # own 1/5 baseline share of that top bin from its alpha_prior fraction
    frac_top_bin_p1 = np.mean((us[:, 1] >= 0.8) & (us[:, 1] <= 1.0))
    expected = 0.5 * 1.0 + 0.5 * 0.2  # alpha_prior*prior_mass_in_bin + (1-alpha_prior)*1.0
    assert abs(frac_top_bin_p1 - expected) < 0.02
    # p0 and p2 (never targeted) should stay uniform regardless of what p1 does
    for j in (0, 2):
        assert abs(np.mean(us[:, j]) - 0.5) < 0.02
        assert abs(np.mean((us[:, j] >= 0.8) & (us[:, j] <= 1.0)) - 0.2) < 0.02


def test_log_density_prior_only_is_zero_in_range():
    ps = _tiny_param_space(3)
    proposal = Proposal.prior_only(0)
    u = np.array([0.2, 0.5, 0.9])
    assert log_density(ps, proposal, u) == pytest.approx(0.0)
    u_bad = np.array([0.2, 1.5, 0.9])
    assert log_density(ps, proposal, u_bad) == -np.inf


def test_prior_component_weight_below_alpha_raises():
    with pytest.raises(ValueError):
        Proposal(round=0, alpha_prior=0.9, bins=5, widen=False,
                components=[__import__("proposal").PriorComponent(kind="prior", weight=0.5),
                            __import__("proposal").ReshapeComponent(
                                kind="reshape", weight=0.5, param="p0",
                                bin_weights=[0.2] * 5)])


def test_serialisation_round_trip(tmp_path):
    ps = _tiny_param_space(4)
    proposal = Proposal.targeted(3, picks=[("p2", [0.1, 0.2, 0.3, 0.2, 0.2])],
                                 alpha_prior=0.6, bins=5, param_space=ps)
    p = tmp_path / "round_003.json"
    proposal.save(str(p))
    loaded = Proposal.load(str(p))
    assert loaded.round == 3
    assert loaded.alpha_prior == pytest.approx(0.6)
    assert [c.kind for c in loaded.components] == [c.kind for c in proposal.components]
    assert loaded.weights() == pytest.approx(proposal.weights())


def test_importance_weights_bounded_by_one_over_alpha_and_zero_ess_sanity():
    ps = _tiny_param_space(3)
    proposal_a = Proposal.targeted(0, picks=[("p0", [0.0, 0.0, 0.0, 0.0, 1.0])],
                                   alpha_prior=0.4, bins=5, param_space=ps)
    proposal_b = Proposal.prior_only(1)
    u_a = np.array([sample_theta(ps, proposal_a, np.random.default_rng(100 + i))[0]
                    for i in range(500)])
    u_b = np.array([sample_theta(ps, proposal_b, np.random.default_rng(900 + i))[0]
                    for i in range(500)])
    records = [RoundRecord(proposal=proposal_a, u=u_a), RoundRecord(proposal=proposal_b, u=u_b)]
    w, ess = importance_weights(ps, records)
    assert w.shape == (1000,)
    assert np.all(w >= 0.0)
    assert np.all(w <= 1.0 / 0.4 + 1e-6)
    assert 0.0 < ess <= 1000.0


def test_importance_weights_monte_carlo_recovers_prior_expectation():
    """E_qbar[w * f(u)] should approximate E_p[f(u)] for a simple f, even though the samples
    were drawn from a mixture that oversamples one region."""
    ps = _tiny_param_space(2)
    proposal = Proposal.targeted(0, picks=[("p0", [0.0, 0.0, 0.0, 0.0, 5.0])],
                                 alpha_prior=0.3, bins=5, param_space=ps)
    n = 20000
    u = np.array([sample_theta(ps, proposal, np.random.default_rng(i))[0] for i in range(n)])
    records = [RoundRecord(proposal=proposal, u=u)]
    w, _ess = importance_weights(ps, records)
    f = u[:, 0]  # E_p[u_0] = 0.5 under the true prior
    estimate = np.mean(w * f)
    assert abs(estimate - 0.5) < 0.05


def test_effective_sample_size_uniform_weights_equals_n():
    w = np.ones(50)
    assert effective_sample_size(w) == pytest.approx(50.0)
