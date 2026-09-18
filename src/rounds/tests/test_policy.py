"""policy.decide() tests -- pure function, synthetic eval histories, no rendering/training
needed at all. Runs under either env (numpy + yaml only)."""
import pytest

from policy import decide, format_decision_log
from proposal import Proposal


def _param(nmse, ci_half=0.05, bins=5, mse=None, binned_mse=None, binned_count=None):
    mse = mse if mse is not None else nmse / 12.0
    return dict(nmse=nmse, nmse_ci=[nmse - ci_half, nmse + ci_half], mse=mse,
               binned_mse=binned_mse or [mse] * bins,
               binned_count=binned_count or [50] * bins)


def _config(**policy_overrides):
    policy = dict(top_m=2, nmse_floor=0.05, delta_min=0.02, patience=2, max_ratio=3.0,
                 min_bin_count=10, prior_mix=0.3, ema=0.5, exclude=[])
    policy.update(policy_overrides)
    return dict(policy=policy, proposal=dict(alpha_prior=0.5, bins=5, widen=False))


def _empty_history():
    return dict(picks_by_round=[], lag_by_round=[], saturation_state={})


def test_no_candidates_gives_prior_only_proposal():
    eval_history = [{"a": _param(0.01), "b": _param(0.02)}]  # both below floor
    proposal, log = decide(eval_history, _empty_history(), _config())
    assert proposal["components"] == [dict(kind="prior", weight=1.0)]
    assert log["n_candidates"] == 0
    assert log["picks"] == []


def test_top_m_picks_worst_by_nmse_ties_broken_by_name():
    eval_history = [{"z": _param(0.5), "a": _param(0.5), "m": _param(0.9), "low": _param(0.01)}]
    proposal, log = decide(eval_history, _empty_history(), _config(top_m=2))
    assert log["picks"] == ["m", "a"]  # highest nMSE first, then alphabetical tie-break


def test_exclude_list_is_never_picked():
    eval_history = [{"a": _param(0.9), "b": _param(0.5)}]
    proposal, log = decide(eval_history, _empty_history(), _config(top_m=2, exclude=["a"]))
    assert "a" not in log["picks"]
    assert log["per_parameter"]["a"]["status"] == "excluded"


def test_below_floor_never_picked():
    eval_history = [{"a": _param(0.9), "b": _param(0.01)}]
    proposal, log = decide(eval_history, _empty_history(), _config(top_m=2, nmse_floor=0.05))
    assert "b" not in log["picks"]
    assert log["per_parameter"]["b"]["status"] == "below_floor"


def test_proposal_weights_sum_to_one_and_match_alpha():
    eval_history = [{"a": _param(0.9), "b": _param(0.8)}]
    proposal, log = decide(eval_history, _empty_history(), _config(top_m=2))
    weights = [c["weight"] for c in proposal["components"]]
    assert weights[0] == pytest.approx(0.5)  # alpha_prior
    assert sum(weights) == pytest.approx(1.0)
    assert len(proposal["components"]) == 3  # prior + 2 picks
    Proposal.from_dict(proposal)  # must round-trip through the real class without raising


def test_saturation_marks_after_patience_consecutive_non_improving_rounds():
    cfg = _config(patience=2, delta_min=0.02)
    # round 0 eval (before), round 1 eval (after targeting "a" with near-zero improvement)
    eval_history = [{"a": _param(0.5)}, {"a": _param(0.49)}]  # delta=0.01 < delta_min=0.02
    targeting_history = dict(
        picks_by_round=[[], [dict(param="a", bin_weights=[0.2] * 5)]],
        lag_by_round=[0, 0], saturation_state={},
    )
    _, log = decide(eval_history, targeting_history, cfg)
    assert log["saturation_state"]["a"]["streak"] == 1
    assert log["saturation_state"]["a"]["saturated"] is False

    # simulate a second non-improving round with the streak carried in
    eval_history2 = eval_history + [_row("a", 0.485)]
    targeting_history2 = dict(
        picks_by_round=targeting_history["picks_by_round"] +
        [[dict(param="a", bin_weights=[0.2] * 5)]],
        lag_by_round=[0, 0, 0], saturation_state=log["saturation_state"],
    )
    _, log2 = decide(eval_history2, targeting_history2, cfg)
    assert log2["saturation_state"]["a"]["streak"] == 2
    assert log2["saturation_state"]["a"]["saturated"] is True
    # a saturated parameter must not be picked again even if it still has high nMSE
    eval_history3 = eval_history2 + [_row("a", 0.4)]
    targeting_history3 = dict(picks_by_round=targeting_history2["picks_by_round"] + [[]],
                              lag_by_round=[0, 0, 0, 0],
                              saturation_state=log2["saturation_state"])
    _, log3 = decide(eval_history3, targeting_history3, _config(top_m=1))
    assert log3["picks"] == []
    assert log3["per_parameter"]["a"]["status"] == "saturated"


def _row(name, nmse):
    return {name: _param(nmse)}


def test_manual_reset_of_saturation_state_takes_effect_immediately():
    cfg = _config(patience=1)
    eval_history = [{"a": _param(0.5)}, {"a": _param(0.5)}]  # no improvement at all
    saturated_state = {"a": {"saturated": True, "streak": 5}}
    targeting_history = dict(picks_by_round=[[], []], lag_by_round=[0, 0],
                             saturation_state=saturated_state)
    _, log = decide(eval_history, targeting_history, cfg)
    # a manual reset means the CALLER passes saturated=False in -- decide() must respect that
    # and not immediately re-derive "saturated" purely from old history on its own
    reset_state = {"a": {"saturated": False, "streak": 0}}
    targeting_history_reset = dict(picks_by_round=[[], []], lag_by_round=[0, 0],
                                   saturation_state=reset_state)
    _, log2 = decide(eval_history, targeting_history_reset, cfg)
    assert log2["per_parameter"]["a"]["status"] != "saturated"


def test_bin_weights_ratio_capped_by_max_ratio():
    binned_mse = [0.001, 0.001, 100.0, 0.001, 0.001]  # bin 2 wildly worse than average
    p = _param(0.5, mse=0.02, binned_mse=binned_mse, binned_count=[50] * 5)
    eval_history = [{"a": p}]
    proposal, log = decide(eval_history, _empty_history(), _config(top_m=1, max_ratio=3.0))
    bw = log["per_parameter"]["a"]["bin_weights"]
    # bin 2's raw ratio would be 100/0.02=5000, clipped to max_ratio=3 -- so its weight should
    # be bounded, not dominate the distribution
    assert bw[2] < 0.9


def test_bin_weights_use_ratio_one_below_min_bin_count():
    # bin 2's real MSE ratio (100/0.02, clipped to max_ratio) would make it dominate -- but its
    # count (2) is below min_bin_count, so it should be treated as ratio=1 (neutral) instead.
    binned_mse = [0.001, 0.001, 100.0, 0.001, 0.001]
    p_untrusted = _param(0.5, mse=0.02, binned_mse=binned_mse, binned_count=[50, 50, 2, 50, 50])
    p_trusted = _param(0.5, mse=0.02, binned_mse=binned_mse, binned_count=[50, 50, 50, 50, 50])
    cfg = _config(top_m=1, min_bin_count=10, max_ratio=3.0)

    _, log_untrusted = decide([{"a": p_untrusted}], _empty_history(), cfg)
    _, log_trusted = decide([{"a": p_trusted}], _empty_history(), cfg)
    bw_untrusted = log_untrusted["per_parameter"]["a"]["bin_weights"]
    bw_trusted = log_trusted["per_parameter"]["a"]["bin_weights"]
    # with enough samples, bin 2's real (clipped) ratio pushes its weight up; below the count
    # threshold, it gets the same neutral ratio=1 as an average bin instead
    assert bw_trusted[2] > bw_untrusted[2]


def test_ema_smooths_toward_previous_round_bin_weights():
    p = _param(0.5, mse=0.02, binned_mse=[0.02] * 5, binned_count=[50] * 5)  # flat -> g_raw uniform
    eval_history = [{"a": p}]
    prev_bw = [0.6, 0.1, 0.1, 0.1, 0.1]
    targeting_history = dict(picks_by_round=[[dict(param="a", bin_weights=prev_bw)]],
                             lag_by_round=[0], saturation_state={})
    _, log = decide(eval_history, targeting_history, _config(top_m=1, ema=0.9))
    bw = log["per_parameter"]["a"]["bin_weights"]
    # heavy ema (0.9 toward prev) should keep bin 0 elevated relative to a flat/uniform result
    assert bw[0] > 0.3


def test_edge_bin_warning_triggers_when_only_edges_are_upweighted():
    binned_mse = [100.0, 0.001, 0.001, 0.001, 100.0]
    p = _param(0.5, mse=0.02, binned_mse=binned_mse, binned_count=[50] * 5)
    eval_history = [{"a": p}]
    _, log = decide(eval_history, _empty_history(), _config(top_m=1, max_ratio=10.0))
    assert log["per_parameter"]["a"]["edge_bin_warning"] is True
    assert any("edge bin" in w for w in log["warnings"])


def test_decide_is_deterministic_and_side_effect_free():
    eval_history = [{"a": _param(0.5), "b": _param(0.6)}]
    th = dict(picks_by_round=[[]], lag_by_round=[0], saturation_state={})
    r1 = decide(eval_history, th, _config())
    r2 = decide(eval_history, th, _config())
    assert r1 == r2
    assert th == dict(picks_by_round=[[]], lag_by_round=[0], saturation_state={})


def test_format_decision_log_runs_without_error():
    eval_history = [{"a": _param(0.9), "b": _param(0.01)}]
    _, log = decide(eval_history, _empty_history(), _config(top_m=1))
    text = format_decision_log(log)
    assert "round" in text and "a" in text


def test_empty_eval_history_raises():
    with pytest.raises(ValueError):
        decide([], _empty_history(), _config())
