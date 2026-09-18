"""Section 2: round proposals (mixture over prior + targeted reshape components) and the
importance-weight machinery that corrects for having sampled from them instead of the prior.

A round's proposal is a MIXTURE. Every component draws a FULL theta vector; components only
differ in how they draw one targeted parameter's marginal -- every other dimension always comes
from the prior (Design decision: "only the targeted parameter is reshaped", never held fixed).
Because ParamSpace's prior is Option A (prior == hard bounds, uniform density 1 on [0, 1] for
every dimension), a component's total density collapses to just its targeted marginal's density
wherever the whole vector is in-range -- `log_density` below computes that directly rather than
carrying redundant products of 1s.
"""
import _pathfix  # noqa: F401

import dataclasses
import json

import numpy as np


def logsumexp(x, axis=None):
    """log(sum(exp(x))), numerically stable, no scipy dependency -- Proposal/importance_weights
    are needed under BOTH envs (generation-side sampling under synmac, finalize.py's
    importance-weighted fine-tune under synmodel), and synmodel has no scipy."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    m_nokeep = np.max(x, axis=axis)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(x - safe_m), axis=axis)
    out = np.log(s) + m_nokeep
    return np.where(np.isfinite(m_nokeep), out, m_nokeep)


def round_filename(round_idx):
    return f"round_{round_idx:03d}.json"


@dataclasses.dataclass
class ReshapeComponent:
    """theta[param] ~ piecewise-constant density `bin_weights` over B equal bins of [0, 1];
    every other dimension ~ prior. `bin_weights` sums to 1 (a probability mass per bin, not a
    density -- `piecewise_constant_log_density` divides by bin width to get the density)."""

    kind: str
    weight: float
    param: str
    bin_weights: list

    def __post_init__(self):
        if self.kind not in ("reshape", "real_data"):
            raise ValueError(f"ReshapeComponent.kind must be 'reshape' or 'real_data': {self.kind}")


@dataclasses.dataclass
class PriorComponent:
    kind: str
    weight: float

    def __post_init__(self):
        if self.kind != "prior":
            raise ValueError(f"PriorComponent.kind must be 'prior': {self.kind}")


def _component_from_dict(d):
    if d["kind"] == "prior":
        return PriorComponent(kind="prior", weight=float(d["weight"]))
    return ReshapeComponent(kind=d["kind"], weight=float(d["weight"]), param=d["param"],
                             bin_weights=list(d["bin_weights"]))


@dataclasses.dataclass
class Proposal:
    round: int
    alpha_prior: float
    bins: int
    widen: bool
    components: list  # first entry is always the PriorComponent

    def __post_init__(self):
        if not self.components or self.components[0].kind != "prior":
            raise ValueError("Proposal.components[0] must be the prior component")
        w = self.weights()
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"component weights must sum to 1, got {w.sum()}")
        if self.components[0].weight < self.alpha_prior - 1e-9:
            raise ValueError(
                f"prior weight {self.components[0].weight} is below alpha_prior "
                f"{self.alpha_prior} -- every round must keep at least alpha_prior on the prior"
            )

    def weights(self):
        return np.array([c.weight for c in self.components], dtype=float)

    # -- construction ---------------------------------------------------------------------------

    @classmethod
    def prior_only(cls, round_idx, alpha_prior=1.0, bins=10, widen=False):
        return cls(round=round_idx, alpha_prior=alpha_prior, bins=bins, widen=widen,
                   components=[PriorComponent(kind="prior", weight=1.0)])

    @classmethod
    def targeted(cls, round_idx, picks, alpha_prior, bins, widen=False, param_space=None):
        """`picks`: list of (param, bin_weights) pairs, one per targeted parameter, each
        already normalised to sum to 1 over `bins` equal bins. The prior keeps exactly
        `alpha_prior`; the remaining `1 - alpha_prior` is split EVENLY across picks (Design
        decision, §7.5). If `param_space` is given, any bin with zero prior mass there is
        zeroed out unless `widen` -- under Option A this never fires (every bin has equal prior
        mass), but the check is real for a future narrower prior.
        """
        if not picks:
            return cls.prior_only(round_idx, alpha_prior=alpha_prior, bins=bins, widen=widen)
        share = (1.0 - alpha_prior) / len(picks)
        comps = [PriorComponent(kind="prior", weight=alpha_prior)]
        for param, bw in picks:
            bw = _normalise_bins(bw)
            if param_space is not None and not widen:
                bw = _zero_bins_without_prior_mass(param_space, param, bw)
            comps.append(ReshapeComponent(kind="reshape", weight=share, param=param,
                                          bin_weights=bw.tolist()))
        return cls(round=round_idx, alpha_prior=alpha_prior, bins=bins, widen=widen,
                   components=comps)

    # -- serialisation ----------------------------------------------------------------------------

    def to_dict(self):
        return dict(round=self.round, alpha_prior=self.alpha_prior, bins=self.bins,
                    widen=self.widen, components=[dataclasses.asdict(c) for c in self.components])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d):
        return cls(round=d["round"], alpha_prior=d["alpha_prior"], bins=d["bins"],
                   widen=d["widen"], components=[_component_from_dict(c) for c in d["components"]])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _normalise_bins(bin_weights):
    bw = np.asarray(bin_weights, dtype=float)
    bw = np.clip(bw, 0.0, None)
    total = bw.sum()
    if total <= 0:
        return np.full(bw.shape, 1.0 / len(bw))
    return bw / total


def _zero_bins_without_prior_mass(param_space, param, bin_weights):
    b = len(bin_weights)
    centres = (np.arange(b) + 0.5) / b
    has_mass = np.array([param_space.log_prior_marginal(param, c) > -np.inf for c in centres])
    if not has_mass.any():
        raise ValueError(f"prior has zero mass everywhere for {param!r}; cannot reshape it")
    out = np.where(has_mass, bin_weights, 0.0)
    return _normalise_bins(out)


# -- sampling -------------------------------------------------------------------------------------

def piecewise_constant_sample(rng, bin_weights):
    """One draw from the piecewise-constant density `bin_weights` (probability MASS per bin,
    summing to 1) over B equal bins of [0, 1]. Always in [0, 1] by construction."""
    bw = _normalise_bins(bin_weights)
    b = int(rng.choice(len(bw), p=bw))
    width = 1.0 / len(bw)
    return float(b * width + rng.random() * width)


def piecewise_constant_log_density(u, bin_weights):
    """log g(u) for the piecewise-constant density: height = (bin's probability mass) / (bin
    width), so that g integrates to 1 over [0, 1]."""
    bw = _normalise_bins(bin_weights)
    b = len(bw)
    idx = min(int(np.clip(u, 0.0, 1.0) * b), b - 1)
    mass = bw[idx]
    if mass <= 0:
        return -np.inf
    return float(np.log(mass) + np.log(b))


def sample_theta(param_space, proposal, rng):
    """One full theta draw AND its component id, from a single per-sample `rng` -- deterministic
    given that rng's seed. Component assignment and every dimension's value are drawn from the
    SAME stream, in a fixed order, so the whole sample is a pure function of the seed regardless
    of worker count or completion order (generate.py is responsible for seeding `rng` per
    `SeedSequence([master_seed, round, index])`).
    """
    weights = proposal.weights()
    comp_id = int(rng.choice(len(proposal.components), p=weights))
    comp = proposal.components[comp_id]
    u = param_space.sample_prior(rng)
    if comp.kind == "reshape":
        k = param_space.index_of(comp.param)
        u[k] = piecewise_constant_sample(rng, comp.bin_weights)
    elif comp.kind == "real_data":
        raise NotImplementedError(
            "the real_data proposal component is a stub (interface only, per spec) -- it must "
            "never be selected, which means it must never carry nonzero weight"
        )
    return u, comp_id


def log_density(param_space, proposal, u):
    """log q_r(u): log of this ONE round's mixture density at u. Under Option A (prior density
    1 on [0,1]^P), a component's density is 1 everywhere in range except at its one targeted
    dimension, where it's g(u_k) -- so log q_c(u) is just log g(u_k) (in range) rather than a sum
    of P-1 zeros plus one nonzero term."""
    u = np.asarray(u, dtype=float)
    if not np.all((u >= 0.0) & (u <= 1.0)):
        return -np.inf
    log_w = np.log(np.clip(proposal.weights(), 1e-300, None))
    log_qc = np.empty(len(proposal.components))
    for i, comp in enumerate(proposal.components):
        if comp.kind == "prior":
            log_qc[i] = 0.0
        elif comp.kind == "reshape":
            k = param_space.index_of(comp.param)
            log_qc[i] = piecewise_constant_log_density(u[k], comp.bin_weights)
        else:
            log_qc[i] = -np.inf  # real_data: never selectable, contributes nothing
    return float(logsumexp(log_w + log_qc))


# -- importance weights across the union of rounds -------------------------------------------------

@dataclasses.dataclass
class RoundRecord:
    """What `importance_weights` needs from one already-generated round: the proposal that
    produced it, and every one of its samples' normalised theta (N_r x P)."""

    proposal: Proposal
    u: np.ndarray  # (N_r, P)

    @property
    def n(self):
        return self.u.shape[0]


def importance_weights(param_space, records):
    """w_i = p(u_i) / qbar(u_i) for every sample across every round in `records`, where
    qbar = sum_r (N_r / N) * q_r is the aggregate proposal actually used to produce the union.
    w = 0 wherever p(u) = 0. Every round guarantees a prior share >= alpha_prior, so
    qbar(u) >= alpha_min * p(u) wherever p(u) > 0, hence w <= 1 / alpha_min -- asserted, not just
    hoped for.

    Returns (weights, ess) where weights is one flat array in the same order as
    `np.concatenate([r.u for r in records])`, and ess = (sum w)^2 / sum(w^2).
    """
    if not records:
        raise ValueError("importance_weights needs at least one round")
    ns = np.array([r.n for r in records], dtype=float)
    total_n = ns.sum()
    log_share = np.log(ns / total_n)
    alpha_min = min(r.proposal.alpha_prior for r in records)

    all_u = np.concatenate([r.u for r in records], axis=0)
    log_p = np.array([param_space.log_prior(u) for u in all_u])
    # log qbar(u) = logsumexp_r [ log_share_r + log q_r(u) ]
    log_qr = np.stack(
        [np.array([log_density(param_space, r.proposal, u) for u in all_u]) for r in records],
        axis=1,
    )  # (N, n_rounds)
    log_qbar = logsumexp(log_qr + log_share[None, :], axis=1)

    w = np.where(np.isfinite(log_p), np.exp(log_p - log_qbar), 0.0)
    cap = 1.0 / alpha_min + 1e-6
    bad = w > cap
    if bad.any():
        raise AssertionError(
            f"{bad.sum()} importance weight(s) exceed 1/alpha_min={cap:.4g} "
            f"(max observed {w.max():.4g}) -- a round's prior share must have dropped below "
            f"its declared alpha_prior"
        )
    return w, effective_sample_size(w)


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    denom = np.sum(w ** 2)
    return float((np.sum(w) ** 2) / denom) if denom > 0 else 0.0
