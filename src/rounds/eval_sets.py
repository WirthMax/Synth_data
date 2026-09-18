"""Section 4: fixed validation/test/stress sets, drawn once from the prior in their own seed
namespace so they never overlap with each other or with any training round.

CLI: `eval_sets.py --run-dir D --config C --kind {val,test,stress} [--n N]`
"""
import _pathfix  # noqa: F401

import argparse
import json
import os

from composition import CompositionConfig
from eval_paths import NAMESPACE, eval_set_dir, require_not_test_path  # noqa: F401 (re-exported)
from generate import ensure_param_space, generate_samples, load_config
from proposal import Proposal


def make_eval_set(run_dir, cfg_dict, kind, n=None, composition_mode="single", repo_dir=None):
    if composition_mode != "single":
        raise NotImplementedError(
            "composition_mode='full' (Phase 2) is not implemented -- eval sets are currently "
            "always drawn under the run's one fixed Phase 1 composition"
        )
    gen_cfg = cfg_dict["generation"]
    comp = CompositionConfig.from_dict(cfg_dict["phase1"]["composition"])
    master_seed = int(cfg_dict["run"]["master_seed"])
    keep = cfg_dict.get("keep")
    if n is None:
        n = cfg_dict["eval_sets"][f"{kind}_n"]

    ps = ensure_param_space(run_dir, comp, keep=keep)
    out_dir = eval_set_dir(run_dir, kind)

    # val/test/stress are always drawn from the PRIOR, never a targeted proposal -- an eval set
    # that itself got curriculum-reshaped would no longer be a fair, fixed yardstick for the
    # policy to score rounds against. Under Option A (prior == hard bounds), "stress" (wider
    # ranges, still within the hard bounds) has no room to be wider than val/test already are --
    # noted as a known no-op until a narrower Option-B prior gives it something to widen beyond.
    proposal = Proposal.prior_only(NAMESPACE[kind])

    result = generate_samples(out_dir, master_seed, NAMESPACE[kind], n, proposal, comp, gen_cfg,
                              ps, keep=keep)
    meta = dict(kind=kind, composition_mode=composition_mode, **result,
               n_fitted_params=len(ps))
    with open(os.path.join(out_dir, "eval_set_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", required=True, choices=sorted(NAMESPACE))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--composition-mode", default="single", choices=["single", "full"])
    a = ap.parse_args()

    cfg_dict = load_config(a.config)
    meta = make_eval_set(a.run_dir, cfg_dict, a.kind, n=a.n,
                         composition_mode=a.composition_mode)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
