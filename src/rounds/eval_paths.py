"""Eval-set path/namespace conventions and the test-set isolation guard -- split out of
eval_sets.py so train.py/evaluate.py/policy.py (some of which run under synmodel, which cannot
import eval_sets.py's generate.py dependency) can enforce test-set isolation without pulling in
the whole generation stack. stdlib only.
"""
import os

# Disjoint from any real round index (rounds run 0, 1, 2, ... in practice) and from each other,
# so val/test/stress/training-round seeds can never collide regardless of how many rounds a run
# ends up with.
NAMESPACE = dict(val=900_000_001, test=900_000_002, stress=900_000_003)

# A path segment that marks "this is a test set" -- checked by require_not_test_path below.
TEST_SET_DIRNAME = "test"


def eval_set_dir(run_dir, kind):
    if kind not in NAMESPACE:
        raise ValueError(f"kind must be one of {sorted(NAMESPACE)}, got {kind!r}")
    return os.path.join(run_dir, "eval_sets", kind)


def require_not_test_path(path, caller):
    """Raise if `path` is (or is inside) a test-set directory. Training, per-round evaluation,
    and the policy must call this on every path they're about to open -- only `finalize` is
    allowed to read a test set. Matching is structural (a path component literally named
    "test" under an "eval_sets" directory), not a heuristic on the string "test" appearing
    anywhere, so e.g. a composition field or run-dir named "latest" is never caught by this.
    """
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    for i, part in enumerate(parts[:-1]):
        if part == "eval_sets" and i + 1 < len(parts) and parts[i + 1] == TEST_SET_DIRNAME:
            raise PermissionError(
                f"{caller} tried to open {path!r}, which is inside a test-set directory -- "
                f"only `finalize` may read the test set"
            )
