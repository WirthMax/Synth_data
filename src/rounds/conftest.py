import os
import sys

_ROUNDS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_ROUNDS)
_TESTS = os.path.join(_ROUNDS, "tests")
# --import-mode=importlib (see pytest.ini) doesn't auto-add a test file's own directory to
# sys.path the way classic mode does, so sibling test-helper modules (helpers.py,
# train_helpers.py) need rounds/tests/ on sys.path explicitly, same as rounds/ itself.
for _p in (_SRC, _ROUNDS, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
