"""Standalone driver for test_process_hygiene_no_orphans_after_sigkill: runs a generate_round
long enough (via ROUNDS_MOCK_SLEEP_S) for the test to observe live worker processes, then get
SIGKILLed itself. Not a pytest test file -- invoked as a subprocess."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate import generate_round  # noqa: E402
from proposal import Proposal  # noqa: E402
from helpers import make_tiny_config  # noqa: E402

if __name__ == "__main__":
    run_dir = sys.argv[1]
    cfg = make_tiny_config(workers=3, shard_size=20)
    generate_round(run_dir, cfg, 0, 20, Proposal.prior_only(0))
