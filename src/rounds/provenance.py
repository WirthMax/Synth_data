"""git commit / config hash -- needed by both generate.py (synmac) and train.py (synmodel), so
this has no scipy/torch/numpy dependency at all, just stdlib."""
import hashlib
import json
import subprocess


def git_commit(repo_dir):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def config_hash(cfg_dict):
    blob = json.dumps(cfg_dict, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
