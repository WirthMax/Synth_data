"""Import first, before any bare `from <src-module> import ...`, in every rounds/*.py entry
point. `src/` is not a package (see CLAUDE.md) -- its modules assume `src/` is on sys.path,
which holds automatically only when running from inside `src/`. This makes every rounds/*.py
script runnable from any working directory, which the orchestrator needs when it invokes them
as subprocesses from a run directory that lives elsewhere.
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
