from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    target = repo_root / "implementation" / "run_depth_pose_sequence.py"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"
    original_command = subprocess.list2cmdline([Path(sys.argv[0]).name, *sys.argv[1:]])

    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        child_env = os.environ.copy()
        child_env["RUN_DEPTH_POSE_SEQUENCE_COMMAND"] = original_command
        raise SystemExit(
            subprocess.call(
                [str(venv_python), str(target), *sys.argv[1:]],
                env=child_env,
            )
        )

    os.environ.setdefault("RUN_DEPTH_POSE_SEQUENCE_COMMAND", original_command)
    runpy.run_path(str(target), run_name="__main__")
