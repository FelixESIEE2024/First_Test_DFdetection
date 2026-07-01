from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    target = repo_root / "implementation" / "run_depth_pose_sequence.py"
    venv_python = repo_root / ".venv" / "Scripts" / "python.exe"

    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        raise SystemExit(subprocess.call([str(venv_python), str(target), *sys.argv[1:]]))

    runpy.run_path(str(target), run_name="__main__")
