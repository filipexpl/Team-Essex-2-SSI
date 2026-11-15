"""Kept for convenience: `python qc.py` == `python scripts/qc.py` (channel names auto-detected)."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "qc.py"), run_name="__main__")
