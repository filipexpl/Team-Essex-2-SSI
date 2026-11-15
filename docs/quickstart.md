# Quickstart on the device (macOS and Windows)

Start BrainAccess Board, connect the EXG8 and wait until every contact dot is green.
Then, in a terminal in the repo folder (the one that contains `pyproject.toml`):

| Step | macOS (Terminal) | Windows (PowerShell) |
|---|---|---|
| Python 3.10+ | `python3 --version` | `python --version` |
| Create env (once) | `python3 -m venv .venv` | `python -m venv .venv` |
| Activate | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` (if blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`) |
| Install (once) | `pip install -e ".[dev,hw]"` | same |
| Tests | `pytest -q` | same |

After activation the commands are identical on both systems:

```bash
python scripts/board_check.py      # fs, channel names, packet rate
python scripts/qc.py               # relaxed vs jaw clench, per channel (or: python qc.py / make qc)
python scripts/calibrate.py --task ss --user P02 --device board --vocab core6 --trials 15
python scripts/realtime.py --task ss --user P02 --device board
```

`qc.py` ends with `OK` or a list of what to fix (flat or noisy electrodes, no response to a jaw
clench = REF/BIAS contact). Calibrate only after `OK`.

Notes
- Channel names depend on the Board installation: `F1`-`F8` on our Mac, `F3, F4, C3, C4, P3, P4,
  O1, O2` on our Windows PC (same EXG8 inputs 1-8). The code detects them automatically, but a
  profile stores the names it was trained on: calibrate and decode on the same computer.
- `make` targets (`make qc`, `make calibrate TASK=ss PID=P02 DEVICE=board`) work on macOS; on
  Windows use the `python ...` commands (no `make` by default).
- Windows only: if a test fails with `UnicodeDecodeError` on an old checkout, `$env:PYTHONUTF8=1`.
