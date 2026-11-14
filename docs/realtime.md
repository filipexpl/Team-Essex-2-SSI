# Real-time decoding

```bash
python scripts/board_check.py                                   # once per setup
make calibrate TASK=ss PID=P07 DEVICE=board                     # ~5-7 min, makes profiles/P07_ss.pkl
make realtime  TASK=ss PID=P07 DEVICE=board                     # continuous decoding, Ctrl+C to stop
make realtime  TASK=ss PID=SIM DEVICE=synthetic                 # no hardware: simulated person
python scripts/realtime.py --task mi --user P07 --device board --lsl-out   # + LSL output for a UI
```

The first 5 s are a rest baseline (stay relaxed); after that no key presses are needed.

## How it works
Every `hop` (SS 0.1 s, MI 0.25 s) the loop asks the source for its newest data
(`latest(seconds)`: BrainAccess Board `get_mne(duration)`, or an LSL ring buffer).

| | Silent speech | Motor imagery |
|---|---|---|
| Trigger | EMG energy onset: envelope above rest median + k x MAD (k = 6) for 50 ms | none: sliding window |
| Window | [onset - offset, onset - offset + 1.5 s]; `offset` = median onset inside the calibration windows, stored in the profile, so live windows match training | last 3 s |
| Decision | `predict_proba` of the calibrated profile; below `--min-conf` (0.4) prints `?` | probabilities smoothed (EMA 0.3); printed when the class changes and stays for 3 hops; below 0.5 prints `rest/unsure` |
| After a decision | refractory 1.5 s | - |

Output: console, `results/realtime_<user>_<task>.jsonl` (time, label, confidence, compute ms),
optional LSL string stream `SSMI_Decoded`.

## Measured on the simulated person (virtual time, `tests/test_realtime.py`)
Silent speech, 6 words, 90 s stream: 23/23 words detected, 0 false triggers, 18/23 correct,
compute <= 8.3 ms per decision. Motor imagery: a class change is reported a few seconds after a
block starts (3 s window + smoothing). Synthetic data: this checks the machinery, not accuracy.

## Latency budget
word end -> decision = remaining window after onset (about 1.5 s - offset) + up to one hop +
database lag of the Board (measured by `board_check.py`) + compute (< 10 ms).
For MI, expect 1.5-3 s from the start of imagery to a reported change.

## Tuning on the device
- Too many false words in rest: raise `--k` (8-10). Missed words: lower it (4-5).
- Too many `?`: lower `--min-conf`, or calibrate with more trials.
- If `get_mne` polling is slow on the Board, use `--hop 0.25`.
- MI has no trained rest class: rest shows as `rest/unsure` only when confidence is low.
  A rest class in calibration is the next improvement.
