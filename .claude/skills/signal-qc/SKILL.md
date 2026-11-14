---
name: signal-qc
description: Check quality of a recording (EEG from BrainAccess MINI or facial EMG from EXG8) before training. Use whenever new data lands in data/processed/own_* or a model suddenly underperforms.
---

# signal-qc

For each channel compute and report in a table:
- RMS amplitude (µV); flag flat (< 1 µV) or saturated/railed channels.
- 50 Hz line-noise ratio: power 49–51 Hz / power in task band (EEG 8–30 Hz, EMG 20–120 Hz).
- EEG only: alpha peak present at rest (eyes closed) on posterior channels; blink artifacts
  on frontal channels (peak-to-peak > 100 µV).
- EMG only: SNR = RMS during articulation / RMS during rest segments; expect > 2.
- Trial counts per class and per session (class balance).
- Effective fs from timestamps vs nominal fs (Bluetooth packet loss → gaps).

Output: a short verdict per channel (ok / check electrode / drop) and a list of trials to
reject. Do not delete raw data; write rejection list to `data/processed/<name>_reject.json`.
