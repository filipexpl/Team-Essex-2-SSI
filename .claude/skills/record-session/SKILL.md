---
name: record-session
description: Prepare or modify the cued recording protocol (MI or silent speech) and the script that saves trials to npz. Use when planning a data collection session with the BrainAccess devices.
---

# record-session

Protocol defaults (change only with the team):
- **MI**: fixation 2 s → cue (arrow ←, →, ↑ both hands, ↓ feet) 4 s imagery → rest 2–3 s
  random. 4 classes × 20 trials per run, randomized order, 3–4 runs per session, pause
  between runs. Epoch 0.5–3.5 s after cue.
- **Silent speech**: 6–10 words (PL, e.g. tak, nie, start, stop, lewo, prawo, pomoc, dalej).
  Cue word 1 s → beep → silent articulation window 1.5 s → rest 1.5 s. 20–30 reps per word
  per session. Record ≥ 3 sessions with electrodes re-attached between them (to test
  cross-session generalization). Also record 30 s of rest and some vocalized repetitions.
- Save each session as `data/processed/own_<task>/<participant>_<session>.npz` with keys
  X, y, fs, ch_names, classes, groups (session id). Participant = anonymous ID only.
- Use LSL markers for cue timing; never rely on wall-clock sleep for epoch alignment.
- After each session run the `signal-qc` skill.
