# Hardware

## BrainAccess MINI — EEG (motor imagery)
- 8 channels EEG, common reference, dry-contact electrodes on a BrainAccess CAP (10-20 positions), Bluetooth.
- Software: BrainAccess Board (GUI, Windows/Linux), BrainAccess SDK (C / Python, `pip install brainaccess`).
- Planned montage for MI (motor strip): **FC3, FC4, C3, Cz, C4, CP3, CPz, CP4**.
  Reference/bias on mastoids or behind the ears. If the cap does not allow these positions, the closest
  ones on the motor strip. Frontal channels are NOT needed for MI (they mostly pick up blinks).

## EXG8 — facial sEMG (silent speech)
- 8 channels. Facial EMG needs **wet/gel Ag/AgCl electrodes** (dry spikes from the cap do not work on skin).
- Placement (based on Gaddy & Klein 2020 + classic SSI literature), one muscle per channel:
  1. levator anguli oris / zygomaticus (cheek, L)
  2. zygomaticus (cheek, R)
  3. orbicularis oris — upper lip
  4. orbicularis oris — lower lip
  5. depressor anguli oris (corner of mouth)
  6. mentalis (chin)
  7. submental / anterior digastric (under the chin — tongue!)
  8. platysma / laryngeal (neck, side)
  Reference: earlobe or mastoid. Ground: forehead or wrist.
  Take a phone photo of the placement every session (relative to landmarks) → `docs/placement/`.

## Do potwierdzenia na sprzęcie (Claude: NIE zgaduj tych wartości)
| Parametr | MINI | EXG8 | Dlaczego ważne |
|---|---|---|---|
| Sampling rate (Hz) | ? (BrainAccess devices: 250 Hz on public specs) | **250 or 500 Hz (team, 2026-11-14) → record at 500 Hz** | 500 Hz: EMG band up to 250 Hz; 250 Hz only for IITK comparisons (downsample offline) |
| Bipolar / monopolar inputs | monopolar, common ref | ? | EMG lepiej bipolarnie |
| Hardware filters | ? | ? | high-pass dla EMG |
| Dry or gel electrodes allowed | dry | ? | |
| Streaming | Board Python API (`brainaccess_board`, 1.0.2): `db_connect` + `get_mne` → `BoardSource` | same | run `python scripts/board_check.py` and paste the output here |
| Units / DB lag / markers in .db | ? | ? | `board_check.py` prints all three |
| Can both devices run at once, synced? | hyperscanning supported | ? | hybrid EEG+EMG |

## Risks
- Record EMG at 500 Hz (Nyquist 250 Hz): keeps most of the sEMG band (20–450 Hz). At 250 Hz part of the band is lost. Still enough for word classification
  (OpenBCI Cyton @250 Hz has been used for silent speech), but no direct transfer from 1000 Hz data
  without resampling.
- Dry EEG + 8 channels: 4-class MI is hard. Expect ~40–60 % for our own data; plan a 2-class
  (left vs right) fallback demo and use `both_hands`/`feet` as stretch goal.
- Bluetooth packet loss → check effective fs in `signal-qc`.

## Acquisition path (2026-11-14)
BrainAccess Board Python API 1.0.2 (`pip install brainaccess_board`, Board app running, amplifier
connected): live data via `db_connect()` + `get_mne(duration)`, recording control via
`msg_connect()` commands (`start_recording` / `stop_recording`), markers as our own LSL outlet
attached with `connect_device`. Implemented in `src/ssmi/acquisition/board.py`; use
`--device board` in `calibrate.py` / `live.py`. LSLSource stays as a fallback.

## Measured on EXG8 via BrainAccess Board (2026-11-14, Mac, first recordings)
- Board streams **25 channels at 1000 Hz**: `F1`-`F8` = EMG electrodes; `contact_F1`-`contact_F8`
  (constant flags), `Accel_x/y/z`, `Gyro_x/y/z`, `Sample` (counter), `Streaming`, `Battery`.
  All are reported as signal channels, so `ssmi.acquisition.channels.select_channels` keeps only
  electrodes (by name, and drops constant channels). Use `--channels F1,F2,...` to override.
- Units after BoardSource (x1e6): microvolts. EMG channels: DC offset ~ -2000..9000 uV (removed by
  the band-pass), std ~80-180 uV during the calibration block.
- Before this fix, calibration fed all 25 channels to the model (numerical overflow, CV 0.30-0.32).
