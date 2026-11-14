# CLAUDE.md — instrukcje dla Claude (Claude Code / GitHub Action)

## Cel projektu
Hackathon neurotech. Dwa strumienie dekodowania, jedna wspólna aplikacja:
1. **Silent speech (sEMG twarzy)** — 8 elektrod na twarzy, klasyfikacja małego słownika **angielskich** słów/komend wypowiadanych bezgłośnie (tylko EN — nie dodawaj polskich słów).
2. **Intencje motoryczne (EEG, motor imagery)** — 4 klasy: `left_hand`, `right_hand`, `both_hands`, `feet` (+ opcjonalnie `rest`).

Sprzęt: **BrainAccess MINI** (8 kanałów EEG, elektrody suche, BT) oraz **EXG8** (8 kanałów, do sEMG twarzy).
Parametry sprzętu, których NIE zgadujemy — patrz `docs/hardware.md` (sekcja "Do potwierdzenia").

## Złote zasady
- **Nie zmieniaj progów w `evals/thresholds.yaml`, żeby testy przeszły.** Jeśli model nie spełnia progu, napisz to wprost w PR.
- **Zawsze oceniaj na podziale cross-session / cross-subject** zdefiniowanym w `src/ssmi/evals/splits.py`. Nigdy random split po oknach z tej samej próby (wyciek danych).
- Każda nowa metoda = nowy plik w `src/ssmi/models/`, rejestracja w `registry.py`, wpis w `docs/decisions.md`.
- Kod musi działać **na CPU** i na danych syntetycznych (`ssmi.acquisition.synthetic`) — CI nie ma sprzętu ani GPU.
- Częstotliwość próbkowania zawsze bierz z metadanych nagrania (`Recording.fs`), nigdy z hardkodu.
- Surowych danych (`data/raw`) i nagrań uczestników nie commitujemy. Dane osobowe = tylko ID uczestnika.

## Komendy
```bash
make setup        # pip install -e ".[dev]"
make test         # pytest -q
make lint         # ruff check + ruff format --check
make eval-smoke   # eval na danych syntetycznych (CI)
make eval TASK=mi DATASET=physionet MODEL=csp_lda
make eval TASK=ss DATASET=gaddy MODEL=td_features_lda
make data         # pobiera publiczne datasety do data/raw (wymaga sieci)
python scripts/board_check.py   # PIERWSZE na sprzęcie: fs, kanały, jednostki, lag, markery
make calibrate TASK=ss PID=P07 DEVICE=synthetic|board   # kalibracja nowej osoby → profiles/
make live TASK=ss PID=P07 DEVICE=synthetic|board        # dekodowanie na żywo z profilu
make calib-curve TASK=ss DATASET=own MODEL=td_features_lda  # ile prób kalibracji trzeba
```

## Architektura
```
src/ssmi/
  acquisition/   # board.py (BrainAccess Board API, główna ścieżka) / LSL / synthetic
  preprocessing/ # filtry (bandpass, notch 50 Hz), epokowanie, resampling
  datasets/      # loadery: PhysioNet EEGMMIDB, BCI IV 2a, Gaddy EMG, nagrania własne
  models/        # baseline'y + registry; interfejs sklearn (fit/predict/predict_proba)
  calibration/   # Euclidean Alignment + UserProfile (kalibracja nowej osoby)
  evals/         # metryki, splity, runner, krzywa kalibracji, raport JSON → results/
scripts/         # calibrate.py (nagranie z cue + profil), live.py (dekodowanie)
configs/vocab.yaml  # słowniki EN (silent speech TYLKO po angielsku) + parametry protokołu
```
Kalibracja i datasety: `docs/calibration.md`, `docs/datasets.md`. Każda zmiana modelu powinna
być sprawdzona także krzywą kalibracji (nowa osoba = najważniejszy scenariusz demo).
Wspólny kontrakt danych: `X: (n_trials, n_channels, n_samples)`, `y: (n_trials,)` int, `meta: dict` (fs, ch_names, subject, session).

## Definicja "done" dla zadania
1. `make lint test eval-smoke` przechodzi lokalnie.
2. Wynik eval zapisany w `results/<task>_<dataset>_<model>.json` i porównany z baseline w opisie PR.
3. Zaktualizowany `docs/decisions.md` (1–3 zdania: co, dlaczego, wynik).

## Konwencje
- Python 3.11, type hints, ruff (line length 100), numpy-style docstringi tylko dla publicznego API.
- Losowość: zawsze `seed` z configu.
- Commity: `feat(mi): ...`, `fix(ss): ...`, `eval: ...`, `docs: ...`.

## Kontekst domenowy (skrót)
- MI: mu (8–12 Hz) i beta (13–30 Hz) nad C3/Cz/C4; ręce = lateralizacja ERD, stopy = Cz (środek). Baseline: filtr 8–30 Hz → CSP (one-vs-rest) → LDA, ewentualnie Riemann (tangent space).
- Klasa `both_hands` vs `feet` jest trudna przy 8 suchych elektrodach — raportuj macierz pomyłek.
- sEMG twarzy: energia głównie 20–450 Hz. Sprzęt obsługuje 250 i 500 Hz — **nagrywamy EMG przy 500 Hz** (pasmo do 250 Hz). Do porównań z IITK (250 Hz) downsamplujemy offline; dataset Gaddy (1000 Hz) trzeba **downsamplować do fs urządzenia** przed transferem.
- Typowe artefakty: mruganie (EOG) w EEG, mimika/żucie w EMG, 50 Hz sieć (Polska!).
