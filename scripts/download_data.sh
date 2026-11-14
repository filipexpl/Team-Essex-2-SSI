#!/usr/bin/env bash
# Public datasets -> data/raw. Run on a machine with internet (not needed for CI smoke tests).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/physionet data/raw/gaddy

echo "== PhysioNet EEGMMIDB (motor imagery, 109 subj, 64ch @160Hz, ~3.4 GB) =="
echo "   Downloaded lazily per subject by mne.datasets.eegbci on first eval."
echo "   Full copy: aws s3 sync --no-sign-request s3://physionet-open/eegmmidb/1.0.0/ data/raw/physionet"

echo "== Gaddy & Klein silent speech EMG (8ch @1000Hz, ~20 h, 3.9 GB) =="
if [ ! -f data/raw/gaddy/.done ]; then
  curl -L -o data/raw/gaddy/emg_data.tar.gz "https://zenodo.org/records/4064409/files/emg_data.tar.gz?download=1" \
    && tar -xzf data/raw/gaddy/emg_data.tar.gz -C data/raw/gaddy && touch data/raw/gaddy/.done \
    || echo "   !! Check exact file name on https://zenodo.org/records/4064409"
fi

echo "== BCI Competition IV 2a (optional, via MOABB: BNCI2014_001) =="
python -c "import moabb" 2>/dev/null && echo "   moabb installed - loads on demand" || echo "   pip install -e '.[eeg]'"
