TASK ?= mi
PID ?= P00
DEVICE ?= synthetic
TRIALS ?= 15
DATASET ?= synthetic
MODEL ?= csp_lda

.PHONY: setup test lint eval eval-smoke data calibrate live calib-curve

setup:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

eval:
	python -m ssmi.evals.run --task $(TASK) --dataset $(DATASET) --model $(MODEL)

eval-smoke:
	python -m ssmi.evals.run --task mi --dataset synthetic --model csp_lda --check-thresholds
	python -m ssmi.evals.run --task ss --dataset synthetic --model td_features_lda --check-thresholds

data:
	bash scripts/download_data.sh

calibrate:
	python scripts/calibrate.py --task $(TASK) --user $(PID) --device $(DEVICE) --trials $(TRIALS)

live:
	python scripts/live.py --task $(TASK) --user $(PID) --device $(DEVICE)

calib-curve:
	python -m ssmi.evals.calibration_curve --task $(TASK) --dataset $(DATASET) --model $(MODEL)
