TASK ?= mi
DATASET ?= synthetic
MODEL ?= csp_lda

.PHONY: setup test lint eval eval-smoke data

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
