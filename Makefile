# NanoGPT-Lab workflows. With Docker (recommended on Windows/ARM) use docker-*.
.DEFAULT_GOAL := help
IMAGE := nanogpt-lab:latest
CONFIG ?= config/smoke.yaml

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup:  ## Install deps + package (host Python)
	python -m pip install -U pip
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

.PHONY: train
train:  ## Train ($(CONFIG)); override with `make train CONFIG=config/full.yaml`
	python -m nanogpt_lab.train --config $(CONFIG)

.PHONY: generate
generate:  ## Generate text from the latest checkpoint
	python -m nanogpt_lab.model.generate --config $(CONFIG)

.PHONY: ablate
ablate:  ## Run the architecture ablation study
	python -m nanogpt_lab.ablation --config $(CONFIG)

.PHONY: test
test:  ## Run tests (excluding slow)
	pytest -m "not slow" --cov --cov-report=term-missing

.PHONY: test-all
test-all:  ## Run all tests including slow (tiny training)
	pytest --cov --cov-report=term-missing

.PHONY: lint
lint:  ## ruff + black --check + mypy
	ruff check src tests
	black --check src tests
	mypy

.PHONY: format
format:  ## Auto-format
	ruff check --fix src tests
	black src tests

# ---- Docker (no host Python needed) --------------------------------------
.PHONY: docker-build
docker-build:  ## Build the image
	docker build -t $(IMAGE) .

.PHONY: docker-train
docker-train: docker-build  ## Train inside Docker
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) python -m nanogpt_lab.train --config $(CONFIG)
