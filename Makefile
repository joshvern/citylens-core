.PHONY: help venv install install-sam2 test sam2-assets sam2-assets-all

PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

help:
	@echo "make venv            # create venv at ./.venv"
	@echo "make install         # pip install -e .[dev]"
	@echo "make install-sam2    # pip install -e .[dev,sam2]"
	@echo "make test            # run pytest"
	@echo "make sam2-assets      # download SAM2.1 small (default)"
	@echo "make sam2-assets-all  # download SAM2.1 small + large"

$(VENV)/bin/activate:
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3,11), sys.version" \
	  || (echo "Python 3.11+ required. Set PYTHON=python3.11 or PYTHON=/home/josh/.local/bin/python3.11" && exit 2)
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip setuptools wheel

venv: $(VENV)/bin/activate
	@echo "venv ready: $(VENV)"

install: venv
	$(PIP) install -e ".[dev]"

install-sam2: venv
	$(PIP) install -e ".[dev,sam2]"

test: install
	$(PY) -m pytest -q

SAM2_YAML_BASE := https://raw.githubusercontent.com/facebookresearch/sam2/refs/heads/main/sam2/configs/sam2.1
SAM2_CKPT_BASE := https://dl.fbaipublicfiles.com/segment_anything_2/092824

configs/sam2.1:
	mkdir -p configs/sam2.1

weights:
	mkdir -p weights

sam2-assets: configs/sam2.1 weights
	@echo "Downloading SAM2.1 small assets into ./configs and ./weights"
	curl -L -o configs/sam2.1/sam2.1_hiera_s.yaml $(SAM2_YAML_BASE)/sam2.1_hiera_s.yaml
	curl -L -o weights/sam2.1_hiera_small.pt $(SAM2_CKPT_BASE)/sam2.1_hiera_small.pt

sam2-assets-all: configs/sam2.1 weights
	@echo "Downloading SAM2.1 small + large assets into ./configs and ./weights"
	curl -L -o configs/sam2.1/sam2.1_hiera_s.yaml $(SAM2_YAML_BASE)/sam2.1_hiera_s.yaml
	curl -L -o configs/sam2.1/sam2.1_hiera_l.yaml $(SAM2_YAML_BASE)/sam2.1_hiera_l.yaml
	curl -L -o weights/sam2.1_hiera_small.pt $(SAM2_CKPT_BASE)/sam2.1_hiera_small.pt
	curl -L -o weights/sam2.1_hiera_large.pt $(SAM2_CKPT_BASE)/sam2.1_hiera_large.pt
