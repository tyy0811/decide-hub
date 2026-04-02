# All targets run through the project venv. `make install` creates it.
# Override the base interpreter: make install BASE_PYTHON=python3.12
BASE_PYTHON ?= python3.11
VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: test install db-reset eval

$(VENV)/bin/python:
	$(BASE_PYTHON) -m venv $(VENV)

install: $(VENV)/bin/python
	$(PYTHON) -m pip install -e ".[dev]"

test: $(VENV)/bin/python
	$(PYTHON) -m pytest tests/ -v

eval: $(VENV)/bin/python
	$(PYTHON) -m src.evaluation.run

db-reset:
	docker compose down -v && docker compose up -d postgres
