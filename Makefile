# All targets run through the project venv. `make install` creates it.
# Override the base interpreter: make install BASE_PYTHON=python3.12
BASE_PYTHON ?= python3.11
VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: test install db-reset eval serve docker-up docker-down e2e

$(VENV)/bin/python:
	$(BASE_PYTHON) -m venv $(VENV)

install: $(VENV)/bin/python
	$(PYTHON) -m pip install -e ".[dev]"

test: $(VENV)/bin/python
	$(PYTHON) -m pytest tests/ -v --ignore=tests/e2e

eval: $(VENV)/bin/python
	$(PYTHON) -m src.evaluation.run

serve: $(VENV)/bin/python
	$(PYTHON) -m uvicorn src.serving.app:app --reload --port 8000

e2e:
	cd operator_ui && npx playwright test

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

db-reset:
	docker compose down -v && docker compose up -d postgres
