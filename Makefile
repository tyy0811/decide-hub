# Requires Python 3.11+ with project deps installed (pip install -e ".[dev]").
# Set PYTHON to whichever interpreter has the deps. Examples:
#   make test PYTHON=python3.11
#   make test PYTHON=/usr/local/opt/python@3.11/bin/python3.11
#   make test PYTHON=.venv/bin/python
PYTHON ?= python3

.PHONY: test install db-reset eval

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) -m src.evaluation.run

db-reset:
	docker compose down -v && docker compose up -d postgres
