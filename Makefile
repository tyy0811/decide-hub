.PHONY: test install db-reset eval

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

eval:
	python3 -m src.evaluation.run

db-reset:
	docker compose down -v && docker compose up -d postgres
