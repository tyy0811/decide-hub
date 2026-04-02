.PHONY: test install db-reset

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

db-reset:
	docker compose down -v && docker compose up -d postgres
