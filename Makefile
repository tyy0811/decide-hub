.PHONY: test eval serve automate install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

eval:
	python -m src.evaluation.run

serve:
	uvicorn src.serving.app:app --reload --port 8000

automate:
	python -m src.automations.run
