# Local commands for the test-bench model, the API, and the Streamlit form.
# Docker builds do not use this file; they run the same Python modules directly.

PYTHON ?= python3
MODEL ?= artifacts/model.joblib
PORT ?= 8000
UI_PORT ?= 8501

.DEFAULT_GOAL := help

.PHONY: help install train test serve ui predict docker-build docker-up docker-down docker-predict

help:
	@echo "Greener Manufacturing"
	@echo "  make install          install the package, tests, and the Streamlit UI"
	@echo "  make train            fit the model and write $(MODEL)"
	@echo "  make test             run pytest"
	@echo "  make serve            API on :$(PORT)  (train first)"
	@echo "  make ui               form on :$(UI_PORT)  (API must already be running)"
	@echo "  make predict          POST examples/known_request.json to the API"
	@echo "  make docker-up        build and start the API and the form"
	@echo "  make docker-down      stop the compose stack"
	@echo "  make docker-predict   health checks, then one prediction"

install:
	$(PYTHON) -m pip install -e ".[dev,ui]"

train:
	$(PYTHON) -m greener.train --output $(MODEL)

test:
	$(PYTHON) -m pytest

serve:
	MODEL_PATH=$(MODEL) $(PYTHON) -m uvicorn greener.api:app --host 0.0.0.0 --port $(PORT)

ui:
	API_URL=http://127.0.0.1:$(PORT) $(PYTHON) -m streamlit run frontend/app.py --server.port $(UI_PORT)

predict:
	curl -fsS http://127.0.0.1:$(PORT)/health
	@echo
	curl -fsS -X POST http://127.0.0.1:$(PORT)/predict \
		-H 'content-type: application/json' \
		--data @examples/known_request.json
	@echo

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-predict:
	curl -fsS http://127.0.0.1:$(PORT)/health
	@echo
	curl -fsS -X POST http://127.0.0.1:$(PORT)/predict \
		-H 'content-type: application/json' \
		--data @examples/known_request.json
	@echo
	curl -fsS http://127.0.0.1:$(UI_PORT)/_stcore/health
	@echo
