SHELL := /bin/bash

PROJECT_ROOT := $(abspath .)
PYTHON ?= python3.11
VENV ?= $(PROJECT_ROOT)/.venv
PYTHON_BIN := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest

BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8001
BACKEND_RELOAD ?= 0
FRONTEND_HOST ?= 0.0.0.0
FRONTEND_PORT ?= 5173
COMPOSE_FRONTEND_PORT ?= 8080
API_UPSTREAM ?= host.docker.internal:$(BACKEND_PORT)

ifeq ($(filter 1 true yes,$(BACKEND_RELOAD)),)
UVICORN_RELOAD_ARGS :=
else
UVICORN_RELOAD_ARGS := --reload --reload-dir backend
endif

BACKEND_INSTALL_STAMP := $(VENV)/.astra-dev-requirements
FRONTEND_INSTALL_STAMP := frontend/node_modules/.astra-dev-dependencies

.DEFAULT_GOAL := help
.PHONY: help env install backend-install frontend-install dev backend frontend \
	test test-backend test-frontend build compose-up compose-down \
	check-backend-port check-frontend-port check-ports

help:
	@printf '%s\n' \
		'make dev              Start the native backend and Vite frontend' \
		'make install          Prepare Python and frontend dependencies' \
		'make backend          Start only the FastAPI backend' \
		'make frontend         Start only the Vite frontend' \
		'make test             Run backend and frontend tests' \
		'make build            Build the frontend production bundle' \
		'make compose-up       Build/start the Docker Nginx frontend' \
		'make compose-down     Stop the Docker Nginx frontend' \
		'' \
		'Override ports, for example: make dev BACKEND_PORT=8000 FRONTEND_PORT=5173' \
		'Enable backend hot reload explicitly: make dev BACKEND_RELOAD=1'

env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		printf '%s\n' 'Created .env from .env.example. Review local model and token settings.'; \
	fi

check-backend-port:
	@$(call check_port,$(BACKEND_PORT),backend)

check-frontend-port:
	@$(call check_port,$(FRONTEND_PORT),frontend)

check-ports: check-backend-port check-frontend-port

$(BACKEND_INSTALL_STAMP): backend/requirements.txt backend/requirements-dev.txt backend/requirements-constraints.txt
	@if [ -f "$@" ]; then \
		$(PYTHON_BIN) -m pip install -r backend/requirements-dev.txt; \
	elif [ -x "$(UVICORN)" ] && [ -x "$(PYTEST)" ]; then \
		printf '%s\n' 'Using existing Python development environment.'; \
	else \
		$(PYTHON) -m venv $(VENV); \
		$(PYTHON_BIN) -m pip install -r backend/requirements-dev.txt; \
	fi
	@touch $@

backend-install: $(BACKEND_INSTALL_STAMP)

$(FRONTEND_INSTALL_STAMP): frontend/package.json frontend/package-lock.json
	@cd frontend && npm ci
	@touch $@

frontend-install: $(FRONTEND_INSTALL_STAMP)

install: env backend-install frontend-install

backend: env backend-install check-backend-port
	@exec env PYTHONPATH=backend ASTRA_CONFIG_PATH=.env $(UVICORN) app.main:app \
		$(UVICORN_RELOAD_ARGS) --host $(BACKEND_HOST) --port $(BACKEND_PORT)

frontend: frontend-install check-frontend-port
	@cd frontend && exec env BACKEND_PORT='$(BACKEND_PORT)' npm run dev -- --host $(FRONTEND_HOST) --port $(FRONTEND_PORT)

dev: env backend-install frontend-install check-ports
	@set -e; \
		$(MAKE) --no-print-directory backend & backend_pid=$$!; \
		$(MAKE) --no-print-directory frontend & frontend_pid=$$!; \
		trap 'kill $$backend_pid $$frontend_pid 2>/dev/null || true' INT TERM EXIT; \
		printf '%s\n' 'Astra development environment started:'; \
		printf '  Backend:  http://127.0.0.1:%s\n' '$(BACKEND_PORT)'; \
		printf '  Frontend: http://127.0.0.1:%s\n' '$(FRONTEND_PORT)'; \
		while kill -0 $$backend_pid 2>/dev/null && kill -0 $$frontend_pid 2>/dev/null; do sleep 1; done; \
		status=0; \
		if ! kill -0 $$backend_pid 2>/dev/null; then wait $$backend_pid || status=$$?; \
		else wait $$frontend_pid || status=$$?; fi; \
		exit $$status

test: test-backend test-frontend

test-backend: backend-install
	@PYTHONPATH=backend $(PYTEST) -q backend/tests

test-frontend: frontend-install
	@cd frontend && npm test

build: frontend-install
	@cd frontend && npm run build

compose-up: env
	@API_UPSTREAM='$(API_UPSTREAM)' FRONTEND_PORT='$(COMPOSE_FRONTEND_PORT)' \
		docker compose up -d --build

compose-down:
	@docker compose down

define check_port
	if lsof -nP -iTCP:$(1) -sTCP:LISTEN >/dev/null 2>&1; then \
		printf '%s\n' 'Error: Astra $(2) port $(1) is already in use.' >&2; \
		lsof -nP -iTCP:$(1) -sTCP:LISTEN >&2; \
		printf '%s\n' 'Choose another port, for example: make dev BACKEND_PORT=18002 FRONTEND_PORT=15173' >&2; \
		exit 1; \
	fi
endef
