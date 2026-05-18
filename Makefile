.PHONY: help up down build logs ps shell bootstrap test eval

COMPOSE := docker compose -f docker/docker-compose.yaml

help:
	@echo 'Targets:'
	@echo '  build       Build api/ui/migrate image (slow first time)'
	@echo '  up          Start postgres + migrate + api + ui (detached)'
	@echo '  down        Stop all services (volumes retained)'
	@echo '  down-clean  Stop AND drop volumes (data loss!)'
	@echo '  ps          Container status'
	@echo '  logs        Tail logs (all services)'
	@echo '  logs-api    Tail api logs only'
	@echo '  shell       Open a shell in the api container'
	@echo '  bootstrap   One-time data load: import vacancies, embed, parse CVs'
	@echo '  seed-hf     Pre-populate ats_hf_cache from ~/.cache/huggingface'

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

down-clean:
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=50

logs-api:
	$(COMPOSE) logs -f --tail=100 api

shell:
	$(COMPOSE) exec api bash

bootstrap:
	$(COMPOSE) exec api python -m ats.utils.import_vacancies
	$(COMPOSE) exec -e HF_HUB_OFFLINE=1 api python -m ats.utils.embed_vacancies
	$(COMPOSE) exec -e HF_HUB_OFFLINE=1 api python -m ats.ingestion.parser data/cvs/

seed-hf:
	docker volume create ats_hf_cache
	docker run --rm \
	  -v ats_hf_cache:/cache \
	  -v $$HOME/.cache/huggingface:/host:ro \
	  alpine sh -c "cp -r /host/. /cache/ && du -sh /cache"
