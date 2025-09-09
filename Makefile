backend-run:
	uv run python -m backend.main

backend-lint-check:
	uv run ruff check backend

backend-format-check:
	uv run ruff format --check backend

pg-stat-statements-reset:
	docker compose exec postgres \
        psql -U postgres -c "SELECT pg_stat_statements_reset()"

docker-compose-validate:
	docker compose config -q

clean-docker-volumes:
	docker volume rm -f xxplain-postgres-data
	docker volume create xxplain-postgres-data

train_models:
	python scripts/b_enrich_dataset.py datasets/train_query_plans.csv > datasets/train_dataset.csv
	python scripts/b_enrich_dataset.py datasets/test_query_plans.csv > datasets/test_dataset.csv
	python scripts/c_train_models.py datasets/train_dataset.csv datasets/test_dataset.csv