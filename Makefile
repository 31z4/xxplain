backend-run:
	python -m backend.main

backend-lint-check:
	ruff check backend

backend-format-check:
	ruff format --check backend

pg-stat-statements-reset:
	docker compose exec postgres \
        psql -U postgres -c "SELECT pg_stat_statements_reset()"

docker-compose-validate:
	docker compose config -q

clean-docker-volumes:
	docker volume rm -f xxplain-postgres-data
	docker volume create xxplain-postgres-data
