docker-compose-validate:
	docker compose config -q

clean-docker-volumes:
	docker volume rm -f xxplain-postgres-data
	docker volume create xxplain-postgres-data
