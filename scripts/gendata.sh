#!/bin/bash

set -e

# Создать схему TPC-H.
docker compose exec postgres \
    psql -U postgres -f /benchmarks/tpc-h/schema.sql

# Сгенерировать данные TPC-H (scale factor 1GB).
uv run tpchgen-cli --scale-factor 1 --format csv --output-dir=./data/tpc-h

# Загрузить данные TPC-H.
for t in customer lineitem nation orders part partsupp region supplier; do
    docker compose exec postgres \
        psql -U postgres -c "\copy $t FROM /data/tpc-h/$t.csv WITH (FORMAT csv, HEADER MATCH)"
done

# Создать схему toast.
docker compose exec postgres \
    psql -U postgres -f /benchmarks/toast/schema.sql

# Сгенерировать данные toast.
uv run python scripts/gentoast.py

# Загрузить данные toast.
docker compose exec postgres \
    psql -U postgres -c "\copy toast FROM /data/toast/toast.csv WITH (FORMAT csv, HEADER MATCH)"
