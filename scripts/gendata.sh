#!/bin/bash

set -e

# Создать схему.
docker compose exec postgres \
    psql -U postgres -f /benchmarks/tpc-h/dss.ddl

# Сгенерировать данные (scale factor 1GB).
uv run tpchgen-cli --scale-factor 1 --format csv --output-dir=./data/tpc-h

# Загрузить данные.
for t in customer lineitem nation orders part partsupp region supplier; do
    docker compose exec postgres \
        psql -U postgres -c "\copy $t FROM /data/tpc-h/$t.csv WITH (FORMAT csv, HEADER MATCH)"
done
