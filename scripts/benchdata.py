import argparse
import csv
import json
from datetime import date
from decimal import Decimal
from glob import glob
from pathlib import Path
from time import time
from typing import Generator

import psycopg
import psycopg.adapt
import structlog

from backend.config import settings

log = structlog.stdlib.get_logger()


class _CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Decimal and datetime.date types."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


def _read(path: str) -> dict:
    query = Path(path).read_text(encoding="utf-8")
    return {
        "path": path,
        "line": 1,
        "query": query,
    }


def _read_many(path: str) -> Generator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for n, query in enumerate(f.readlines()):
            yield {
                "path": path,
                "line": n + 1,
                "query": query,
            }


def _exec(sql_glob: list[str], single: bool, output: str, timeout: int) -> None:
    data = []
    for g in sql_glob:
        for path in glob(g):
            if single:
                data.append(_read(path))
            else:
                data.extend(_read_many(path))

    with (
        psycopg.Connection.connect(str(settings.POSTGRES_DSN)) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(f"SET statement_timeout TO '{timeout}s'")
        conn.commit()

        for i, _ in enumerate(data):
            _log = log.bind(path=f"{data[i]['path']}:{data[i]['line']}")
            _log.info("Выполняю запрос")

            sql = data[i]["query"]
            explain_sql = f"EXPLAIN (BUFFERS, SETTINGS, FORMAT JSON) {sql}"
            obj = cur.execute(explain_sql).fetchone()
            data[i]["explain"] = obj[0][0]

            start = time()
            try:
                objs = cur.execute(sql).fetchall()
            except psycopg.errors.QueryCanceled:
                _log.warning("Таймаут запроса")
                data[i]["output"] = None
                data[i]["latency"] = timeout
                conn.rollback()
                continue
            latency = time() - start

            data[i]["output"] = objs
            data[i]["latency"] = latency

    if output:
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                (
                    "path",
                    "line",
                    "query",
                    "explain",
                    "output",
                    "latency",
                )
            )
            for row in data:
                writer.writerow(
                    (
                        row["path"],
                        row["line"],
                        row["query"],
                        json.dumps(row["explain"], cls=_CustomJSONEncoder),
                        json.dumps(row["output"], cls=_CustomJSONEncoder),
                        row["latency"],
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Собирает данные, необходимые для обучения моделей и экспериментов, "
            "выполняя заданные SQL запросы"
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/benchdata.csv",
        help="куда сохранять вывод (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        default=False,
        help="один запрос на файл (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        default=False,
        help="не сохранять вывод, только выполнить запросы (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=30,
        help="таймаут запроса (по умолчанию: %(default)s)",
    )
    parser.add_argument("globs", nargs="+")

    args = parser.parse_args()
    output = "" if args.no_output else args.output

    _exec(args.globs, args.single, output, args.timeout)


if __name__ == "__main__":
    main()
