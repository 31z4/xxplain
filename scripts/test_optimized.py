import argparse
import csv
import json
import sys
from time import time

import psycopg
import structlog
from deepdiff import DeepDiff

from backend.config import settings
from scripts.benchdata import configure_string_adapters

log = structlog.stdlib.get_logger()


def _test(input: str, output: str, timeout: int) -> None:
    data = []

    with psycopg.Connection.connect(str(settings.POSTGRES_DSN)) as conn:
        configure_string_adapters(conn)

        conn.execute(f"SET statement_timeout TO '{timeout}s'")
        conn.commit()

        csv.field_size_limit(sys.maxsize)
        with open(input, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

                structlog.contextvars.bind_contextvars(
                    path=f"{row['path']}:{row['line']}"
                )
                log.info("Тестирую запрос")

                sql = row["optimized_query"]
                explain_sql = f"EXPLAIN (BUFFERS, SETTINGS, FORMAT JSON) {sql}"

                try:
                    obj = conn.execute(explain_sql).fetchone()
                except psycopg.errors.SyntaxError:
                    log.warning("Невалидный запрос")
                    continue
                row["optimized_explain"] = obj[0][0]

                start = time()
                try:
                    objs = conn.execute(sql).fetchall()
                except psycopg.errors.QueryCanceled:
                    log.warning("Таймаут запроса")
                    row["optimized_latency"] = timeout
                    conn.rollback()
                    continue
                latency = time() - start
                row["optimized_latency"] = latency
                row["optimized_output"] = objs

                explain = json.loads(row["explain"])
                cost = row["optimized_explain"]["Plan"]["Total Cost"]
                cost_ratio = cost / explain["Plan"]["Total Cost"]

                expected = json.loads(row["output"])
                output_match = (
                    DeepDiff(objs, expected, ignore_type_in_groups=[(list, tuple)])
                    == {}
                )
                row["output_match"] = output_match

                latency_ratio = latency / float(row["latency"])
                log.info(
                    "Результат теста",
                    output_match=output_match,
                    cost_ratio=cost_ratio,
                    latency_ratio=latency_ratio,
                )

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
                "optimized_query",
                "optimization_time",
                "optimized_explain",
                "optimized_output",
                "optimized_latency",
                "output_match",
            )
        )
        for row in data:
            writer.writerow(
                (
                    row["path"],
                    row["line"],
                    row["query"],
                    row["explain"],
                    row["output"],
                    row["latency"],
                    row["optimized_query"],
                    row["optimization_time"],
                    json.dumps(row.get("optimized_explain")),
                    json.dumps(row.get("optimized_output")),
                    row.get("optimized_latency"),
                    row.get("output_match"),
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Тестирует оптимизированные запросы, выполняя их, и собирает статистику."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/optimized_test.csv",
        help="куда сохранять вывод (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/optimized.csv",
        help="откуда читать запросы (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=30,
        help="таймаут запроса (по умолчанию: %(default)s)",
    )

    args = parser.parse_args()

    _test(args.input, args.output, args.timeout)


if __name__ == "__main__":
    main()
