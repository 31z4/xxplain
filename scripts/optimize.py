import argparse
import asyncio
import csv
import sys
from time import time

import structlog

from backend.optimize import llm

log = structlog.stdlib.get_logger()


async def _optimize(input: str, output: str):
    data = []

    csv.field_size_limit(sys.maxsize)
    with open(input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            structlog.contextvars.bind_contextvars(path=f"{row['path']}:{row['line']}")
            log.info("Оптимизирую запрос")

            start = time()
            optimized_query = await llm(row["query"])
            optimization_time = time() - start

            row["optimized_query"] = optimized_query
            row["optimization_time"] = optimization_time

            data.append(row)

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
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Использует LLM для оптимизации заданных запросов и записывает результат."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/optimized.csv",
        help="куда сохранять вывод (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/benchdata.csv",
        help="откуда читать запросы (по умолчанию: %(default)s)",
    )

    args = parser.parse_args()

    asyncio.run(_optimize(args.input, args.output))


if __name__ == "__main__":
    main()
