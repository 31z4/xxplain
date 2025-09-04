import os
import json
import psycopg
import csv
from collections import defaultdict

from dotenv import load_dotenv
from pathlib import Path

# Specify the path to your custom .env file
dotenv_path = Path(".env.postgres")
load_dotenv(dotenv_path=dotenv_path)


def get_conn(dsn: str):
    return psycopg.connect(dsn)


def run_explain_analyze_json(cur, query: str) -> dict:
    # FORMAT JSON возвращает одну строку с JSON-массивом
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING ON, FORMAT JSON) {query}")
    row = cur.fetchone()
    plan_json = row[0]
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    return plan_json  # list with dict, plan_json[0]["Plan"], etc.


def read_queries(path):
    """
    Читает SQL-запросы из файла или всех .sql файлов в директории.
    Возвращает список строк (запросов).
    """
    queries = []
    if os.path.isfile(path):
        # Если это файл, читаем его
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            queries = [q for q in lines if q and not q.startswith("--")]
    elif os.path.isdir(path):
        # Если это директория, читаем все .sql файлы
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".sql"):
                file_path = os.path.join(path, filename)
                queries.extend(read_queries(file_path))
    else:
        raise ValueError(f"Путь {path} не является файлом или директорией.")
    return queries


def collect_query_plans(conn, queries: list[str], timeout):
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout TO '{timeout}s'")
        for q in queries:
            print('.', end='', flush=True)
            try:
                plan_json = run_explain_analyze_json(cur, q)
                obj = plan_json[0]

                yield {
                    "query": q,
                    "plan": json.dumps(obj),
                }
                conn.commit()  # Фиксируем транзакцию после успешного выполнения
            except psycopg.errors.QueryCanceled:
                # Можно логировать текст ошибки
                print('T', end='', flush=True)
                conn.rollback()
                continue
            except psycopg.errors.SyntaxError:
                print('S', end='', flush=True)
                conn.rollback()
                continue
            except Exception:
                print('E', end='', flush=True)
                conn.rollback()
                continue

if __name__ == "__main__":
    DSN = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )

    TRAIN_SQL_FILE = "benchmarks/tpc-h/generated/"
    TEST_SQL_FOLDER = "benchmarks/tpc-h/queries/"

    conn = get_conn(DSN)
    train_q = read_queries(TRAIN_SQL_FILE)
    csv_filename = "datasets/train_query_plans.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["query", "plan"], delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        writer.writeheader()
        for row in collect_query_plans(conn, train_q, timeout=20):
            writer.writerow(row)

    test_q = read_queries(TEST_SQL_FOLDER)
    csv_filename = "datasets/test_query_plans.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["query", "plan"], delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None
        )
        writer.writeheader()
        for row in collect_query_plans(conn, test_q, timeout=100):
            writer.writerow(row)

    conn.close()
