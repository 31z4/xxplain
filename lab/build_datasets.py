import os
import json
import psycopg2
import csv
from collections import defaultdict

from dotenv import load_dotenv
from pathlib import Path

# Specify the path to your custom .env file
dotenv_path = Path(".env.postgres")
load_dotenv(dotenv_path=dotenv_path)


def get_conn(dsn: str):
    return psycopg2.connect(dsn)


def run_explain_analyze_json(cur, query: str) -> dict:
    # FORMAT JSON возвращает одну строку с JSON-массивом
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING ON, FORMAT JSON) {query}")
    row = cur.fetchone()
    plan_json = row[0]
    if isinstance(plan_json, str):
        plan_json = json.loads(plan_json)
    return plan_json  # list with dict, plan_json[0]["Plan"], etc.


def get_server_params(cur) -> dict:
    # Несколько параметров, которые влияют на cost/time
    params = {}
    keys = [
        "random_page_cost",
        "seq_page_cost",
        "cpu_tuple_cost",
        "cpu_index_tuple_cost",
        "cpu_operator_cost",
        "effective_cache_size",
        "work_mem",
        "max_parallel_workers_per_gather",
        "jit",
    ]
    for k in keys:
        cur.execute(f"SHOW {k}")
        v = cur.fetchone()[0]
        # Пробуем привести к числу, иначе оставим строкой (например, 'on'/'off')
        try:
            vnum = float(v.replace("kB", "").replace("MB", "").strip())
            params[k] = vnum
        except Exception:
            params[k] = (
                1.0
                if v.lower() in ("on", "true")
                else 0.0
                if v.lower() in ("off", "false")
                else v
            )
    return params


# -----------------------------
# Feature extraction from EXPLAIN (FORMAT JSON)
# Only "estimated" fields — no leakage
# -----------------------------


def sanitize_key(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("-", "_")


def traverse_plan(node: dict, agg: dict, depth: int = 0):
    node_type = node.get("Node Type", "Unknown")
    node_key = sanitize_key(node_type)
    agg["n_nodes"] += 1
    agg["max_depth"] = max(agg["max_depth"], depth)

    # Counts by node type
    agg[f"count_node_{node_key}"] = agg.get(f"count_node_{node_key}", 0) + 1

    # Estimated fields
    startup_cost = node.get("Startup Cost", 0.0) or 0.0
    total_cost = node.get("Total Cost", 0.0) or 0.0
    plan_rows = node.get("Plan Rows", 0) or 0
    plan_width = node.get("Plan Width", 0) or 0

    agg["sum_est_startup_cost"] += startup_cost
    agg["sum_est_total_cost"] += total_cost
    agg["sum_plan_rows"] += plan_rows
    agg["sum_plan_width"] += plan_width

    agg["max_est_total_cost"] = max(agg["max_est_total_cost"], total_cost)
    agg["max_plan_rows"] = max(agg["max_plan_rows"], plan_rows)
    agg["max_plan_width"] = max(agg["max_plan_width"], plan_width)

    # Common flags/counters
    if node.get("Parallel Aware", False):
        agg["n_parallel_aware"] += 1

    join_type = node.get("Join Type")
    if join_type:
        agg[f"count_join_{sanitize_key(join_type)}"] = (
            agg.get(f"count_join_{sanitize_key(join_type)}", 0) + 1
        )

    if "Sort Key" in node or node_type == "Sort":
        agg["n_sorts"] += 1

    if node_type in ("Aggregate", "HashAggregate", "GroupAggregate"):
        agg["n_aggregates"] += 1

    if node_type in ("WindowAgg",):
        agg["n_window"] += 1

    if node_type in ("Gather", "Gather Merge"):
        agg["n_gather"] += 1
        agg["sum_workers_planned"] += node.get("Workers Planned", 0) or 0

    if node_type in ("Limit",):
        agg["has_limit"] = 1

    if "Filter" in node:
        agg["n_filters"] += 1
        # длина текстового фильтра как суррогат сложности предикатов
        try:
            agg["sum_filter_len"] += len(str(node.get("Filter", "")))
        except Exception:
            pass

    # children
    for ch in node.get("Plans", []) or []:
        traverse_plan(ch, agg, depth + 1)


def extract_features(plan_root: dict, query_text: str, server_params: dict) -> dict:
    # plan_root = obj["Plan"]
    feats = defaultdict(float)
    feats["n_nodes"] = 0.0
    feats["max_depth"] = 0.0
    feats["sum_est_startup_cost"] = 0.0
    feats["sum_est_total_cost"] = 0.0
    feats["sum_plan_rows"] = 0.0
    feats["sum_plan_width"] = 0.0
    feats["max_est_total_cost"] = 0.0
    feats["max_plan_rows"] = 0.0
    feats["max_plan_width"] = 0.0
    feats["n_parallel_aware"] = 0.0
    feats["n_sorts"] = 0.0
    feats["n_aggregates"] = 0.0
    feats["n_window"] = 0.0
    feats["n_gather"] = 0.0
    feats["sum_workers_planned"] = 0.0
    feats["has_limit"] = 0.0
    feats["n_filters"] = 0.0
    feats["sum_filter_len"] = 0.0

    traverse_plan(plan_root, feats, depth=0)

    # Root-level estimate features
    feats["root_total_cost"] = plan_root.get("Total Cost", 0.0) or 0.0
    feats["root_startup_cost"] = plan_root.get("Startup Cost", 0.0) or 0.0
    feats["root_plan_rows"] = plan_root.get("Plan Rows", 0.0) or 0.0
    feats["root_plan_width"] = plan_root.get("Plan Width", 0.0) or 0.0
    feats["ratio_startup_total_cost"] = feats["root_startup_cost"] / (
        feats["root_total_cost"] + 1e-9
    )

    # Normalized by number of nodes
    if feats["n_nodes"] > 0:
        feats["avg_est_total_cost_per_node"] = (
            feats["sum_est_total_cost"] / feats["n_nodes"]
        )
        feats["avg_plan_rows_per_node"] = feats["sum_plan_rows"] / feats["n_nodes"]
        feats["avg_plan_width_per_node"] = feats["sum_plan_width"] / feats["n_nodes"]

    # Query-level simple signals
    feats["query_length"] = len(query_text)
    feats["query_tokens"] = len(query_text.split())

    # Server params (as numeric features)
    for k, v in server_params.items():
        if isinstance(v, (int, float)):
            feats[f"pgconf_{sanitize_key(k)}"] = float(v)
        else:
            # leave string params out or encode basic on/off already handled
            pass

    return dict(feats)


# -----------------------------
# Dataset building
# -----------------------------


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
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip().replace("\n", " ")
                    if content and not content.startswith(
                        "--"
                    ):  # Проверка на пустой или комментарий
                        queries.append(content)
    else:
        raise ValueError(f"Путь {path} не является файлом или директорией.")
    return queries


def build_dataset(conn, queries: list[str], timeout):
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout TO '{timeout}s'")
        server_params = get_server_params(cur)
        for q in queries:
            try:
                plan_json = run_explain_analyze_json(cur, q)
                obj = plan_json[0]
                plan_root = obj["Plan"]

                feats = extract_features(plan_root, q, server_params)
                yield {
                    "feats": feats,
                    "query": q,
                    "plan": obj,
                    "target": float(plan_root.get("Actual Total Time", None)),
                }
            except Exception as e:
                # Можно логировать текст ошибки
                print("EXPLAIN failed:", e)
                conn.rollback()
                continue


if __name__ == "__main__":
    DSN = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )

    TRAIN_SQL_FILE = "benchmarks/tpc-h/generated/samples.sql"
    TEST_SQL_FOLDER = "benchmarks/tpc-h/queries/"

    # Используем функцию read_queries для чтения всех SQL файлов
    train_q = read_queries(TRAIN_SQL_FILE)
    test_q = read_queries(TEST_SQL_FOLDER)

    conn = get_conn(DSN)
    csv_filename = "lab/train.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["query", "plan", "feats", "target"], delimiter="\t"
        )
        writer.writeheader()
        for row in build_dataset(conn, train_q, timeout=20):
            print(".", end="")
            writer.writerow(row)

    csv_filename = "lab/test.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["query", "plan", "feats", "target"], delimiter="\t"
        )
        writer.writeheader()
        for row in build_dataset(conn, test_q, timeout=100):
            print(".", end="")
            writer.writerow(row)

    conn.close()
