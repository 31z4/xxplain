import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import sqlglot
from sqlglot import exp

# Constants from old features
ACTIVE_NODE_TYPES = [
    "Seq_Scan", "Index_Scan", "Index_Only_Scan", "Bitmap_Index_Scan",
    "Bitmap_Heap_Scan", "Nested_Loop", "Hash_Join", "Merge_Join",
    "Hash", "Sort", "Aggregate", "Limit", "Gather", "Gather_Merge",
    "Materialize", "Memoize", "Result", "Unique", "BitmapAnd",
    "Subquery_Scan", "WindowAgg", "Incremental_Sort"
]
_PLAN_READ_BLOCKS = (
    "Shared Hit Blocks",
    "Shared Read Blocks",
    "Local Hit Blocks",
    "Local Read Blocks",
    "Temp Read Blocks",
)
_BLOCK_SIZE_BYTES = 8_192
ACTIVE_JOIN_TYPES = ["Inner", "Left", "Right", "Full", "Semi", "Anti"]
MIN_DEPTH = 0
MAX_DEPTH = None
FEATURE_PREFIX = ""


def get_time_from_plan(plan: Union[Dict[str, Any], List[Any]]) -> float:
    if isinstance(plan, list):
        if len(plan) > 0 and isinstance(plan[0], dict):
            plan = plan[0]
        else:
            return 0.0
    return plan["Plan"].get("Actual Total Time", 0.0)


def get_size_from_plan(plan: Union[Dict[str, Any], List[Any]]) -> int:
    if isinstance(plan, list):
        if len(plan) > 0 and isinstance(plan[0], dict):
            plan = plan[0]
        else:
            return 0
    data_read_bytes = sum(plan["Plan"][i] for i in _PLAN_READ_BLOCKS if i in plan["Plan"])
    data_read_bytes += sum(plan["Planning"][i] for i in _PLAN_READ_BLOCKS if i in plan["Planning"])
    data_read_bytes *= _BLOCK_SIZE_BYTES
    return data_read_bytes


def _get_server_params(plan_dict: Union[Dict[str, Any], List[Any]]) -> Dict[str, Any]:
    if isinstance(plan_dict, list):
        if len(plan_dict) > 0 and isinstance(plan_dict[0], dict):
            plan_dict = plan_dict[0]
        else:
            return {}
    settings = plan_dict.get("Settings", {})
    if not settings:
        settings = plan_dict.get("Query", {}).get("Settings", {})

    normalized_settings = {}
    for key, value in settings.items():
        normalized_settings[key] = _normalize_setting_value(value)

    return normalized_settings
def _normalize_setting_value(value: str) -> Union[float, str]:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return str(value)
    if value.endswith("kB"):
        return float(value[:-2].strip())
    elif value.endswith("MB"):
        return float(value[:-2].strip()) * 1024
    elif value.endswith("GB"):
        return float(value[:-2].strip()) * 1024 * 1024
    elif value.lower() in ("on", "true", "yes"):
        return 1.0
    elif value.lower() in ("off", "false", "no"):
        return 0.0
    try:
        return float(value)
    except (ValueError, AttributeError):
        return value


def _initialize_features_plan(features: defaultdict) -> None:
    features["n_nodes"] = 0.0
    features["max_depth"] = 0.0
    features["sum_est_startup_cost"] = 0.0
    features["sum_est_total_cost"] = 0.0
    features["sum_plan_rows"] = 0.0
    features["sum_plan_width"] = 0.0
    features["max_est_total_cost"] = 0.0
    features["max_plan_rows"] = 0.0
    features["max_plan_width"] = 0.0
    features["n_parallel_aware"] = 0.0
    features["n_gather"] = 0.0
    features["sum_workers_planned"] = 0.0
    features["n_sorts"] = 0.0
    features["n_aggregates"] = 0.0
    features["n_window"] = 0.0
    features["has_limit"] = 0.0
    features["n_filters"] = 0.0
    features["sum_filter_len"] = 0.0


def _traverse_plan(node: Dict[str, Any], features: defaultdict, depth: int = 0) -> None:
    if MAX_DEPTH is not None and depth > MAX_DEPTH:
        return
    if depth < MIN_DEPTH:
        return

    node_type = node.get("Node Type", "Unknown")
    node_key = _sanitize_key(node_type)

    features["n_nodes"] += 1
    features["max_depth"] = max(features["max_depth"], depth)

    if node_key in [n.lower().replace(" ", "_") for n in ACTIVE_NODE_TYPES]:
        features[f"count_node_{node_key}"] += 1

    startup_cost = node.get("Startup Cost", 0.0) or 0.0
    total_cost = node.get("Total Cost", 0.0) or 0.0
    plan_rows = node.get("Plan Rows", 0) or 0
    plan_width = node.get("Plan Width", 0) or 0

    features["sum_est_startup_cost"] += startup_cost
    features["sum_est_total_cost"] += total_cost
    features["sum_plan_rows"] += plan_rows
    features["sum_plan_width"] += plan_width

    features["max_est_total_cost"] = max(features["max_est_total_cost"], total_cost)
    features["max_plan_rows"] = max(features["max_plan_rows"], plan_rows)
    features["max_plan_width"] = max(features["max_plan_width"], plan_width)

    join_type = node.get("Join Type")
    if join_type and join_type in ACTIVE_JOIN_TYPES:
        join_key = _sanitize_key(join_type)
        features[f"count_join_{join_key}"] += 1

    if node.get("Parallel Aware", False):
        features["n_parallel_aware"] += 1

    if node_type in ("Gather", "Gather Merge"):
        features["n_gather"] += 1
        features["sum_workers_planned"] += node.get("Workers Planned", 0) or 0

    if "Sort Key" in node or node_type == "Sort":
        features["n_sorts"] += 1

    if node_type in ("Aggregate", "HashAggregate", "GroupAggregate"):
        features["n_aggregates"] += 1

    if node_type in ("WindowAgg",):
        features["n_window"] += 1

    if node_type in ("Limit",):
        features["has_limit"] = 1

    if "Filter" in node:
        features["n_filters"] += 1
        try:
            features["sum_filter_len"] += len(str(node.get("Filter", "")))
        except Exception:
            pass

    for child in node.get("Plans", []) or []:
        _traverse_plan(child, features, depth + 1)


def _extract_root_features(plan_root: Dict[str, Any], features: defaultdict) -> None:
    features["root_total_cost"] = plan_root.get("Total Cost", 0.0) or 0.0
    features["root_startup_cost"] = plan_root.get("Startup Cost", 0.0) or 0.0
    features["root_plan_rows"] = plan_root.get("Plan Rows", 0.0) or 0.0
    features["root_plan_width"] = plan_root.get("Plan Width", 0.0) or 0.0


def _compute_derived_features(features: defaultdict) -> None:
    features["ratio_startup_total_cost"] = (
        features["root_startup_cost"] / (features["root_total_cost"] + 1e-9)
    )

    if features["n_nodes"] > 0:
        features["avg_est_total_cost_per_node"] = (
            features["sum_est_total_cost"] / features["n_nodes"]
        )
        features["avg_plan_rows_per_node"] = features["sum_plan_rows"] / features["n_nodes"]
        features["avg_plan_width_per_node"] = features["sum_plan_width"] / features["n_nodes"]


def _extract_query_features_plan(query_text: str, features: defaultdict) -> None:
    features["query_length"] = len(query_text)
    features["query_tokens"] = len(query_text.split())


def _extract_server_params_plan(server_params: Dict[str, Any], features: defaultdict) -> None:
    for key, value in server_params.items():
        if isinstance(value, (int, float)):
            features[f"pgconf_{_sanitize_key(key)}"] = float(value)


def _normalize_features_plan(features: defaultdict) -> None:
    if features["max_est_total_cost"] > 0:
        cost_features = ["sum_est_startup_cost", "sum_est_total_cost", "root_total_cost", "root_startup_cost"]
        max_cost = features["max_est_total_cost"]
        for feature in cost_features:
            if feature in features:
                features[f"{feature}_normalized"] = features[feature] / max_cost

    if features["max_plan_rows"] > 0:
        row_features = ["sum_plan_rows", "root_plan_rows"]
        max_rows = features["max_plan_rows"]
        for feature in row_features:
            if feature in features:
                features[f"{feature}_normalized"] = features[feature] / max_rows


def _sanitize_key(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").replace("-", "_")


def extract_features_from_plan(plan_json: Union[Dict[str, Any], List[Any]], query_text: str = "") -> Dict[str, float]:
    if isinstance(plan_json, list) and len(plan_json) > 0:
        first_item = plan_json[0]
        if isinstance(first_item, dict):
            plan_root = first_item.get("Plan", {})
        else:
            raise ValueError("Неверный формат элемента в списке plan_json")
    elif isinstance(plan_json, dict):
        plan_root = plan_json.get("Plan", plan_json)
    else:
        raise ValueError("Неверный формат plan_json")

    features = defaultdict(float)
    server_params = _get_server_params(plan_json)
    for node_key in ACTIVE_NODE_TYPES:
        features[f"count_node_{_sanitize_key(node_key)}"] += 0
    for join_key in ACTIVE_JOIN_TYPES:
        features[f"count_join_{_sanitize_key(join_key)}"] += 0
    _initialize_features_plan(features)
    _traverse_plan(plan_root, features, depth=0)
    _extract_root_features(plan_root, features)
    _compute_derived_features(features)

    if query_text:
        _extract_query_features_plan(query_text, features)

    if server_params:
        _extract_server_params_plan(server_params, features)

    _normalize_features_plan(features)

    if FEATURE_PREFIX:
        features = {f"{FEATURE_PREFIX}{k}": v for k, v in features.items()}

    return dict(features)


def _extract_features_from_sql(features, sql):
    try:
        query_tree = sqlglot.parse_one(sql)
    except Exception:
        return

    features['n_tables'] = len(list(query_tree.find_all(exp.Table)))
    features['n_joins'] = len(list(query_tree.find_all(exp.Join)))
    features['has_subquery'] = 1.0 if query_tree.find(exp.Subquery) else 0.0
    features['has_cte'] = 1.0 if query_tree.find(exp.CTE) else 0.0
    features['has_groupby'] = 1.0 if query_tree.find(exp.Group) else 0.0
    features['has_distinct'] = 1.0 if 'DISTINCT' in sql.upper() else 0.0
    features['has_orderby'] = 1.0 if query_tree.find(exp.Order) else 0.0
    features['has_window_fn'] = 1.0 if query_tree.find(exp.Window) else 0.0
    expressions = query_tree.find_all(exp.Expression)
    features['complexity_depth'] = max((node.depth for node in expressions), default=0.0)
    features['selectivity_est'] = 0.0
    joins = list(query_tree.find_all(exp.Join))
    features['has_cartesian'] = 1.0 if any(join.args.get('on') is None for join in joins) else 0.0
    features['is_update_or_delete'] = 1.0 if isinstance(query_tree, (exp.Update, exp.Delete)) else 0.0

def _extract_from_pg_stat_statement(features, data) -> tuple[float, float]:
    calls = data.get('calls', 0)
    mean_exec_time = data.get('mean_exec_time', 0)
    wal_records = data.get('wal_records', 0)
    wal_fpi = data.get('wal_fpi', 0)
    wal_bytes = data.get('wal_bytes', 0)
    stddev_exec_time = data.get('stddev_exec_time', 0)
    total_exec_time = data.get('total_exec_time', 0)
    rows = data.get('rows', 0)
    shared_blks_hit = data.get('shared_blks_hit', 0)
    shared_blks_read = data.get('shared_blks_read', 0)
    shared_blks_written = data.get('shared_blks_written', 0)
    temp_blks_read = data.get('temp_blks_read', 0)
    temp_blks_written = data.get('temp_blks_written', 0)
    stddev_plan_time = data.get('stddev_plan_time', 0)
    mean_plan_time = data.get('mean_plan_time', 0)
    blk_read_time = data.get('blk_read_time', 0)
    blk_write_time = data.get('blk_write_time', 0)
    temp_blk_read_time = data.get('temp_blk_read_time', 0)
    temp_blk_write_time = data.get('temp_blk_write_time', 0)
    jit_generation_time = data.get('jit_generation_time', 0)
    jit_inlining_time = data.get('jit_inlining_time', 0)
    jit_optimization_time = data.get('jit_optimization_time', 0)
    jit_emission_time = data.get('jit_emission_time', 0)

    jit_total_time = jit_generation_time + jit_inlining_time + jit_optimization_time + jit_emission_time
    total_temp_io = temp_blks_read + temp_blks_written
    total_io = shared_blks_read + shared_blks_written + total_temp_io
    total_blks = shared_blks_hit + shared_blks_read
    total_io_time = blk_read_time + blk_write_time + temp_blk_read_time + temp_blk_write_time

    # features['calls'] = calls
    # features['mean_exec_time'] = mean_exec_time
    # features['wal_records'] = wal_records
    # features['wal_fpi'] = wal_fpi
    # features['wal_bytes'] = wal_bytes
    # features['stddev_exec_time'] = stddev_exec_time
    # features['exec_time_per_row'] = total_exec_time / rows if rows else 0
    # features['rows_per_call'] = rows / calls if calls else 0
    # features['cache_hit_ratio'] = shared_blks_hit / total_blks if total_blks else 0
    # features['io_intensity'] = (shared_blks_read + shared_blks_written) / total_exec_time if total_exec_time else 0
    # features['total_io_time'] = total_io_time
    # features['io_time_ratio'] = total_io_time / total_exec_time if total_exec_time else 0
    # features['temp_io_ratio'] = total_temp_io / total_io if total_io else 0
    # features['temp_heaviness'] = total_temp_io / calls if calls else 0
    # features['plan_stability'] = stddev_plan_time / mean_plan_time if mean_plan_time else 0
    # features['exec_stability'] = stddev_exec_time / mean_exec_time if mean_exec_time else 0
    # features['plan_exec_ratio'] = mean_plan_time / mean_exec_time if mean_exec_time else 0
    # features['jit_heaviness'] = jit_total_time / total_exec_time if total_exec_time else 0
    # features['jit_time_ratio'] = jit_total_time / total_exec_time if total_exec_time else 0
    # features['temp_disk_usage'] = total_temp_io / calls if calls else 0

    return float(stddev_exec_time), float(total_io)

def _extract_tables_info(features, sql, pg_stat_user_tables, pg_indexes):
    try:
        query_tree = sqlglot.parse_one(sql)
    except Exception:
        return

    # Извлекаем все таблицы из SQL-запроса
    tables = []
    for table in query_tree.find_all(exp.Table):
        name = table.name
        schema = table.args.get('db')
        if schema:
            full_name = f"{schema}.{name}"
        else:
            full_name = name
        tables.append(full_name)

    # Инициализируем суммы
    total_n_live_tup = 0
    total_seq_scan = 0
    total_idx_scan = 0
    total_n_indexes = 0

    # Обрабатываем каждую таблицу
    for table_name in tables:
        # Статистика из pg_stat_user_tables
        if pg_stat_user_tables is not None:
            if '.' in table_name:
                schema, name = table_name.split('.', 1)
                matching_rows = pg_stat_user_tables[(pg_stat_user_tables['schemaname'] == schema) & (pg_stat_user_tables['relname'] == name)]
            else:
                matching_rows = pg_stat_user_tables[pg_stat_user_tables['relname'] == table_name]
            if not matching_rows.empty:
                stat = matching_rows.iloc[0]
                total_n_live_tup += stat.get('n_live_tup', 0)
                total_seq_scan += stat.get('seq_scan', 0)
                total_idx_scan += stat.get('idx_scan', 0)

        # Индексы из pg_indexes
        if pg_indexes is not None:
            if '.' in table_name:
                schema, name = table_name.split('.', 1)
                table_indexes = pg_indexes[(pg_indexes['schemaname'] == schema) & (pg_indexes['tablename'] == name)]
            else:
                table_indexes = pg_indexes[pg_indexes['tablename'] == table_name]
            total_n_indexes += len(table_indexes)

    # Вычисляем производные метрики
    has_index = 1.0 if total_n_indexes > 0 else 0.0
    index_usage_ratio = total_idx_scan / (total_idx_scan + total_seq_scan) if (total_idx_scan + total_seq_scan) > 0 else 0.0
    table_bloat_ratio = 0.0  # Не реализовано, так как требует отдельного запроса

    # Добавляем в features
    features['table_size_rows'] = int(total_n_live_tup)
    features['has_index'] = has_index
    features['n_indexes'] = total_n_indexes
    features['n_seq_scans'] = int(total_seq_scan)
    features['n_idx_scans'] = float(total_idx_scan)
    features['index_usage_ratio'] = index_usage_ratio
    features['table_bloat_ratio'] = table_bloat_ratio


def extract_features(
    query: str = "",
    pg_stat_user_tables: Optional[pd.DataFrame] = None,
    pg_indexes: Optional[pd.DataFrame] = None,
    plan_json: Optional[Union[Dict[str, Any], List[Any]]] = None,
    plan_analyze: Dict = None,
) -> Dict[str, Any]:
    features = defaultdict(float)
    time = 0.0
    size = 0

    # Extract features from SQL query
    _extract_features_from_sql(features, query)

    # Extract features from plan if provided
    if plan_json is not None:
        plan_features = extract_features_from_plan(plan_json, query)
        features.update(plan_features)
        # Get time and size from plan
        time = get_time_from_plan(plan_analyze)
        size = get_size_from_plan(plan_analyze)

    # Extract features from tables/indexes if available
    if pg_stat_user_tables is not None and pg_indexes is not None:
        _extract_tables_info(features, query, pg_stat_user_tables, pg_indexes)

    # Return features with time/size if plan was provided
    if plan_json is not None:
        return {
            'features': json.dumps(dict(features)),
            'time': time,
            'size': size,
        }

    # If no plan, return features dict
    return dict(features)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: features.py benchmarks/tpc-h/queries/q02.sql | jq")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        sql = f.read()
    print(json.dumps(extract_features(sql)))
