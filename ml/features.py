import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Union


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


def get_time_from_plan(plan: Dict[str, Any]) -> int:
    return plan["Plan"].get("Actual Total Time", 0)


def get_size_from_plan(plan: Dict[str, Any]) -> int:
    data_read_bytes = sum(plan["Plan"][i] for i in _PLAN_READ_BLOCKS)
    data_read_bytes += sum(plan["Planning"][i] for i in _PLAN_READ_BLOCKS)
    data_read_bytes *= _BLOCK_SIZE_BYTES

    return data_read_bytes


def _get_server_params(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
    settings = plan_dict.get("Settings", {})
    if not settings:
        # Попробуем найти в других местах
        settings = plan_dict.get("Query", {}).get("Settings", {})
    
    # Преобразуем значения в числовые, где возможно
    normalized_settings = {}
    for key, value in settings.items():
        normalized_settings[key] = _normalize_setting_value(value)
    
    return normalized_settings


def _normalize_setting_value(value: str) -> Union[float, str]:
    """
    Нормализует значение настройки PostgreSQL
    
    Args:
        value: Строковое значение настройки
        
    Returns:
        Нормализованное значение (число или строка)
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return str(value)
    
    # Убираем единицы измерения и преобразуем в число
    try:
        # Обрабатываем размеры памяти (kB, MB, GB)
        if value.endswith("kB"):
            return float(value[:-2].strip())
        elif value.endswith("MB"):
            return float(value[:-2].strip()) * 1024
        elif value.endswith("GB"):
            return float(value[:-2].strip()) * 1024 * 1024
        
        # Обрабатываем булевые значения
        if value.lower() in ("on", "true", "yes"):
            return 1.0
        elif value.lower() in ("off", "false", "no"):
            return 0.0
        
        # Пробуем преобразовать в число
        return float(value)
    except (ValueError, AttributeError):
        # Если не удалось преобразовать, возвращаем как строку
        return value


def extract_features(
    plan_json: Dict[str, Any],
    query_text: str = "",
) -> Dict[str, float]:
    if isinstance(plan_json, list) and len(plan_json) > 0:
        first_item = plan_json[0] # type: ignore
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
    ## initialize keys
    for node_key in ACTIVE_NODE_TYPES:
        features[f"count_node_{node_key}"] += 0
    ## initialize keys
    for join_key in ACTIVE_JOIN_TYPES:
        features[f"count_join_{join_key}"] += 0
    _initialize_features(features)
    _traverse_plan(plan_root, features, depth=0)
    _extract_root_features(plan_root, features)
    _compute_derived_features(features)

    if query_text:
        _extract_query_features(query_text, features)

    if server_params:
        _extract_server_params(server_params, features)

    _normalize_features(features)

    if FEATURE_PREFIX:
        features = {f"{FEATURE_PREFIX}{k}": v for k, v in features.items()}

    return dict(features)

def _initialize_features(features: defaultdict) -> None:
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

    if node_key in ACTIVE_NODE_TYPES:
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
    if join_type:
        join_key = _sanitize_key(join_type)
        if join_type in ACTIVE_JOIN_TYPES:
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

def _extract_query_features(query_text: str, features: defaultdict) -> None:
    features["query_length"] = len(query_text)
    features["query_tokens"] = len(query_text.split())

def _extract_server_params(server_params: Dict[str, Any], features: defaultdict) -> None:
    for key, value in server_params.items():
        if isinstance(value, (int, float)):
            features[f"pgconf_{_sanitize_key(key)}"] = float(value)

def _normalize_features(features: defaultdict) -> None:
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: features.py lab/plan.json | jq")
        sys.exit(0)

    test_plan = json.load(open(sys.argv[1], 'r'))
    print(get_time_from_plan(test_plan))
    print(get_size_from_plan(test_plan))
    print(json.dumps(extract_features(test_plan)))
