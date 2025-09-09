import bisect
from typing import Any

from ml.loader import predict as ml_predict

COST_CLASSES = (
    101,
    10_001,
    1_000_001,
    100_000_001,
)

TIME_CLASSES = (
    3_001,
    15_001,
    120_001,
    900_001,
)

SIZE_CLASSES = (
    10_000_001,
    100_000_001,
    1_000_000_001,
    10_000_000_001,
)


def predict(sql: str, plan: dict[str, Any]) -> dict:
    """Предсказывает метрики по плану запроса."""
    cost = plan["Plan"]["Total Cost"]
    cost_class = bisect.bisect_right(COST_CLASSES, cost) + 1

    total_time_ms = ml_predict("catboost", "time_models", sql, plan)
    total_time_class = bisect.bisect_right(TIME_CLASSES, total_time_ms) + 1

    data_read_bytes = ml_predict("catboost", "size_models", sql, plan)
    data_read_class = bisect.bisect_right(SIZE_CLASSES, data_read_bytes) + 1

    return {
        "cost": cost,
        "cost_class": cost_class,
        "total_time_ms": total_time_ms,
        "total_time_class": total_time_class,
        "data_read_bytes": data_read_bytes,
        "data_read_class": data_read_class,
    }
