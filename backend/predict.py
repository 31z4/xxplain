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
    101,
    1_001,
    5_001,
    100_001,
)


def predict(sql: str, plan: dict[str, Any]) -> dict:
    """Предсказывает метрики по плану запроса."""
    cost = plan["Plan"]["Total Cost"]
    cost_class = bisect.bisect_right(COST_CLASSES, cost) + 1
    total_time_ms = ml_predict("randomforest", sql, plan)
    total_time_class = bisect.bisect_right(TIME_CLASSES, total_time_ms)

    return {
        "cost": cost,
        "cost_class": cost_class,
        "total_time_ms": total_time_ms,
        "total_time_class": total_time_class,
        "data_read_bytes": 0,  # TODO: Здесь будет настоящии предсказанные данные.
        "data_read_class": 0,  # TODO: Здесь будет настоящий класс.
    }
