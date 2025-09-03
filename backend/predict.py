import bisect

COST_CLASSES = (
    101,
    10_001,
    1_000_001,
    100_000_001,
)


def predict(plan: dict) -> dict:
    """Предсказывает метрики по плану запроса."""
    cost = plan["Plan"]["Total Cost"]
    cost_class = bisect.bisect_right(COST_CLASSES, cost) + 1

    return {
        "cost": cost,
        "cost_class": cost_class,
        "total_time_ms": 0,  # TODO: Здесь будет настоящее предсказанное время.
        "total_time_class": 0,  # TODO: Здесь будет настоящий предсказанный класс.
        "data_read_bytes": 0,  # TODO: Здесь будет настоящии предсказанные данные.
        "data_read_class": 0,  # TODO: Здесь будет настоящий класс.
    }
