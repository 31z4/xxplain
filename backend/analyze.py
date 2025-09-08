import bisect

from .predict import TIME_CLASSES, SIZE_CLASSES

# Для простоты предполагаем стандартный размер блока.
# Для прода лучше читать текущее значение из БД.
_BLOCK_SIZE_BYTES = 8_192
# Ключи плана, соответствующие чтению данных.
_PLAN_READ_BLOCKS = (
    "Shared Hit Blocks",
    "Shared Read Blocks",
    "Local Hit Blocks",
    "Local Read Blocks",
    "Temp Read Blocks",
)


def analyze_plan(plan: dict) -> dict:
    """Суммаризирует фактические метрики запроса, если они доступны."""
    total_time_ms = plan["Plan"].get("Actual Total Time")
    if not total_time_ms:
        return {}

    data_read_bytes = sum(plan["Plan"][i] for i in _PLAN_READ_BLOCKS)
    data_read_bytes += sum(plan["Planning"][i] for i in _PLAN_READ_BLOCKS)
    data_read_bytes *= _BLOCK_SIZE_BYTES

    return {
        "total_time_ms": total_time_ms,
        "total_time_class": bisect.bisect_right(TIME_CLASSES, total_time_ms),
        "data_read_bytes": data_read_bytes,
        "data_read_class": bisect.bisect_right(SIZE_CLASSES, data_read_bytes)
    }
