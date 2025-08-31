import bisect

COST_CLASSES = (
    101,
    10_001,
    1_000_001,
    100_000_001,
)


def predict(plan: dict) -> dict:
    cost = plan["Plan"]["Total Cost"]
    cost_class = bisect.bisect_right(COST_CLASSES, cost) + 1

    return {
        "cost": cost,
        "cost_class": cost_class,
    }
