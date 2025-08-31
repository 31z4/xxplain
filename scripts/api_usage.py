"""
Пример базового использования xxplain API
"""

import requests
import json

# Конфигурация
API_BASE_URL = "http://localhost:8000"

def main():
    print("=== Пример использования xxplain API ===\n")
    test_query = "SELECT COUNT(*) FROM pg_tables"

    # 1. Получение query plan для простого запроса
    print("1. Получение query plan...")
    explain_response = requests.post(
        f"{API_BASE_URL}/explain",
        data=test_query,
        params={"analyze": False}
    )
    print(explain_response.content)

    complex_query = "SELECT     l_returnflag,     l_linestatus,     sum(l_quantity) AS sum_qty,     sum(l_extendedprice) AS sum_base_price,     sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,     sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,     avg(l_quantity) AS avg_qty,     avg(l_extendedprice) AS avg_price,     avg(l_discount) AS avg_disc,     count(*) AS count_order FROM     lineitem WHERE     l_shipdate <= CAST('1998-09-02' AS date) GROUP BY     l_returnflag,     l_linestatus ORDER BY     l_returnflag,     l_linestatus"
    print("2. Получение предсказания времени запроса")
    predict_response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "query_text": complex_query,
            "model_name": "RandomForest"
        }
    )
    print(predict_response.content)


if __name__ == "__main__":
    main()