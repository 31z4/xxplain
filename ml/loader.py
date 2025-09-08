import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import extract_features


class ModelLoadException(Exception):
    pass


def load_model(model_name: str, model_dir: Path):
    model_path = model_dir / f"{model_name.lower()}_model.pkl"
    scaler_path = model_dir / "scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Файл скейлера не найден: {scaler_path}")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model_name, target, sql, plan):
    model, scaler = load_model(model_name, Path(target))
    features = extract_features(plan, sql)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    y_pred_log = model.predict(features_scaled)
    # Обратное преобразование из логарифмического масштаба
    y_pred = np.expm1(y_pred_log)
    # Ограничиваем снизу нулем
    y_pred = np.maximum(y_pred, 0)
    return float(y_pred[0]) if len(y_pred) == 1 else y_pred


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: loader.py benchmarks/tpc-h/queries/q01.sql")
        sys.exit(0)

    import psycopg
    from dotenv import load_dotenv

    from ml.features import get_time_from_plan, get_size_from_plan
    from scripts.a_collect_sql_plans import run_explain_analyze_json

    # Specify the path to your custom .env file
    dotenv_path = Path(".env.postgres")
    load_dotenv(dotenv_path=dotenv_path)

    DSN = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )

    def get_conn(dsn: str):
        return psycopg.connect(dsn)

    conn = get_conn(DSN)
    sql = open(sys.argv[1]).read()
    with conn.cursor() as cur:
        plan = run_explain_analyze_json(cur, sql, analyze=True)
    plan = plan[0]
    features = extract_features(plan, sql)
    prediction = predict("catboost", "time_models", sql, plan)
    actual_time = get_time_from_plan(plan)
    print(f"actual time: {actual_time}, prediction: {prediction}")

    prediction = predict("catboost", "size_models", sql, plan)
    actual_size = get_size_from_plan(plan)
    print(f"actual size: {actual_size}, prediction: {prediction}")
