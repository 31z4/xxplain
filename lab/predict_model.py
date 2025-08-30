import ast
import json
import sys
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def try_multiple_models(X_train, y_train, X_test) -> Dict:
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = y_pred
    
    return results


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Возвращает метрики MAE, RMSE, R2
    """
    y_t = np.asarray(y_true.values, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    metrics = {"MAE": mae, "RMSE": rmse}
    try:
        r2 = r2_score(y_t, y_p)
    except Exception:
        # На случай вырожденного случая (мало точек, одна константа и т.п.)
        r2 = float("nan")
    metrics["R2"] = float(r2)
    return metrics


def main():
    # Пути по умолчанию
    train_path = "lab/train.csv"
    test_path = "lab/test.csv"
    out_preds_path = "lab/test_with_pred_{0}.csv"

    # Загружаем данные
    train_df = pd.read_csv(train_path, delimiter='\t')
    X_train = pd.DataFrame(
        train_df['feats'].map(lambda x: json.loads(x.replace("'", '"'))).to_list()
    ).fillna(0.0)
    X_train = X_train[[
    # можно попробовать добавить обучение по этим колонкам, но не помогает
       'n_nodes', 'max_depth', 'sum_est_startup_cost', 'sum_est_total_cost',
       'sum_plan_rows', 'sum_plan_width', 'max_est_total_cost',
       'max_plan_rows', 'max_plan_width', 'n_parallel_aware', 'n_sorts',
       'n_aggregates', 'n_window', 'n_gather', 'sum_workers_planned',
       'has_limit', 'n_filters', 'sum_filter_len', 'count_node_Sort',
       'count_node_Aggregate', 'count_node_Nested_Loop', 'count_join_Inner',
       'count_node_Index_Scan', 'count_node_Seq_Scan',
       'root_total_cost', 'root_startup_cost', 'root_plan_rows', 'root_plan_width',
       'ratio_startup_total_cost', 'avg_est_total_cost_per_node',
       'avg_plan_rows_per_node', 'avg_plan_width_per_node', 'query_length',
       'query_tokens', 'pgconf_random_page_cost', 'pgconf_seq_page_cost',
       'pgconf_cpu_tuple_cost', 'pgconf_cpu_index_tuple_cost',
       'pgconf_cpu_operator_cost', 'pgconf_work_mem',
       'pgconf_max_parallel_workers_per_gather', 'pgconf_jit',
       'count_node_Hash_Join', 'count_node_Hash', 'count_node_Gather',
       'count_node_Materialize', 'count_node_Gather_Merge',
       'count_node_Bitmap_Heap_Scan', 'count_node_Bitmap_Index_Scan',
       'count_node_Limit', 'count_node_Merge_Join', 'count_node_Memoize',
       'count_join_Right', 'count_join_Semi', 'count_node_Index_Only_Scan',
       'count_node_Incremental_Sort', 'count_join_Left', 'count_node_Result',
       'count_node_Unique', 'count_node_BitmapAnd', 'count_node_Subquery_Scan',
       'count_node_WindowAgg'
    ]]
    test_df = pd.read_csv(test_path, delimiter='\t')
    X_test = pd.DataFrame(
        test_df['feats'].map(lambda x: json.loads(x.replace("'", '"'))).to_list(),
        columns=X_train.columns
    ).fillna(0.0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # Найти параметры и применить к train
    X_test_scaled = scaler.transform(X_test)       # Применить параметры из train к test

    # Целевая переменная
    y_train = pd.to_numeric(train_df["target"], errors="coerce").astype(float)
    y_train_log = np.log1p(y_train)
    if y_train.isna().any():
        raise ValueError("В колонке 'target' train.csv есть некорректные значения, не приводимые к float.")


    results = try_multiple_models(X_train, y_train_log, X_test)
    for model_name, y_pred_log in results.items():
        print(f'---{model_name}---')
        # Предсказываем для test с обратным преобразованием
        y_pred = np.expm1(y_pred_log)
        
        # Ограничиваем предсказания снизу нулем
        y_pred = np.maximum(y_pred, 0)
        test_out = test_df.copy()
        test_out["pred"] = y_pred

        # Оцениваем на test, если есть 'target'
        if "target" in test_df.columns:
            y_test = pd.to_numeric(test_df["target"], errors="coerce").astype(float)
            if not y_test.isna().any():
                metrics = evaluate(y_test, y_pred)
                print("Test metrics:")
                for k, v in metrics.items():
                    print(f"  {k}: {v:.6f}")
            else:
                print("В test.csv колонка 'target' содержит некорректные значения; метрики не посчитаны.")
        else:
            print("В test.csv нет колонки 'target'; метрики не посчитаны.")

        # Сохраняем предсказания
        test_out.to_csv(out_preds_path.format(model_name), index=False)
        print(f"Предсказания сохранены в: {out_preds_path.format(model_name)}")


if __name__ == "__main__":
    main()