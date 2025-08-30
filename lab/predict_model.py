import ast
import json
import sys
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_models_and_metadata(models_dict, scaler, feature_names, model_dir="models"):
    """
    Сохраняет модели, скейлер и метаданные
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # Сохраняем каждую модель
    for model_name, model in models_dict.items():
        model_path = model_dir / f"{model_name.lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Модель {model_name} сохранена: {model_path}")
    
    # Сохраняем скейлер
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Скейлер сохранен: {scaler_path}")
    
    # Сохраняем метаданные
    metadata = {
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "models_available": list(models_dict.keys())
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Метаданные сохранены: {metadata_path}")


def try_multiple_models(X_train_scaled, y_train_log, X_test) -> tuple[Dict, Dict]:
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    results = {}

    # Обучаем модели
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train_log)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        results[name] = y_pred

    return trained_models, results


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


    trained_models, results = try_multiple_models(X_train, y_train_log, X_test)
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

    # Сохраняем модели и метаданные
    save_models_and_metadata(
        trained_models, 
        scaler, 
        X_train.columns, 
    )


class ModelLoader:
    """
    Класс для загрузки и использования сохраненных моделей
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = None
        self.metadata = None
        self.load_all()
    
    def load_all(self):
        """Загружает все модели, скейлер и метаданные"""
        try:
            # Загружаем метаданные
            metadata_path = self.model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Загружаем скейлер
            scaler_path = self.model_dir / "scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            
            # Загружаем модели
            for model_name in self.metadata["models_available"]:
                model_path = self.model_dir / f"{model_name.lower()}_model.pkl"
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Модель {model_name} загружена")
                else:
                    print(f"Файл модели {model_name} не найден: {model_path}")
            
            print(f"Загружено {len(self.models)} моделей")
            
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            raise
    
    def predict(self, features, model_name="GradientBoosting"):
        """
        Делает предсказание для заданных признаков
        
        Args:
            features: dict или pandas.DataFrame с признаками
            model_name: название модели для предсказания
        
        Returns:
            float: предсказанное время выполнения
        """
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(self.models.keys())}")
        
        # Преобразуем входные данные
        if isinstance(features, dict):
            import pandas as pd
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Проверяем наличие всех необходимых признаков
        missing_features = set(self.metadata["feature_names"]) - set(features_df.columns)
        if missing_features:
            # Добавляем отсутствующие признаки как 0
            for feature in missing_features:
                features_df[feature] = 0.0
        
        # Упорядочиваем колонки как при обучении
        features_df = features_df[self.metadata["feature_names"]]
        
        # Масштабируем
        features_scaled = self.scaler.transform(features_df)
        
        # Предсказываем
        model = self.models[model_name]
        y_pred_log = model.predict(features_scaled)
        
        # Обратное преобразование из логарифмического масштаба
        y_pred = np.expm1(y_pred_log)
        
        # Ограничиваем снизу нулем
        y_pred = np.maximum(y_pred, 0)
        
        return y_pred[0] if len(y_pred) == 1 else y_pred
    
    def predict_all_models(self, features):
        """Предсказания от всех доступных моделей"""
        predictions = {}
        for model_name in self.models.keys():
            try:
                predictions[model_name] = self.predict(features, model_name)
            except Exception as e:
                predictions[model_name] = f"Error: {e}"
        return predictions
    
    def get_model_info(self, model_name="GradientBoosting"):
        """Информация о модели"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        info = {
            "model_type": type(model).__name__,
            "n_features": self.metadata["n_features"],
            "feature_names": self.metadata["feature_names"]
        }
        
        # Дополнительная информация для разных типов моделей
        if hasattr(model, 'feature_importances_'):
            # Для RandomForest и GradientBoosting
            importances = model.feature_importances_
            feature_importance = dict(zip(self.metadata["feature_names"], importances))
            info["feature_importances"] = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        if hasattr(model, 'n_estimators'):
            info["n_estimators"] = model.n_estimators
        
        if hasattr(model, 'learning_rate'):
            info["learning_rate"] = model.learning_rate
            
        return info


if __name__ == "__main__":
    main()
    model_loader = ModelLoader("models")
    expected_time = 254
    features = {'n_nodes': 4.0, 'max_depth': 3, 'sum_est_startup_cost': 518085.66000000003, 'sum_est_total_cost': 689881.5800000001, 'sum_plan_rows': 46560.0, 'sum_plan_width': 108.0, 'max_est_total_cost': 173028.71, 'max_plan_rows': 46556, 'max_plan_width': 32, 'n_parallel_aware': 1.0, 'n_sorts': 0.0, 'n_aggregates': 2.0, 'n_window': 0.0, 'n_gather': 1.0, 'sum_workers_planned': 2.0, 'has_limit': 0.0, 'n_filters': 1.0, 'sum_filter_len': 157.0, 'count_node_Aggregate': 2, 'count_node_Gather': 1, 'count_node_Seq_Scan': 1, 'root_total_cost': 173028.71, 'root_startup_cost': 173028.7, 'root_plan_rows': 1, 'root_plan_width': 32, 'ratio_startup_total_cost': 0.9999999422061173, 'avg_est_total_cost_per_node': 172470.39500000002, 'avg_plan_rows_per_node': 11640.0, 'avg_plan_width_per_node': 27.0, 'query_length': 242, 'query_tokens': 30, 'pgconf_random_page_cost': 4.0, 'pgconf_seq_page_cost': 1.0, 'pgconf_cpu_tuple_cost': 0.01, 'pgconf_cpu_index_tuple_cost': 0.005, 'pgconf_cpu_operator_cost': 0.0025, 'pgconf_work_mem': 4.0, 'pgconf_max_parallel_workers_per_gather': 2.0, 'pgconf_jit': 1.0}
    print(f'Expected: {expected_time}')
    for model in model_loader.models.keys():
        print(f'Model {model} prediction is {model_loader.predict(features, model)}')