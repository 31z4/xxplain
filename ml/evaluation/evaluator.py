"""
Model evaluator for comprehensive assessment
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json

from .metrics import calculate_metrics, evaluate_model_performance, compare_models
from ..models import ModelManager


class ModelEvaluator:
    """Класс для комплексной оценки моделей"""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """
        Инициализация evaluator
        
        Args:
            model_manager: Менеджер моделей для загрузки и предсказаний
        """
        self.model_manager = model_manager
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_single_model(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        model_name: str = "Unknown",
        dataset_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Оценивает одну модель
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            model_name: Название модели
            dataset_name: Название датасета
            
        Returns:
            Результаты оценки
        """
        evaluation_result = evaluate_model_performance(y_true, y_pred, model_name)
        evaluation_result["dataset_name"] = dataset_name
        evaluation_result["evaluation_timestamp"] = pd.Timestamp.now().isoformat()
        
        # Сохраняем в историю
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def evaluate_multiple_models(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        predictions: Dict[str, Union[np.ndarray, pd.Series, List]],
        dataset_name: str = "Unknown",
        primary_metric: str = "Q_Error_Median"
    ) -> Dict[str, Any]:
        """
        Сравнивает несколько моделей
        
        Args:
            y_true: Истинные значения
            predictions: Словарь {название_модели: предсказания}
            dataset_name: Название датасета
            primary_metric: Основная метрика для ранжирования
            
        Returns:
            Результаты сравнения
        """
        comparison_results = compare_models(y_true, predictions, primary_metric)
        
        # Добавляем метаинформацию
        for model_name, results in comparison_results.items():
            results["dataset_name"] = dataset_name
            results["evaluation_timestamp"] = pd.Timestamp.now().isoformat()
            results["primary_metric"] = primary_metric
            
            # Сохраняем в историю
            self.evaluation_history.append(results)
        
        # Создаем сводку
        summary = self._create_comparison_summary(comparison_results, primary_metric)
        
        return {
            "results": comparison_results,
            "summary": summary,
            "dataset_name": dataset_name,
            "primary_metric": primary_metric
        }
    
    def evaluate_with_model_manager(
        self,
        test_features: List[Dict[str, Any]],
        y_true: Union[np.ndarray, pd.Series, List],
        model_names: Optional[List[str]] = None,
        dataset_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Оценивает модели используя ModelManager
        
        Args:
            test_features: Список словарей с признаками для каждого теста
            y_true: Истинные значения времени выполнения
            model_names: Список моделей для оценки (если None, использует все доступные)
            dataset_name: Название датасета
            
        Returns:
            Результаты оценки
        """
        if not self.model_manager:
            raise ValueError("ModelManager не инициализирован")
        
        if model_names is None:
            available_models = self.model_manager.get_available_models()
            model_names = [m["name"] for m in available_models if m["loaded"]]
        
        # Получаем предсказания от каждой модели
        predictions = {}
        for model_name in model_names:
            model_predictions = []
            for features in test_features:
                try:
                    result = self.model_manager.predict(features, model_name)
                    if result["status"] == "success":
                        model_predictions.append(result["prediction"])
                    else:
                        model_predictions.append(np.nan)
                except Exception:
                    model_predictions.append(np.nan)
            
            predictions[model_name] = np.array(model_predictions)
        
        # Оцениваем модели
        return self.evaluate_multiple_models(y_true, predictions, dataset_name)
    
    def cross_validate_features(
        self,
        features_list: List[Dict[str, Any]],
        y_true: Union[np.ndarray, pd.Series, List],
        model_name: str = "GradientBoosting",
        n_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Кросс-валидация на уровне признаков
        
        Args:
            features_list: Список словарей с признаками
            y_true: Истинные значения
            model_name: Название модели
            n_folds: Количество фолдов
            
        Returns:
            Результаты кросс-валидации
        """
        if not self.model_manager:
            raise ValueError("ModelManager не инициализирован")
        
        n_samples = len(features_list)
        fold_size = n_samples // n_folds
        
        fold_results = []
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            # Выделяем тестовую часть для этого фолда
            test_features = features_list[start_idx:end_idx]
            test_y = y_true[start_idx:end_idx]
            
            # Получаем предсказания
            predictions = []
            for features in test_features:
                try:
                    result = self.model_manager.predict(features, model_name)
                    if result["status"] == "success":
                        predictions.append(result["prediction"])
                    else:
                        predictions.append(np.nan)
                except Exception:
                    predictions.append(np.nan)
            
            # Оцениваем этот фолд
            fold_metrics = calculate_metrics(test_y, predictions)
            fold_metrics["fold"] = fold
            fold_metrics["n_samples"] = len(test_y)
            fold_results.append(fold_metrics)
        
        # Агрегируем результаты
        aggregated_metrics = self._aggregate_cv_results(fold_results)
        
        return {
            "model_name": model_name,
            "n_folds": n_folds,
            "fold_results": fold_results,
            "aggregated_metrics": aggregated_metrics
        }
    
    def _create_comparison_summary(
        self, 
        comparison_results: Dict[str, Dict[str, Any]], 
        primary_metric: str
    ) -> Dict[str, Any]:
        """Создает сводку по результатам сравнения"""
        valid_results = {
            name: res for name, res in comparison_results.items()
            if "rank" in res
        }
        
        if not valid_results:
            return {"error": "Нет валидных результатов для сравнения"}
        
        # Лучшая модель
        best_model = min(valid_results.items(), key=lambda x: x[1]["rank"])
        
        # Статистика по метрикам
        metrics_stats = {}
        metric_names = ["MAE", "RMSE", "Q_Error_Median", "Q_Error_95p", "R2"]
        
        for metric in metric_names:
            values = [res.get(metric, np.nan) for res in valid_results.values()]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                metrics_stats[metric] = {
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values))
                }
        
        return {
            "total_models": len(comparison_results),
            "valid_models": len(valid_results),
            "best_model": {
                "name": best_model[0],
                "rank": best_model[1]["rank"],
                primary_metric: best_model[1].get(primary_metric, "N/A")
            },
            "metrics_statistics": metrics_stats
        }
    
    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирует результаты кросс-валидации"""
        aggregated = {}
        
        # Список метрик для агрегации
        metric_names = ["MAE", "RMSE", "Q_Error_Median", "Q_Error_95p", "R2", "MAPE"]
        
        for metric in metric_names:
            values = [fold.get(metric, np.nan) for fold in fold_results]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_values:
                aggregated[f"{metric}_mean"] = float(np.mean(valid_values))
                aggregated[f"{metric}_std"] = float(np.std(valid_values))
                aggregated[f"{metric}_min"] = float(np.min(valid_values))
                aggregated[f"{metric}_max"] = float(np.max(valid_values))
        
        return aggregated
    
    def get_evaluation_history(
        self, 
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Возвращает историю оценок с фильтрацией
        
        Args:
            model_name: Фильтр по названию модели
            dataset_name: Фильтр по названию датасета
            
        Returns:
            Отфильтрованная история оценок
        """
        filtered_history = self.evaluation_history
        
        if model_name:
            filtered_history = [
                eval_result for eval_result in filtered_history
                if eval_result.get("model_name") == model_name
            ]
        
        if dataset_name:
            filtered_history = [
                eval_result for eval_result in filtered_history
                if eval_result.get("dataset_name") == dataset_name
            ]
        
        return filtered_history
    
    def save_evaluation_report(
        self,
        filepath: str,
        evaluation_results: Dict[str, Any]
    ) -> None:
        """
        Сохраняет отчет об оценке в JSON файл
        
        Args:
            filepath: Путь к файлу
            evaluation_results: Результаты оценки
        """
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Преобразуем numpy типы в стандартные Python типы
        serializable_results = self._make_json_serializable(evaluation_results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Преобразует объект для JSON сериализации"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def clear_history(self) -> None:
        """Очищает историю оценок"""
        self.evaluation_history.clear()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Возвращает сводную статистику по всем оценкам"""
        if not self.evaluation_history:
            return {"message": "Нет данных для анализа"}
        
        # Группируем по моделям
        models_stats = {}
        for eval_result in self.evaluation_history:
            model_name = eval_result.get("model_name", "Unknown")
            if model_name not in models_stats:
                models_stats[model_name] = []
            models_stats[model_name].append(eval_result)
        
        # Агрегируем статистику
        summary = {
            "total_evaluations": len(self.evaluation_history),
            "unique_models": len(models_stats),
            "models_summary": {}
        }
        
        for model_name, evaluations in models_stats.items():
            q_errors = [e.get("Q_Error_Median", np.nan) for e in evaluations]
            valid_q_errors = [q for q in q_errors if not np.isnan(q)]
            
            if valid_q_errors:
                summary["models_summary"][model_name] = {
                    "evaluations_count": len(evaluations),
                    "avg_q_error_median": float(np.mean(valid_q_errors)),
                    "best_q_error_median": float(np.min(valid_q_errors)),
                    "worst_q_error_median": float(np.max(valid_q_errors))
                }
        
        return summary