"""
Metrics for evaluating query execution time predictions
"""

from typing import Dict, Union, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, List], 
    y_pred: Union[np.ndarray, pd.Series, List]
) -> Dict[str, float]:
    """
    Рассчитывает комплексный набор метрик для оценки предсказаний времени выполнения
    
    Args:
        y_true: Истинные значения времени выполнения
        y_pred: Предсказанные значения времени выполнения
        
    Returns:
        Словарь с метриками
    """
    # Преобразуем в numpy arrays
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    if len(y_t) != len(y_p):
        raise ValueError("Длины массивов y_true и y_pred должны совпадать")
    
    if len(y_t) == 0:
        raise ValueError("Массивы не должны быть пустыми")
    
    metrics = {}
    
    # Основные регрессионные метрики
    metrics.update(_calculate_regression_metrics(y_t, y_p))
    
    # Q-error метрики (специфичные для баз данных)
    metrics.update(_calculate_q_error_metrics(y_t, y_p))
    
    # Процентные ошибки
    metrics.update(_calculate_percentage_metrics(y_t, y_p))
    
    # Дополнительные статистики
    metrics.update(_calculate_additional_stats(y_t, y_p))
    
    return metrics


def _calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Рассчитывает стандартные регрессионные метрики"""
    metrics = {}
    
    try:
        # Mean Absolute Error
        metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
        
        # Root Mean Squared Error
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # Mean Squared Error
        metrics["MSE"] = float(mean_squared_error(y_true, y_pred))
        
        # R² Score
        try:
            r2 = r2_score(y_true, y_pred)
            metrics["R2"] = float(r2)
        except Exception:
            metrics["R2"] = float("nan")
        
        # Mean Absolute Percentage Error (MAPE)
        try:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics["MAPE"] = float(mape)
        except (ZeroDivisionError, RuntimeWarning):
            metrics["MAPE"] = float("nan")
        
    except Exception as e:
        print(f"Ошибка при расчете регрессионных метрик: {e}")
        metrics.update({
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MSE": float("nan"),
            "R2": float("nan"),
            "MAPE": float("nan")
        })
    
    return metrics


def _calculate_q_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Рассчитывает Q-error метрики"""
    metrics = {}
    
    try:
        q_errors = calculate_q_error(y_true, y_pred)
        
        if len(q_errors) > 0 and not np.all(np.isnan(q_errors)):
            # Среднее геометрическое Q-error
            log_q_errors = np.log(q_errors[~np.isnan(q_errors)])
            if len(log_q_errors) > 0:
                metrics["Q_Error_Mean"] = float(np.exp(np.mean(log_q_errors)))
            else:
                metrics["Q_Error_Mean"] = float("nan")
            
            # Процентили Q-error
            metrics["Q_Error_Median"] = float(np.nanpercentile(q_errors, 50))
            metrics["Q_Error_90p"] = float(np.nanpercentile(q_errors, 90))
            metrics["Q_Error_95p"] = float(np.nanpercentile(q_errors, 95))
            metrics["Q_Error_99p"] = float(np.nanpercentile(q_errors, 99))
            
            # Максимальный Q-error
            metrics["Q_Error_Max"] = float(np.nanmax(q_errors))
            
        else:
            metrics.update({
                "Q_Error_Mean": float("nan"),
                "Q_Error_Median": float("nan"),
                "Q_Error_90p": float("nan"),
                "Q_Error_95p": float("nan"),
                "Q_Error_99p": float("nan"),
                "Q_Error_Max": float("nan")
            })
            
    except Exception as e:
        print(f"Ошибка при расчете Q-error метрик: {e}")
        metrics.update({
            "Q_Error_Mean": float("nan"),
            "Q_Error_Median": float("nan"),
            "Q_Error_90p": float("nan"),
            "Q_Error_95p": float("nan"),
            "Q_Error_99p": float("nan"),
            "Q_Error_Max": float("nan")
        })
    
    return metrics


def _calculate_percentage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Рассчитывает процентные метрики точности"""
    metrics = {}
    
    try:
        # Процент предсказаний в пределах различных факторов от истинного значения
        factors = [1.5, 2.0, 5.0, 10.0]
        
        for factor in factors:
            within_factor = np.sum(
                (y_pred >= y_true / factor) & (y_pred <= y_true * factor)
            ) / len(y_true) * 100
            metrics[f"Within_{factor}x"] = float(within_factor)
            
    except Exception as e:
        print(f"Ошибка при расчете процентных метрик: {e}")
        for factor in [1.5, 2.0, 5.0, 10.0]:
            metrics[f"Within_{factor}x"] = float("nan")
    
    return metrics


def _calculate_additional_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Рассчитывает дополнительные статистики"""
    metrics = {}
    
    try:
        # Bias (средняя ошибка)
        bias = np.mean(y_pred - y_true)
        metrics["Bias"] = float(bias)
        
        # Относительный bias
        relative_bias = bias / np.mean(y_true) * 100
        metrics["Relative_Bias"] = float(relative_bias)
        
        # Корреляция
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        metrics["Correlation"] = float(correlation)
        
        # Стандартное отклонение ошибок
        errors = y_pred - y_true
        metrics["Error_Std"] = float(np.std(errors))
        
        # Нормализованная RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        nrmse = rmse / (np.max(y_true) - np.min(y_true)) * 100
        metrics["NRMSE"] = float(nrmse)
        
    except Exception as e:
        print(f"Ошибка при расчете дополнительных статистик: {e}")
        metrics.update({
            "Bias": float("nan"),
            "Relative_Bias": float("nan"),
            "Correlation": float("nan"),
            "Error_Std": float("nan"),
            "NRMSE": float("nan")
        })
    
    return metrics


def calculate_q_error(
    y_true: Union[np.ndarray, pd.Series, List], 
    y_pred: Union[np.ndarray, pd.Series, List]
) -> np.ndarray:
    """
    Рассчитывает Q-error для каждого предсказания
    
    Q-error = max(predicted/actual, actual/predicted)
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
        
    Returns:
        Массив Q-error значений
    """
    # Преобразуем в numpy arrays
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    # Избегаем деления на ноль, добавляя малое значение
    epsilon = 1e-6
    y_t_safe = np.maximum(y_t, epsilon)
    y_p_safe = np.maximum(y_p, epsilon)
    
    # Рассчитываем Q-error для каждого предсказания
    q_errors = np.maximum(y_t_safe / y_p_safe, y_p_safe / y_t_safe)
    
    return q_errors


def evaluate_model_performance(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List],
    model_name: str = "Unknown"
) -> Dict[str, Union[float, str, int]]:
    """
    Комплексная оценка производительности модели
    
    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения
        model_name: Название модели
        
    Returns:
        Словарь с метриками и оценкой качества
    """
    base_metrics = calculate_metrics(y_true, y_pred)
    
    # Создаем новый словарь с расширенными типами
    metrics: Dict[str, Union[float, str, int]] = dict(base_metrics)
    
    # Добавляем метаинформацию
    metrics["model_name"] = model_name
    metrics["n_samples"] = len(y_true)
    
    # Определяем общую оценку качества на основе Q-error
    q_error_median_val = metrics.get("Q_Error_Median", float("inf"))
    
    # Убеждаемся что это число
    if isinstance(q_error_median_val, (int, float)):
        q_error_median = float(q_error_median_val)
    else:
        q_error_median = float("inf")
    
    if np.isnan(q_error_median) or q_error_median == float("inf"):
        quality = "Unknown"
    elif q_error_median <= 1.5:
        quality = "Excellent"
    elif q_error_median <= 2.0:
        quality = "Good"
    elif q_error_median <= 5.0:
        quality = "Fair"
    elif q_error_median <= 10.0:
        quality = "Poor"
    else:
        quality = "Very Poor"
    
    metrics["quality_rating"] = quality
    
    return metrics


def compare_models(
    y_true: Union[np.ndarray, pd.Series, List],
    predictions: Dict[str, Union[np.ndarray, pd.Series, List]],
    primary_metric: str = "Q_Error_Median"
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Сравнивает несколько моделей
    
    Args:
        y_true: Истинные значения
        predictions: Словарь {название_модели: предсказания}
        primary_metric: Основная метрика для ранжирования
        
    Returns:
        Словарь с результатами для каждой модели
    """
    results = {}
    
    for model_name, y_pred in predictions.items():
        try:
            metrics = evaluate_model_performance(y_true, y_pred, model_name)
            results[model_name] = metrics
        except Exception as e:
            results[model_name] = {
                "model_name": model_name,
                "error": str(e),
                "quality_rating": "Error"
            }
    
    # Ранжируем модели по основной метрике
    valid_results = {
        name: res for name, res in results.items() 
        if primary_metric in res and not np.isnan(res[primary_metric])
    }
    
    if valid_results:
        # Для Q-error меньше = лучше
        lower_is_better = primary_metric.startswith("Q_Error") or primary_metric in ["MAE", "RMSE", "MSE", "MAPE"]
        
        sorted_models = sorted(
            valid_results.items(),
            key=lambda x: x[1][primary_metric],
            reverse=not lower_is_better
        )
        
        # Добавляем ранги
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            results[model_name]["rank"] = rank
    
    return results