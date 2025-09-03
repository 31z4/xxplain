"""
Model trainer for experiments
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import json

from ..evaluation import ModelEvaluator, calculate_metrics
from ..features import FeatureExtractor, FeatureConfig


class ModelTrainer:
    """Класс для обучения моделей в экспериментах"""
    
    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        random_state: int = 42
    ):
        """
        Инициализация тренера
        
        Args:
            feature_config: Конфигурация для извлечения признаков
            random_state: Сид для воспроизводимости
        """
        self.feature_config = feature_config
        self.random_state = random_state
        self.feature_extractor = FeatureExtractor(feature_config)
        self.evaluator = ModelEvaluator()
        self.scaler: Optional[StandardScaler] = None
        
        # Доступные модели
        self.available_models = {
            'Linear': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor
        }
    
    def prepare_data(
        self,
        plans_and_targets: List[Dict[str, Any]],
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения
        
        Args:
            plans_and_targets: Список словарей с планами и целевыми значениями
            test_size: Размер тестовой выборки
            validation_size: Размер валидационной выборки
            
        Returns:
            Кортеж (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Извлекаем признаки
        features_list = []
        targets = []
        
        for item in plans_and_targets:
            try:
                plan = item.get('plan', {})
                query_text = item.get('query_text', '')
                server_params = item.get('server_params', {})
                target = item.get('target')
                
                if target is None:
                    continue
                
                features = self.feature_extractor.extract_features(
                    plan, query_text, server_params
                )
                
                features_list.append(features)
                targets.append(float(target))
                
            except Exception as e:
                print(f"Ошибка обработки элемента: {e}")
                continue
        
        if not features_list:
            raise ValueError("Не удалось извлечь признаки ни из одного элемента")
        
        # Преобразуем в DataFrame
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0.0)
        
        # Целевая переменная (логарифм для лучшего обучения)
        y = np.log1p(np.array(targets))
        X = features_df.values
        
        # Разделяем на train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])
        
        # Нормализация признаков
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if validation_size > 0:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = np.array([])
        
        print(f"Данные подготовлены:")
        print(f"  Train: {X_train_scaled.shape}")
        print(f"  Validation: {X_val_scaled.shape if validation_size > 0 else 'None'}")
        print(f"  Test: {X_test_scaled.shape}")
        print(f"  Признаков: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train_single_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Обучает одну модель
        
        Args:
            model_type: Тип модели
            X_train: Обучающие признаки
            y_train: Обучающие цели
            X_val: Валидационные признаки
            y_val: Валидационные цели
            model_params: Параметры модели
            
        Returns:
            Результаты обучения
        """
        if model_type not in self.available_models:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Создаем модель с параметрами
        model_class = self.available_models[model_type]
        model_params = model_params or {}
        
        # Добавляем random_state для моделей, которые его поддерживают
        if hasattr(model_class(), 'random_state'):
            model_params.setdefault('random_state', self.random_state)
        
        model = model_class(**model_params)
        
        # Обучаем модель
        print(f"Обучение модели {model_type}...")
        model.fit(X_train, y_train)
        
        # Предсказания на обучающей выборке
        y_train_pred_log = model.predict(X_train)
        y_train_pred = np.expm1(y_train_pred_log)
        y_train_true = np.expm1(y_train)
        
        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        
        # Предсказания на валидационной выборке (если есть)
        val_metrics = {}
        if X_val is not None and len(X_val) > 0 and y_val is not None and len(y_val) > 0:
            y_val_pred_log = model.predict(X_val)
            y_val_pred = np.expm1(y_val_pred_log)
            y_val_true = np.expm1(y_val)
            val_metrics = calculate_metrics(y_val_true, y_val_pred)
        
        return {
            'model': model,
            'model_type': model_type,
            'model_params': model_params,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'feature_names': self.feature_extractor.get_feature_names()
        }
    
    def train_multiple_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Обучает несколько моделей
        
        Args:
            X_train: Обучающие признаки
            y_train: Обучающие цели
            X_val: Валидационные признаки
            y_val: Валидационные цели
            model_types: Список типов моделей
            model_configs: Конфигурации для каждого типа модели
            
        Returns:
            Результаты обучения всех моделей
        """
        if model_types is None:
            model_types = list(self.available_models.keys())
        
        model_configs = model_configs or {}
        results = {}
        
        for model_type in model_types:
            try:
                model_params = model_configs.get(model_type, {})
                result = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, model_params
                )
                results[model_type] = result
                
                # Выводим основные метрики
                train_mae = result['train_metrics'].get('MAE', 'N/A')
                val_mae = result['val_metrics'].get('MAE', 'N/A') if result['val_metrics'] else 'N/A'
                print(f"{model_type}: Train MAE={train_mae:.4f}, Val MAE={val_mae}")
                
            except Exception as e:
                print(f"Ошибка обучения модели {model_type}: {e}")
                continue
        
        return results
    
    def evaluate_on_test(
        self,
        trained_models: Dict[str, Dict[str, Any]],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Оценивает обученные модели на тестовой выборке
        
        Args:
            trained_models: Результаты обучения моделей
            X_test: Тестовые признаки
            y_test: Тестовые цели
            
        Returns:
            Результаты оценки на тесте
        """
        test_results = {}
        
        y_test_true = np.expm1(y_test)
        
        for model_name, model_data in trained_models.items():
            try:
                model = model_data['model']
                
                # Предсказания
                y_test_pred_log = model.predict(X_test)
                y_test_pred = np.expm1(y_test_pred_log)
                y_test_pred = np.maximum(y_test_pred, 0)  # Ограничиваем снизу нулем
                
                # Метрики
                test_metrics = calculate_metrics(y_test_true, y_test_pred)
                
                test_results[model_name] = {
                    'test_metrics': test_metrics,
                    'predictions': y_test_pred.tolist()
                }
                
                # Выводим основные метрики
                test_mae = test_metrics.get('MAE', 'N/A')
                test_q_error = test_metrics.get('Q_Error_Median', 'N/A')
                print(f"{model_name} Test: MAE={test_mae:.4f}, Q-Error Median={test_q_error:.4f}")
                
            except Exception as e:
                print(f"Ошибка оценки модели {model_name}: {e}")
                continue
        
        return test_results
    
    def save_trained_models(
        self,
        trained_models: Dict[str, Dict[str, Any]],
        output_dir: str = "models",
        experiment_name: str = "experiment"
    ) -> Dict[str, str]:
        """
        Сохраняет обученные модели
        
        Args:
            trained_models: Результаты обучения моделей
            output_dir: Директория для сохранения
            experiment_name: Название эксперимента
            
        Returns:
            Пути к сохраненным моделям
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Сохраняем скейлер
        scaler_path = output_path / f"{experiment_name}_scaler.pkl"
        if self.scaler:
            joblib.dump(self.scaler, scaler_path)
            saved_paths['scaler'] = str(scaler_path)
        
        # Сохраняем модели
        models_dict = {}
        for model_name, model_data in trained_models.items():
            model_path = output_path / f"{experiment_name}_{model_name.lower()}_model.pkl"
            joblib.dump(model_data['model'], model_path)
            saved_paths[model_name] = str(model_path)
            models_dict[model_name] = model_data['model']
        
        # Сохраняем метаданные
        if trained_models:
            first_model = next(iter(trained_models.values()))
            feature_names = first_model.get('feature_names', [])
            
            metadata = {
                "experiment_name": experiment_name,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "models_available": list(trained_models.keys()),
                "feature_config": self.feature_config.__dict__ if self.feature_config else None
            }
            
            metadata_path = output_path / f"{experiment_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_paths['metadata'] = str(metadata_path)
        
        print(f"Модели сохранены в {output_path}")
        return saved_paths
    
    def hyperparameter_search(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Dict[str, List[Any]],
        metric: str = "MAE"
    ) -> Dict[str, Any]:
        """
        Простой поиск гиперпараметров по сетке
        
        Args:
            model_type: Тип модели
            X_train: Обучающие признаки
            y_train: Обучающие цели
            X_val: Валидационные признаки
            y_val: Валидационные цели
            param_grid: Сетка параметров
            metric: Метрика для оптимизации
            
        Returns:
            Результаты поиска
        """
        from itertools import product
        
        param_combinations = [
            dict(zip(param_grid.keys(), values))
            for values in product(*param_grid.values())
        ]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        all_results = []
        
        print(f"Поиск гиперпараметров для {model_type}, {len(param_combinations)} комбинаций...")
        
        for i, params in enumerate(param_combinations):
            try:
                result = self.train_single_model(
                    model_type, X_train, y_train, X_val, y_val, params
                )
                
                score = result['val_metrics'].get(metric, float('inf'))
                all_results.append({
                    'params': params,
                    'score': score,
                    'metrics': result['val_metrics']
                })
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = result['model']
                
                if (i + 1) % 10 == 0:
                    print(f"  Обработано {i + 1}/{len(param_combinations)}")
                
            except Exception as e:
                print(f"  Ошибка с параметрами {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'all_results': all_results
        }
    
    def get_default_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """Возвращает стандартные сетки параметров для поиска"""
        return {
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }