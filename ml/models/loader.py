"""
Model loader for trained ML models
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ModelLoader:
    """Загрузчик обученных моделей"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Инициализация загрузчика
        
        Args:
            model_dir: Директория с моделями
        """
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._loaded = False
    
    def load_all(self) -> None:
        """Загружает все модели, скейлер и метаданные"""
        try:
            # Загружаем метаданные
            self._load_metadata()
            
            # Загружаем скейлер
            self._load_scaler()
            
            # Загружаем модели
            self._load_models()
            
            self._loaded = True
            print(f"Загружено {len(self.models)} моделей")
            
        except Exception as e:
            print(f"Ошибка при загрузке моделей: {e}")
            raise
    
    def _load_metadata(self) -> None:
        """Загружает метаданные моделей"""
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def _load_scaler(self) -> None:
        """Загружает скейлер для нормализации признаков"""
        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Файл скейлера не найден: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
    
    def _load_models(self) -> None:
        """Загружает все доступные модели"""
        if not self.metadata:
            raise ValueError("Метаданные не загружены")
        
        for model_name in self.metadata["models_available"]:
            model_path = self.model_dir / f"{model_name.lower()}_model.pkl"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"Модель {model_name} загружена")
            else:
                print(f"Файл модели {model_name} не найден: {model_path}")
    
    def predict(
        self, 
        features: Dict[str, Any], 
        model_name: str = "GradientBoosting"
    ) -> float:
        """
        Делает предсказание для заданных признаков
        
        Args:
            features: Словарь с признаками
            model_name: Название модели для предсказания
        
        Returns:
            Предсказанное время выполнения
        """
        if not self._loaded:
            self.load_all()
        
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(self.models.keys())}")
        
        # Подготавливаем признаки
        features_array = self._prepare_features(features)
        
        # Предсказываем
        model = self.models[model_name]
        y_pred_log = model.predict(features_array)
        
        # Обратное преобразование из логарифмического масштаба
        y_pred = np.expm1(y_pred_log)
        
        # Ограничиваем снизу нулем
        y_pred = np.maximum(y_pred, 0)
        
        return float(y_pred[0]) if len(y_pred) == 1 else y_pred
    
    def predict_all_models(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Предсказания от всех доступных моделей
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Словарь с предсказаниями от каждой модели
        """
        predictions = {}
        for model_name in self.models.keys():
            try:
                predictions[model_name] = self.predict(features, model_name)
            except Exception as e:
                predictions[model_name] = f"Error: {e}"
        return predictions
    
    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Подготавливает признаки для предсказания
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Нормализованный массив признаков
        """
        if not self.metadata or not self.scaler:
            raise ValueError("Метаданные или скейлер не загружены")
        
        # Преобразуем в DataFrame
        features_df = pd.DataFrame([features])
        
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
        
        return features_scaled
    
    def get_model_info(self, model_name: str = "GradientBoosting") -> Optional[Dict[str, Any]]:
        """
        Информация о модели
        
        Args:
            model_name: Название модели
            
        Returns:
            Словарь с информацией о модели
        """
        if not self._loaded:
            self.load_all()
            
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        info = {
            "model_type": type(model).__name__,
            "n_features": self.metadata["n_features"] if self.metadata else 0,
            "feature_names": self.metadata["feature_names"] if self.metadata else []
        }
        
        # Дополнительная информация для разных типов моделей
        if hasattr(model, 'feature_importances_'):
            # Для RandomForest и GradientBoosting
            importances = model.feature_importances_
            if self.metadata:
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
        
        if hasattr(model, 'alpha'):
            info["alpha"] = model.alpha
            
        return info
    
    def get_available_models(self) -> List[str]:
        """Возвращает список доступных моделей"""
        if not self._loaded:
            self.load_all()
        return list(self.models.keys())
    
    def is_loaded(self) -> bool:
        """Проверяет, загружены ли модели"""
        return self._loaded
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список имен признаков"""
        if not self.metadata:
            if not self._loaded:
                self.load_all()
        return self.metadata["feature_names"] if self.metadata else []