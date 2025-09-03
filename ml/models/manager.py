"""
High-level model manager combining loader and registry
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .loader import ModelLoader
from .registry import ModelRegistry, ModelInfo
from ..features import FeatureExtractor


class ModelManager:
    """Высокоуровневый менеджер для работы с моделями"""
    
    def __init__(
        self, 
        model_dir: str = "models",
        registry_path: Optional[str] = None
    ):
        """
        Инициализация менеджера
        
        Args:
            model_dir: Директория с моделями
            registry_path: Путь к файлу реестра (по умолчанию model_dir/registry.json)
        """
        self.model_dir = Path(model_dir)
        
        if registry_path is None:
            registry_path = str(self.model_dir / "registry.json")
        
        self.loader = ModelLoader(model_dir)
        self.registry = ModelRegistry(registry_path)
        self.feature_extractor = FeatureExtractor()
    
    def predict(
        self, 
        features: Dict[str, Any], 
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Делает предсказание времени выполнения
        
        Args:
            features: Словарь с признаками или query plan
            model_name: Название модели (если не указано, использует лучшую)
            
        Returns:
            Словарь с результатом предсказания
        """
        # Если модель не указана, используем лучшую
        if model_name is None:
            best_model = self.registry.get_best_model()
            if best_model:
                model_name = best_model.name
            else:
                model_name = "GradientBoosting"  # Fallback
        
        try:
            # Делаем предсказание
            prediction = self.loader.predict(features, model_name)
            
            # Получаем информацию о модели
            model_info = self.loader.get_model_info(model_name)
            
            return {
                "prediction": prediction,
                "model_used": model_name,
                "model_info": model_info,
                "status": "success"
            }
        
        except Exception as e:
            return {
                "prediction": None,
                "model_used": model_name,
                "error": str(e),
                "status": "error"
            }
    
    def predict_with_plan(
        self, 
        plan_json: Dict[str, Any], 
        query_text: str = "",
        server_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Делает предсказание по query plan
        
        Args:
            plan_json: JSON представление плана
            query_text: Текст SQL запроса
            server_params: Параметры сервера PostgreSQL
            model_name: Название модели
            
        Returns:
            Словарь с результатом предсказания
        """
        try:
            # Извлекаем признаки из плана
            features = self.feature_extractor.extract_features(
                plan_json, query_text, server_params
            )
            
            # Делаем предсказание
            result = self.predict(features, model_name)
            result["features_extracted"] = len(features)
            result["plan"] = json.dumps(plan_json)
            
            return result
            
        except Exception as e:
            return {
                "prediction": None,
                "model_used": model_name,
                "error": f"Feature extraction failed: {e}",
                "status": "error"
            }
    
    def get_all_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Получает предсказания от всех доступных моделей
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Словарь с предсказаниями от всех моделей
        """
        try:
            predictions = self.loader.predict_all_models(features)
            
            # Добавляем информацию о моделях
            detailed_predictions = {}
            for model_name, prediction in predictions.items():
                model_info = self.registry.get_model_info(model_name)
                detailed_predictions[model_name] = {
                    "prediction": prediction,
                    "model_info": model_info
                }
            
            return {
                "predictions": detailed_predictions,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "predictions": {},
                "error": str(e),
                "status": "error"
            }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Возвращает список доступных моделей с их информацией
        
        Returns:
            Список словарей с информацией о моделях
        """
        models_info = []
        
        # Модели из loader
        loaded_models = self.loader.get_available_models()
        
        for model_name in loaded_models:
            # Информация из loader
            loader_info = self.loader.get_model_info(model_name)
            
            # Информация из registry
            registry_info = self.registry.get_model_info(model_name)
            
            model_info = {
                "name": model_name,
                "loaded": True,
                "loader_info": loader_info,
                "registry_info": registry_info.__dict__ if registry_info else None
            }
            
            models_info.append(model_info)
        
        return models_info
    
    def get_best_model_info(self, metric: str = "Q_Error_Median") -> Optional[Dict[str, Any]]:
        """
        Возвращает информацию о лучшей модели
        
        Args:
            metric: Метрика для определения лучшей модели
            
        Returns:
            Информация о лучшей модели
        """
        best_model = self.registry.get_best_model(metric)
        if not best_model:
            return None
        
        loader_info = self.loader.get_model_info(best_model.name)
        
        return {
            "name": best_model.name,
            "registry_info": best_model.__dict__,
            "loader_info": loader_info,
            "is_loaded": best_model.name in self.loader.get_available_models()
        }
    
    def validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Проверяет корректность признаков для предсказания
        
        Args:
            features: Словарь с признаками
            
        Returns:
            Результат валидации
        """
        try:
            # Получаем ожидаемые признаки
            expected_features = set(self.loader.get_feature_names())
            provided_features = set(features.keys())
            
            missing_features = expected_features - provided_features
            extra_features = provided_features - expected_features
            
            return {
                "valid": len(missing_features) == 0,
                "missing_features": list(missing_features),
                "extra_features": list(extra_features),
                "expected_count": len(expected_features),
                "provided_count": len(provided_features),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "status": "error"
            }
    
    def get_feature_importance(self, model_name: str = "GradientBoosting") -> Optional[Dict[str, float]]:
        """
        Возвращает важность признаков для модели
        
        Args:
            model_name: Название модели
            
        Returns:
            Словарь с важностью признаков
        """
        model_info = self.loader.get_model_info(model_name)
        if model_info and "feature_importances" in model_info:
            return model_info["feature_importances"]
        return None
    
    def reload_models(self) -> Dict[str, Any]:
        """
        Перезагружает модели
        
        Returns:
            Результат перезагрузки
        """
        try:
            # Перезагружаем loader
            self.loader = ModelLoader(str(self.model_dir))
            
            # Перезагружаем registry
            self.registry._load_registry()
            
            loaded_models = self.loader.get_available_models()
            
            return {
                "models_loaded": len(loaded_models),
                "models": loaded_models,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "models_loaded": 0,
                "error": str(e),
                "status": "error"
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по состоянию менеджера
        
        Returns:
            Словарь с информацией о состоянии
        """
        try:
            registry_summary = self.registry.get_model_summary()
            loaded_models = self.loader.get_available_models()
            
            return {
                "registry_summary": registry_summary,
                "loaded_models": loaded_models,
                "models_loaded_count": len(loaded_models),
                "loader_ready": self.loader.is_loaded(),
                "feature_extractor_ready": True,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }