"""
Business logic services for the backend API
"""

from typing import Dict, Any, Optional

from ml.models import ModelManager
from ml.plans import QueryPlanService
from .config import settings


class PredictionService:
    """Сервис для предсказания времени выполнения запросов"""
    
    def __init__(self):
        """Инициализация сервиса"""
        self._model_manager: Optional[ModelManager] = None
        self._query_plan_service: Optional[QueryPlanService] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Асинхронная инициализация сервиса"""
        if self._initialized:
            return
        
        try:
            # Инициализируем ModelManager
            self._model_manager = ModelManager(model_dir="models")
            
            # Инициализируем QueryPlanService
            self._query_plan_service = QueryPlanService(str(settings.POSTGRES_DSN))
            
            self._initialized = True
            print("PredictionService инициализирован успешно")
            
        except Exception as e:
            print(f"Ошибка инициализации PredictionService: {e}")
            raise
    
    async def predict_regression(
        self,
        query_text: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Предсказывает целевую метрику по SQL запросу
        
        Args:
            query_text: Текст SQL запроса
            
        Returns:
            Результат предсказания
        """
        await self.initialize()
        
        # Проверка успешной инициализации сервисов
        if not self._initialized or self._query_plan_service is None:
            return {
                "prediction": None,
                "error": "QueryPlanService не инициализирован",
                "status": "error"
            }
            
        if not self._model_manager:
            return {
                "prediction": None,
                "error": "ModelManager не инициализирован",
                "status": "error"
            }

        # Обработка случая, когда query_text может быть None
        if query_text is None or not query_text.strip():
            return {
                "prediction": None,
                "error": "Не предоставлен SQL запрос",
                "status": "error"
            }
        
        try:
            plan, server_params = await self._query_plan_service.get_plan_with_server_params(query_text)
            result = self._model_manager.predict_with_plan(
                plan, query_text or "", server_params, model_name
            )
            return result
            
        except Exception as e:
            return {
                "prediction": None,
                "plan": None,
                "error": f"Ошибка предсказания: {str(e)}",
                "status": "error"
            }


# Глобальный экземпляр сервиса
prediction_service = PredictionService()


# Dependency для FastAPI
async def get_prediction_service() -> PredictionService:
    """Dependency для получения сервиса предсказаний"""
    await prediction_service.initialize()
    return prediction_service