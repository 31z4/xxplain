"""
Pydantic models for API requests and responses
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Запрос на предсказание времени выполнения"""
    query_text: Optional[str] = Field(None, description="Текст SQL запроса")
    model_name: Optional[str] = Field(None, description="Модель для предсказания")


class PredictionResponse(BaseModel):
    """Ответ с предсказанием времени выполнения"""
    prediction: Optional[float] = Field(None, description="Предсказанное время выполнения в миллисекундах")
    status: str = Field(..., description="Статус выполнения: success или error")
    error: Optional[str] = Field(None, description="Описание ошибки, если status = error")