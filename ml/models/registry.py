"""
Model registry for tracking available models
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ModelInfo:
    """Информация о модели"""
    name: str
    model_type: str
    created_at: datetime
    metrics: Dict[str, float]
    feature_count: int
    model_path: Path
    metadata_path: Optional[Path] = None
    description: str = ""
    tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """Реестр доступных моделей"""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Инициализация реестра
        
        Args:
            registry_path: Путь к файлу реестра
        """
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelInfo] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Загружает реестр из файла"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                
                for model_name, model_data in data.items():
                    self.models[model_name] = ModelInfo(
                        name=model_data["name"],
                        model_type=model_data["model_type"],
                        created_at=datetime.fromisoformat(model_data["created_at"]),
                        metrics=model_data["metrics"],
                        feature_count=model_data["feature_count"],
                        model_path=Path(model_data["model_path"]),
                        metadata_path=Path(model_data["metadata_path"]) if model_data.get("metadata_path") else None,
                        description=model_data.get("description", ""),
                        tags=model_data.get("tags", [])
                    )
            except Exception as e:
                print(f"Ошибка при загрузке реестра: {e}")
                self.models = {}
    
    def _save_registry(self) -> None:
        """Сохраняет реестр в файл"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for model_name, model_info in self.models.items():
            data[model_name] = {
                "name": model_info.name,
                "model_type": model_info.model_type,
                "created_at": model_info.created_at.isoformat(),
                "metrics": model_info.metrics,
                "feature_count": model_info.feature_count,
                "model_path": str(model_info.model_path),
                "metadata_path": str(model_info.metadata_path) if model_info.metadata_path else None,
                "description": model_info.description,
                "tags": model_info.tags
            }
        
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        name: str,
        model_type: str,
        model_path: Path,
        metrics: Dict[str, float],
        feature_count: int,
        metadata_path: Optional[Path] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Регистрирует новую модель
        
        Args:
            name: Название модели
            model_type: Тип модели (например, "RandomForest")
            model_path: Путь к файлу модели
            metrics: Метрики модели
            feature_count: Количество признаков
            metadata_path: Путь к файлу метаданных
            description: Описание модели
            tags: Теги для категоризации
        """
        model_info = ModelInfo(
            name=name,
            model_type=model_type,
            created_at=datetime.now(),
            metrics=metrics,
            feature_count=feature_count,
            model_path=model_path,
            metadata_path=metadata_path,
            description=description,
            tags=tags or []
        )
        
        self.models[name] = model_info
        self._save_registry()
    
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """
        Получает информацию о модели
        
        Args:
            name: Название модели
            
        Returns:
            Информация о модели или None
        """
        return self.models.get(name)
    
    def list_models(
        self, 
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """
        Возвращает список моделей с возможной фильтрацией
        
        Args:
            model_type: Фильтр по типу модели
            tags: Фильтр по тегам
            
        Returns:
            Список информации о моделях
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if tags:
            models = [m for m in models if m.tags and any(tag in m.tags for tag in tags)]
        
        return models
    
    def get_best_model(self, metric: str = "Q_Error_Median", lower_is_better: bool = True) -> Optional[ModelInfo]:
        """
        Находит лучшую модель по указанной метрике
        
        Args:
            metric: Название метрики
            lower_is_better: Меньшее значение лучше?
            
        Returns:
            Информация о лучшей модели
        """
        models = list(self.models.values())
        if not models:
            return None
        
        valid_models = [m for m in models if metric in m.metrics]
        if not valid_models:
            return None
        
        if lower_is_better:
            return min(valid_models, key=lambda m: m.metrics[metric])
        else:
            return max(valid_models, key=lambda m: m.metrics[metric])
    
    def remove_model(self, name: str) -> bool:
        """
        Удаляет модель из реестра
        
        Args:
            name: Название модели
            
        Returns:
            True если модель была удалена
        """
        if name in self.models:
            del self.models[name]
            self._save_registry()
            return True
        return False
    
    def update_model_metrics(self, name: str, metrics: Dict[str, float]) -> bool:
        """
        Обновляет метрики модели
        
        Args:
            name: Название модели
            metrics: Новые метрики
            
        Returns:
            True если метрики были обновлены
        """
        if name in self.models:
            self.models[name].metrics.update(metrics)
            self._save_registry()
            return True
        return False
    
    def add_model_tags(self, name: str, tags: List[str]) -> bool:
        """
        Добавляет теги к модели
        
        Args:
            name: Название модели
            tags: Теги для добавления
            
        Returns:
            True если теги были добавлены
        """
        if name in self.models:
            existing_tags = set(self.models[name].tags or [])
            new_tags = existing_tags.union(set(tags))
            self.models[name].tags = list(new_tags)
            self._save_registry()
            return True
        return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по всем моделям
        
        Returns:
            Словарь со статистикой по моделям
        """
        if not self.models:
            return {"total_models": 0}
        
        model_types = {}
        total_models = len(self.models)
        latest_model = max(self.models.values(), key=lambda m: m.created_at)
        
        for model in self.models.values():
            model_type = model.model_type
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
        
        return {
            "total_models": total_models,
            "model_types": model_types,
            "latest_model": {
                "name": latest_model.name,
                "created_at": latest_model.created_at.isoformat(),
                "model_type": latest_model.model_type
            }
        }