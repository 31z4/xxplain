"""
Experiment management for model development
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import pickle

from ..evaluation import ModelEvaluator
from ..models import ModelManager


class Experiment:
    """Класс для управления экспериментами с моделями"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        workspace_dir: str = "experiments",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация эксперимента
        
        Args:
            name: Название эксперимента
            description: Описание эксперимента
            workspace_dir: Директория для сохранения результатов
            config: Конфигурация эксперимента
        """
        self.name = name
        self.description = description
        self.workspace_dir = Path(workspace_dir)
        self.experiment_dir = self.workspace_dir / name
        self.config = config or {}
        
        # Создаем директории
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        
        # Метаданные эксперимента
        self.metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "config": self.config,
            "runs": []
        }
        
        # Компоненты
        self.evaluator = ModelEvaluator()
        self.model_manager: Optional[ModelManager] = None
        
        # Загружаем существующие метаданные если есть
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Загружает метаданные эксперимента"""
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    stored_metadata = json.load(f)
                    # Обновляем только runs, сохраняя текущие настройки
                    self.metadata["runs"] = stored_metadata.get("runs", [])
            except Exception as e:
                print(f"Ошибка загрузки метаданных: {e}")
    
    def _save_metadata(self) -> None:
        """Сохраняет метаданные эксперимента"""
        metadata_path = self.experiment_dir / "metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения метаданных: {e}")
    
    def add_run(
        self,
        run_name: str,
        model_type: str,
        results: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> str:
        """
        Добавляет результат запуска в эксперимент
        
        Args:
            run_name: Название запуска
            model_type: Тип модели
            results: Результаты запуска
            config: Конфигурация запуска
            notes: Заметки
            
        Returns:
            ID запуска
        """
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_data = {
            "run_id": run_id,
            "run_name": run_name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "config": config or {},
            "notes": notes
        }
        
        self.metadata["runs"].append(run_data)
        self._save_metadata()
        
        # Сохраняем детальные результаты
        results_path = self.experiment_dir / "results" / f"{run_id}.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(run_data, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения результатов: {e}")
        
        return run_id
    
    def get_runs(
        self,
        model_type: Optional[str] = None,
        sort_by: str = "timestamp",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список запусков с фильтрацией
        
        Args:
            model_type: Фильтр по типу модели
            sort_by: Поле для сортировки
            ascending: Порядок сортировки
            
        Returns:
            Список запусков
        """
        runs = self.metadata["runs"].copy()
        
        # Фильтрация
        if model_type:
            runs = [r for r in runs if r.get("model_type") == model_type]
        
        # Сортировка
        try:
            runs.sort(
                key=lambda x: x.get(sort_by, ""),
                reverse=not ascending
            )
        except Exception:
            pass
        
        return runs
    
    def get_best_run(
        self,
        metric: str = "Q_Error_Median",
        model_type: Optional[str] = None,
        lower_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Находит лучший запуск по метрике
        
        Args:
            metric: Метрика для сравнения
            model_type: Фильтр по типу модели
            lower_is_better: Меньшее значение лучше?
            
        Returns:
            Лучший запуск или None
        """
        runs = self.get_runs(model_type=model_type)
        
        valid_runs = []
        for run in runs:
            results = run.get("results", {})
            if metric in results and not pd.isna(results[metric]):
                valid_runs.append(run)
        
        if not valid_runs:
            return None
        
        if lower_is_better:
            return min(valid_runs, key=lambda x: x["results"][metric])
        else:
            return max(valid_runs, key=lambda x: x["results"][metric])
    
    def compare_runs(
        self,
        run_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Сравнивает запуски в табличном виде
        
        Args:
            run_ids: Список ID запусков (если None, берет все)
            metrics: Список метрик для сравнения
            
        Returns:
            DataFrame со сравнением
        """
        if metrics is None:
            metrics = ["MAE", "RMSE", "Q_Error_Median", "Q_Error_95p", "R2"]
        
        runs = self.metadata["runs"]
        if run_ids:
            runs = [r for r in runs if r["run_id"] in run_ids]
        
        comparison_data = []
        for run in runs:
            row = {
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "model_type": run["model_type"],
                "timestamp": run["timestamp"]
            }
            
            results = run.get("results", {})
            for metric in metrics:
                row[metric] = results.get(metric, None)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Сохраняет модель в эксперимент
        
        Args:
            model: Объект модели
            model_name: Название модели
            model_type: Тип модели
            metadata: Дополнительные метаданные
            
        Returns:
            Путь к сохраненной модели
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.experiment_dir / "models" / model_filename
        
        try:
            # Сохраняем модель
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Сохраняем метаданные модели
            model_metadata = {
                "model_name": model_name,
                "model_type": model_type,
                "saved_at": datetime.now().isoformat(),
                "file_path": str(model_path),
                "metadata": metadata or {}
            }
            
            metadata_path = self.experiment_dir / "models" / f"{model_name}_{timestamp}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            print(f"Модель сохранена: {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"Ошибка сохранения модели: {e}")
            raise
    
    def load_model(self, model_path: str) -> Any:
        """
        Загружает модель из файла
        
        Args:
            model_path: Путь к файлу модели
            
        Returns:
            Загруженная модель
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по эксперименту"""
        runs = self.metadata["runs"]
        
        if not runs:
            return {
                "name": self.name,
                "description": self.description,
                "total_runs": 0,
                "created_at": self.metadata["created_at"]
            }
        
        # Статистика по типам моделей
        model_types = {}
        for run in runs:
            model_type = run.get("model_type", "Unknown")
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Лучший результат
        best_run = self.get_best_run()
        
        return {
            "name": self.name,
            "description": self.description,
            "total_runs": len(runs),
            "model_types": model_types,
            "created_at": self.metadata["created_at"],
            "last_run": runs[-1]["timestamp"] if runs else None,
            "best_run": {
                "run_id": best_run["run_id"],
                "model_type": best_run["model_type"],
                "Q_Error_Median": best_run["results"].get("Q_Error_Median", "N/A")
            } if best_run else None
        }
    
    def export_results(self, format: str = "csv") -> str:
        """
        Экспортирует результаты в файл
        
        Args:
            format: Формат файла ("csv", "json", "excel")
            
        Returns:
            Путь к экспортированному файлу
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            df = self.compare_runs()
            export_path = self.experiment_dir / f"results_export_{timestamp}.csv"
            df.to_csv(export_path, index=False)
        
        elif format == "json":
            export_path = self.experiment_dir / f"results_export_{timestamp}.json"
            with open(export_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        elif format == "excel":
            df = self.compare_runs()
            export_path = self.experiment_dir / f"results_export_{timestamp}.xlsx"
            df.to_excel(export_path, index=False)
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
        
        print(f"Результаты экспортированы: {export_path}")
        return str(export_path)
    
    def cleanup_old_runs(self, keep_latest: int = 10) -> None:
        """
        Удаляет старые запуски, оставляя только последние
        
        Args:
            keep_latest: Количество последних запусков для сохранения
        """
        runs = sorted(
            self.metadata["runs"],
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        if len(runs) <= keep_latest:
            return
        
        # Удаляем старые запуски
        runs_to_remove = runs[keep_latest:]
        
        for run in runs_to_remove:
            # Удаляем файл с результатами
            results_path = self.experiment_dir / "results" / f"{run['run_id']}.json"
            if results_path.exists():
                results_path.unlink()
        
        # Обновляем метаданные
        self.metadata["runs"] = runs[:keep_latest]
        self._save_metadata()
        
        print(f"Удалено {len(runs_to_remove)} старых запусков")