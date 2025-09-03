"""
Dataset builder for model training
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv
from collections import defaultdict

from ..plans.service import SyncQueryPlanService
from ..plans import PlanParser
from ..features import FeatureExtractor, FeatureConfig


class DatasetBuilder:
    """Класс для построения датасетов из query plans"""
    
    def __init__(
        self,
        dsn: Optional[str] = None,
        feature_config: Optional[FeatureConfig] = None
    ):
        """
        Инициализация билдера
        
        Args:
            dsn: Строка подключения к PostgreSQL
            feature_config: Конфигурация для извлечения признаков
        """
        self.dsn = dsn
        self.feature_config = feature_config
        self.feature_extractor = FeatureExtractor(feature_config)
        self.plan_parser = PlanParser()
        
        if dsn:
            self.plan_service = SyncQueryPlanService(dsn)
        else:
            self.plan_service = None
    
    def build_from_queries(
        self,
        queries: List[str],
        output_path: Optional[str] = None,
        analyze: bool = True,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Строит датасет из списка SQL запросов
        
        Args:
            queries: Список SQL запросов
            output_path: Путь для сохранения (опционально)
            analyze: Выполнять ANALYZE для получения фактического времени
            timeout: Таймаут для запросов в секундах
            
        Returns:
            Список записей датасета
        """
        if not self.plan_service:
            raise ValueError("Не указана строка подключения к БД")
        
        dataset = []
        
        print(f"Обработка {len(queries)} запросов...")
        
        for i, query in enumerate(queries):
            try:
                # Получаем план с параметрами сервера
                plan_dict, server_params = self.plan_service.get_plan_with_server_params(
                    query, analyze=analyze, timeout=timeout
                )
                
                # Извлекаем признаки
                features = self.feature_extractor.extract_features(
                    plan_dict, query, server_params
                )
                
                # Получаем целевую переменную (если analyze=True)
                target = None
                if analyze:
                    target = plan_dict.get("Plan", {}).get("Actual Total Time")
                
                # Создаем запись
                record = {
                    "query": query,
                    "plan": plan_dict,
                    "features": features,
                    "target": target,
                    "server_params": server_params
                }
                
                dataset.append(record)
                
                if (i + 1) % 10 == 0:
                    print(f"  Обработано {i + 1}/{len(queries)}")
                
            except Exception as e:
                print(f"  Ошибка обработки запроса {i + 1}: {e}")
                continue
        
        print(f"Успешно обработано {len(dataset)} запросов")
        
        # Сохраняем датасет если указан путь
        if output_path:
            self.save_dataset(dataset, output_path)
        
        return dataset
    
    def build_from_files(
        self,
        query_files: List[str],
        output_path: Optional[str] = None,
        analyze: bool = True,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Строит датасет из файлов с SQL запросами
        
        Args:
            query_files: Список путей к файлам с запросами
            output_path: Путь для сохранения
            analyze: Выполнять ANALYZE
            timeout: Таймаут для запросов
            
        Returns:
            Список записей датасета
        """
        all_queries = []
        
        for file_path in query_files:
            try:
                queries = self._read_queries_from_file(file_path)
                all_queries.extend(queries)
                print(f"Загружено {len(queries)} запросов из {file_path}")
            except Exception as e:
                print(f"Ошибка чтения файла {file_path}: {e}")
        
        print(f"Всего загружено {len(all_queries)} запросов")
        
        return self.build_from_queries(all_queries, output_path, analyze, timeout)
    
    def load_dataset_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Загружает датасет из CSV файла (формат как в lab/)
        
        Args:
            csv_path: Путь к CSV файлу
            
        Returns:
            Список записей датасета
        """
        dataset = []
        
        try:
            df = pd.read_csv(csv_path, delimiter='\t')
            
            for _, row in df.iterrows():
                try:
                    # Парсим признаки из строки
                    features_str = row.get('feats', '{}')
                    if isinstance(features_str, str):
                        features = json.loads(features_str.replace("'", '"'))
                    else:
                        features = {}
                    
                    # Парсим план
                    plan_str = row.get('plan', '{}')
                    if isinstance(plan_str, str):
                        plan = json.loads(plan_str.replace("'", '"'))
                    else:
                        plan = {}
                    
                    record = {
                        "query": row.get('query', ''),
                        "plan": plan,
                        "features": features,
                        "target": pd.to_numeric(row.get('target', np.nan), errors='coerce'),
                        "server_params": {}
                    }
                    
                    # Пропускаем записи с некорректными целевыми значениями
                    if pd.isna(record["target"]):
                        continue
                    
                    dataset.append(record)
                    
                except Exception as e:
                    print(f"Ошибка обработки строки: {e}")
                    continue
            
            print(f"Загружено {len(dataset)} записей из {csv_path}")
            
        except Exception as e:
            print(f"Ошибка чтения CSV файла: {e}")
            raise
        
        return dataset
    
    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Сохраняет датасет в файл
        
        Args:
            dataset: Датасет для сохранения
            output_path: Путь для сохранения
            format: Формат файла ("json", "csv", "parquet")
        """
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
        
        elif format == "csv":
            # Сохраняем в формате как в lab/
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, 
                    fieldnames=["query", "plan", "features", "target"],
                    delimiter='\t'
                )
                writer.writeheader()
                
                for record in dataset:
                    csv_record = {
                        "query": record.get("query", ""),
                        "plan": json.dumps(record.get("plan", {})),
                        "features": json.dumps(record.get("features", {})),
                        "target": record.get("target")
                    }
                    writer.writerow(csv_record)
        
        elif format == "parquet":
            # Преобразуем в DataFrame и сохраняем
            df_data = []
            for record in dataset:
                df_data.append({
                    "query": record.get("query", ""),
                    "plan": json.dumps(record.get("plan", {})),
                    "features": json.dumps(record.get("features", {})),
                    "target": record.get("target"),
                    "server_params": json.dumps(record.get("server_params", {}))
                })
            
            df = pd.DataFrame(df_data)
            df.to_parquet(file_path, index=False)
        
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
        
        print(f"Датасет сохранен: {file_path}")

    def get_dataset_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Возвращает статистику по датасету
        
        Args:
            dataset: Датасет для анализа
            
        Returns:
            Словарь со статистикой
        """
        if not dataset:
            return {"error": "Пустой датасет"}
        
        # Базовая статистика
        total_records = len(dataset)
        records_with_target = sum(1 for r in dataset if r.get("target") is not None)
        
        # Статистика по целевой переменной
        targets = [r.get("target") for r in dataset if r.get("target") is not None]
        targets = [t for t in targets if not pd.isna(t)]
        
        target_stats = {}
        if targets:
            target_stats = {
                "count": len(targets),
                "mean": float(np.mean(targets)),
                "std": float(np.std(targets)),
                "min": float(np.min(targets)),
                "max": float(np.max(targets)),
                "median": float(np.median(targets)),
                "q25": float(np.percentile(targets, 25)),
                "q75": float(np.percentile(targets, 75))
            }
        
        # Статистика по признакам
        all_features = []
        for record in dataset:
            features = record.get("features", {})
            if features:
                all_features.append(features)
        
        feature_stats = {}
        if all_features:
            # Объединяем все признаки
            all_feature_names = set()
            for features in all_features:
                all_feature_names.update(features.keys())
            
            feature_stats = {
                "total_features": len(all_feature_names),
                "avg_features_per_record": float(np.mean([len(f) for f in all_features]))
            }
        
        # Статистика по запросам
        query_lengths = [len(r.get("query", "")) for r in dataset]
        query_stats = {
            "avg_query_length": float(np.mean(query_lengths)) if query_lengths else 0,
            "min_query_length": int(np.min(query_lengths)) if query_lengths else 0,
            "max_query_length": int(np.max(query_lengths)) if query_lengths else 0
        }
        
        return {
            "total_records": total_records,
            "records_with_target": records_with_target,
            "target_coverage": records_with_target / total_records if total_records > 0 else 0,
            "target_stats": target_stats,
            "feature_stats": feature_stats,
            "query_stats": query_stats
        }
    
    def _read_queries_from_file(self, file_path: str) -> List[str]:
        """Читает запросы из файла"""
        path_obj = Path(file_path)
        queries = []
        
        if path_obj.suffix == '.sql':
            # Один запрос на файл
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content and not content.startswith('--'):
                    queries.append(content)
        
        elif path_obj.suffix == '.txt':
            # Один запрос на строку
            with open(path_obj, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('--'):
                        queries.append(line)
        
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {path_obj.suffix}")
        
        return queries
