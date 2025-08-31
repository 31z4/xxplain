"""
Feature extractor for PostgreSQL query plans
"""

from typing import Dict, Any, Optional, List, Union
from collections import defaultdict
import json

from .config import FeatureConfig, DEFAULT_CONFIG


class FeatureExtractor:
    """Извлекает признаки из PostgreSQL query plans"""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Инициализация экстрактора
        
        Args:
            config: Конфигурация для извлечения признаков
        """
        self.config = config or DEFAULT_CONFIG
    
    def extract_features(
        self,
        plan_json: Union[Dict[str, Any], List[Dict[str, Any]]],
        query_text: str = "",
        server_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Извлекает признаки из query plan
        
        Args:
            plan_json: JSON представление плана из EXPLAIN
            query_text: Текст SQL запроса
            server_params: Параметры сервера PostgreSQL
            
        Returns:
            Словарь с признаками
        """
        # Получаем корневой узел плана
        if isinstance(plan_json, list) and len(plan_json) > 0:
            first_item = plan_json[0]
            if isinstance(first_item, dict):
                plan_root = first_item.get("Plan", {})
            else:
                raise ValueError("Неверный формат элемента в списке plan_json")
        elif isinstance(plan_json, dict):
            plan_root = plan_json.get("Plan", plan_json)
        else:
            raise ValueError("Неверный формат plan_json")
        
        features = defaultdict(float)
        
        # Инициализируем базовые счетчики
        self._initialize_features(features)
        
        # Обходим план и собираем статистики
        self._traverse_plan(plan_root, features, depth=0)
        
        # Добавляем признаки корневого узла
        if self.config.extract_cost_estimates:
            self._extract_root_features(plan_root, features)
        
        # Вычисляем производные признаки
        self._compute_derived_features(features)
        
        # Добавляем признаки запроса
        if self.config.extract_query_features and query_text:
            self._extract_query_features(query_text, features)
        
        # Добавляем параметры сервера
        if self.config.extract_pg_config and server_params:
            self._extract_server_params(server_params, features)
        
        # Нормализация
        if self.config.normalize_costs or self.config.normalize_rows:
            self._normalize_features(features)
        
        # Добавляем префикс если нужно
        if self.config.feature_prefix:
            features = {f"{self.config.feature_prefix}{k}": v for k, v in features.items()}
        
        return dict(features)
    
    def _initialize_features(self, features: defaultdict) -> None:
        """Инициализирует базовые признаки"""
        if self.config.extract_basic_stats:
            features["n_nodes"] = 0.0
            features["max_depth"] = 0.0
        
        if self.config.extract_cost_estimates:
            features["sum_est_startup_cost"] = 0.0
            features["sum_est_total_cost"] = 0.0
            features["sum_plan_rows"] = 0.0
            features["sum_plan_width"] = 0.0
            features["max_est_total_cost"] = 0.0
            features["max_plan_rows"] = 0.0
            features["max_plan_width"] = 0.0
        
        if self.config.extract_parallel_info:
            features["n_parallel_aware"] = 0.0
            features["n_gather"] = 0.0
            features["sum_workers_planned"] = 0.0
        
        if self.config.extract_aggregation_info:
            features["n_sorts"] = 0.0
            features["n_aggregates"] = 0.0
            features["n_window"] = 0.0
            features["has_limit"] = 0.0
        
        if self.config.extract_filter_info:
            features["n_filters"] = 0.0
            features["sum_filter_len"] = 0.0
    
    def _traverse_plan(self, node: Dict[str, Any], features: defaultdict, depth: int = 0) -> None:
        """Рекурсивно обходит узлы плана и собирает статистики"""
        # Проверяем ограничения по глубине
        if self.config.max_depth is not None and depth > self.config.max_depth:
            return
        if depth < self.config.min_depth:
            return
        
        node_type = node.get("Node Type", "Unknown")
        node_key = self._sanitize_key(node_type)
        
        # Базовые статистики
        if self.config.extract_basic_stats:
            features["n_nodes"] += 1
            features["max_depth"] = max(features["max_depth"], depth)
        
        # Подсчет по типам узлов
        if self.config.extract_node_counts:
            if self.config.active_node_types is None or node_key in self.config.active_node_types:
                features[f"count_node_{node_key}"] += 1
        
        # Оценки стоимости и размеров
        if self.config.extract_cost_estimates:
            startup_cost = node.get("Startup Cost", 0.0) or 0.0
            total_cost = node.get("Total Cost", 0.0) or 0.0
            plan_rows = node.get("Plan Rows", 0) or 0
            plan_width = node.get("Plan Width", 0) or 0
            
            features["sum_est_startup_cost"] += startup_cost
            features["sum_est_total_cost"] += total_cost
            features["sum_plan_rows"] += plan_rows
            features["sum_plan_width"] += plan_width
            
            features["max_est_total_cost"] = max(features["max_est_total_cost"], total_cost)
            features["max_plan_rows"] = max(features["max_plan_rows"], plan_rows)
            features["max_plan_width"] = max(features["max_plan_width"], plan_width)
        
        # Информация о джойнах
        if self.config.extract_join_info:
            join_type = node.get("Join Type")
            if join_type:
                join_key = self._sanitize_key(join_type)
                if self.config.active_join_types is None or join_type in self.config.active_join_types:
                    features[f"count_join_{join_key}"] += 1
        
        # Параллельные операции
        if self.config.extract_parallel_info:
            if node.get("Parallel Aware", False):
                features["n_parallel_aware"] += 1
            
            if node_type in ("Gather", "Gather Merge"):
                features["n_gather"] += 1
                features["sum_workers_planned"] += node.get("Workers Planned", 0) or 0
        
        # Агрегация и сортировка
        if self.config.extract_aggregation_info:
            if "Sort Key" in node or node_type == "Sort":
                features["n_sorts"] += 1
            
            if node_type in ("Aggregate", "HashAggregate", "GroupAggregate"):
                features["n_aggregates"] += 1
            
            if node_type in ("WindowAgg",):
                features["n_window"] += 1
            
            if node_type in ("Limit",):
                features["has_limit"] = 1
        
        # Фильтры
        if self.config.extract_filter_info:
            if "Filter" in node:
                features["n_filters"] += 1
                try:
                    features["sum_filter_len"] += len(str(node.get("Filter", "")))
                except Exception:
                    pass
        
        # Рекурсивно обрабатываем дочерние узлы
        for child in node.get("Plans", []) or []:
            self._traverse_plan(child, features, depth + 1)
    
    def _extract_root_features(self, plan_root: Dict[str, Any], features: defaultdict) -> None:
        """Извлекает признаки корневого узла"""
        features["root_total_cost"] = plan_root.get("Total Cost", 0.0) or 0.0
        features["root_startup_cost"] = plan_root.get("Startup Cost", 0.0) or 0.0
        features["root_plan_rows"] = plan_root.get("Plan Rows", 0.0) or 0.0
        features["root_plan_width"] = plan_root.get("Plan Width", 0.0) or 0.0
    
    def _compute_derived_features(self, features: defaultdict) -> None:
        """Вычисляет производные признаки"""
        if self.config.extract_cost_estimates:
            # Отношение startup к total cost
            features["ratio_startup_total_cost"] = (
                features["root_startup_cost"] / (features["root_total_cost"] + 1e-9)
            )
            
            # Средние значения на узел
            if features["n_nodes"] > 0:
                features["avg_est_total_cost_per_node"] = (
                    features["sum_est_total_cost"] / features["n_nodes"]
                )
                features["avg_plan_rows_per_node"] = features["sum_plan_rows"] / features["n_nodes"]
                features["avg_plan_width_per_node"] = features["sum_plan_width"] / features["n_nodes"]
    
    def _extract_query_features(self, query_text: str, features: defaultdict) -> None:
        """Извлекает признаки из текста запроса"""
        features["query_length"] = len(query_text)
        features["query_tokens"] = len(query_text.split())
    
    def _extract_server_params(self, server_params: Dict[str, Any], features: defaultdict) -> None:
        """Извлекает признаки из параметров сервера"""
        for key, value in server_params.items():
            if isinstance(value, (int, float)):
                features[f"pgconf_{self._sanitize_key(key)}"] = float(value)
    
    def _normalize_features(self, features: defaultdict) -> None:
        """Нормализует признаки"""
        if self.config.normalize_costs and features["max_est_total_cost"] > 0:
            cost_features = ["sum_est_startup_cost", "sum_est_total_cost", "root_total_cost", "root_startup_cost"]
            max_cost = features["max_est_total_cost"]
            for feature in cost_features:
                if feature in features:
                    features[f"{feature}_normalized"] = features[feature] / max_cost
        
        if self.config.normalize_rows and features["max_plan_rows"] > 0:
            row_features = ["sum_plan_rows", "root_plan_rows"]
            max_rows = features["max_plan_rows"]
            for feature in row_features:
                if feature in features:
                    features[f"{feature}_normalized"] = features[feature] / max_rows
    
    @staticmethod
    def _sanitize_key(s: str) -> str:
        """Очищает ключ от специальных символов"""
        return s.replace(" ", "_").replace("/", "_").replace("-", "_")
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список имен признаков, которые извлекает данная конфигурация"""
        # Создаем пустой план для получения списка признаков
        dummy_plan = {"Plan": {"Node Type": "Result", "Total Cost": 1.0, "Startup Cost": 1.0, "Plan Rows": 1, "Plan Width": 1}}
        dummy_features = self.extract_features(dummy_plan, "", {})
        return list(dummy_features.keys())
