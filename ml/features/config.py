"""
Configuration for feature extraction
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """Конфигурация для извлечения признаков из query plans"""
    
    # Базовые признаки плана
    extract_basic_stats: bool = True
    extract_node_counts: bool = True
    extract_join_info: bool = True
    extract_cost_estimates: bool = True
    
    # Признаки запроса
    extract_query_features: bool = True
    
    # Признаки конфигурации PostgreSQL
    extract_pg_config: bool = True
    
    # Дополнительные признаки
    extract_parallel_info: bool = True
    extract_filter_info: bool = True
    extract_aggregation_info: bool = True
    
    # Список активных типов узлов для подсчета
    active_node_types: Optional[List[str]] = None
    
    # Список активных типов джойнов
    active_join_types: Optional[List[str]] = None
    
    # Минимальная глубина плана для извлечения признаков
    min_depth: int = 0
    
    # Максимальная глубина плана для извлечения признаков (None = без ограничений)
    max_depth: Optional[int] = None
    
    # Префикс для добавления к именам признаков
    feature_prefix: str = ""
    
    # Нормализация признаков
    normalize_costs: bool = False
    normalize_rows: bool = False
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if self.active_node_types is None:
            self.active_node_types = [
                "Seq_Scan", "Index_Scan", "Index_Only_Scan", "Bitmap_Index_Scan",
                "Bitmap_Heap_Scan", "Nested_Loop", "Hash_Join", "Merge_Join",
                "Hash", "Sort", "Aggregate", "Limit", "Gather", "Gather_Merge",
                "Materialize", "Memoize", "Result", "Unique", "BitmapAnd",
                "Subquery_Scan", "WindowAgg", "Incremental_Sort"
            ]
        
        if self.active_join_types is None:
            self.active_join_types = ["Inner", "Left", "Right", "Full", "Semi", "Anti"]


# Предопределенные конфигурации
DEFAULT_CONFIG = FeatureConfig()

MINIMAL_CONFIG = FeatureConfig(
    extract_basic_stats=True,
    extract_node_counts=False,
    extract_join_info=False,
    extract_cost_estimates=True,
    extract_query_features=False,
    extract_pg_config=False,
    extract_parallel_info=False,
    extract_filter_info=False,
    extract_aggregation_info=False
)

FULL_CONFIG = FeatureConfig(
    extract_basic_stats=True,
    extract_node_counts=True,
    extract_join_info=True,
    extract_cost_estimates=True,
    extract_query_features=True,
    extract_pg_config=True,
    extract_parallel_info=True,
    extract_filter_info=True,
    extract_aggregation_info=True,
    normalize_costs=True,
    normalize_rows=True
)