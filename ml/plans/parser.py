"""
Parser for PostgreSQL query plans
"""

from typing import Dict, Any, List, Optional, Union
import json


class PlanParser:
    """Парсер для обработки и нормализации query plans"""
    
    @staticmethod
    def parse_explain_result(explain_result: Any) -> Dict[str, Any]:
        """
        Парсит результат EXPLAIN и возвращает нормализованный план
        
        Args:
            explain_result: Результат выполнения EXPLAIN (может быть строкой JSON, списком или словарем)
            
        Returns:
            Нормализованный словарь с планом
        """
        if isinstance(explain_result, str):
            try:
                parsed = json.loads(explain_result)
            except json.JSONDecodeError as e:
                raise ValueError(f"Не удалось распарсить JSON: {e}")
        else:
            parsed = explain_result
        
        # Если результат - список, берем первый элемент
        if isinstance(parsed, list) and len(parsed) > 0:
            plan_dict = parsed[0]
        elif isinstance(parsed, dict):
            plan_dict = parsed
        else:
            raise ValueError("Неожиданный формат результата EXPLAIN")
        
        # Проверяем наличие ключа "Plan"
        if "Plan" not in plan_dict:
            raise ValueError("Отсутствует ключ 'Plan' в результате EXPLAIN")
        
        return plan_dict
    
    @staticmethod
    def extract_server_params(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлекает параметры сервера из плана
        
        Args:
            plan_dict: Словарь с планом
            
        Returns:
            Словарь с параметрами сервера
        """
        settings = plan_dict.get("Settings", {})
        if not settings:
            # Попробуем найти в других местах
            settings = plan_dict.get("Query", {}).get("Settings", {})
        
        # Преобразуем значения в числовые, где возможно
        normalized_settings = {}
        for key, value in settings.items():
            normalized_settings[key] = PlanParser._normalize_setting_value(value)
        
        return normalized_settings
    
    @staticmethod
    def _normalize_setting_value(value: str) -> Union[float, str]:
        """
        Нормализует значение настройки PostgreSQL
        
        Args:
            value: Строковое значение настройки
            
        Returns:
            Нормализованное значение (число или строка)
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if not isinstance(value, str):
            return str(value)
        
        # Убираем единицы измерения и преобразуем в число
        try:
            # Обрабатываем размеры памяти (kB, MB, GB)
            if value.endswith("kB"):
                return float(value[:-2].strip())
            elif value.endswith("MB"):
                return float(value[:-2].strip()) * 1024
            elif value.endswith("GB"):
                return float(value[:-2].strip()) * 1024 * 1024
            
            # Обрабатываем булевые значения
            if value.lower() in ("on", "true", "yes"):
                return 1.0
            elif value.lower() in ("off", "false", "no"):
                return 0.0
            
            # Пробуем преобразовать в число
            return float(value)
        except (ValueError, AttributeError):
            # Если не удалось преобразовать, возвращаем как строку
            return value
    
    @staticmethod
    def validate_plan(plan_dict: Dict[str, Any]) -> bool:
        """
        Проверяет корректность структуры плана
        
        Args:
            plan_dict: Словарь с планом
            
        Returns:
            True если план корректный, False иначе
        """
        try:
            # Проверяем наличие основных ключей
            if "Plan" not in plan_dict:
                return False
            
            plan = plan_dict["Plan"]
            
            # Проверяем обязательные поля корневого узла
            required_fields = ["Node Type"]
            for field in required_fields:
                if field not in plan:
                    return False
            
            # Рекурсивно проверяем дочерние узлы
            return PlanParser._validate_node(plan)
        
        except Exception:
            return False
    
    @staticmethod
    def _validate_node(node: Dict[str, Any]) -> bool:
        """
        Рекурсивно проверяет корректность узла плана
        
        Args:
            node: Узел плана
            
        Returns:
            True если узел корректный, False иначе
        """
        # Проверяем тип узла
        if "Node Type" not in node:
            return False
        
        # Проверяем дочерние узлы
        plans = node.get("Plans", [])
        if plans:
            for child_plan in plans:
                if not PlanParser._validate_node(child_plan):
                    return False
        
        return True
    
    @staticmethod
    def get_plan_summary(plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Возвращает краткую сводку по плану
        
        Args:
            plan_dict: Словарь с планом
            
        Returns:
            Словарь с краткой информацией о плане
        """
        plan = plan_dict["Plan"]
        
        summary = {
            "root_node_type": plan.get("Node Type"),
            "total_cost": plan.get("Total Cost"),
            "startup_cost": plan.get("Startup Cost"),
            "plan_rows": plan.get("Plan Rows"),
            "plan_width": plan.get("Plan Width"),
            "actual_time": plan.get("Actual Total Time"),
            "actual_rows": plan.get("Actual Rows"),
        }
        
        # Подсчитываем общее количество узлов
        summary["total_nodes"] = PlanParser._count_nodes(plan)
        
        # Максимальная глубина плана
        summary["max_depth"] = PlanParser._get_max_depth(plan)
        
        return summary
    
    @staticmethod
    def _count_nodes(node: Dict[str, Any]) -> int:
        """Подсчитывает общее количество узлов в плане"""
        count = 1
        for child in node.get("Plans", []):
            count += PlanParser._count_nodes(child)
        return count
    
    @staticmethod
    def _get_max_depth(node: Dict[str, Any], current_depth: int = 0) -> int:
        """Определяет максимальную глубину плана"""
        max_depth = current_depth
        for child in node.get("Plans", []):
            child_depth = PlanParser._get_max_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth