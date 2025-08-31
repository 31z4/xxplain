"""
Service for obtaining PostgreSQL query plans
"""

from .parser import PlanParser
import psycopg
from psycopg import sql
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, Tuple


def _build_explain_sql(
    query: str,
    analyze: bool,
    format_json: bool,
    include_buffers: bool,
    include_settings: bool,
    include_timing: bool
) -> str:
    """Строит SQL команду EXPLAIN с нужными опциями"""
    options = []
    
    if analyze:
        options.append("ANALYZE")
    
    if include_buffers:
        options.append("BUFFERS")
    
    if include_settings:
        options.append("SETTINGS")
    
    if include_timing and analyze:
        options.append("TIMING ON")
    
    if format_json:
        options.append("FORMAT JSON")
    
    options_str = ", ".join(options)
    
    # Простая защита от SQL инъекций
    if ";" in query.rstrip(";"):
        raise ValueError("Запрос содержит несколько statement'ов")
    
    return f"EXPLAIN ({options_str}) {query.rstrip(';')}"


def _process_plan_result(parser: PlanParser, result) -> Dict[str, Any]:
    """Обрабатывает результат EXPLAIN и возвращает нормализованный план"""
    if result is None:
        raise ValueError("EXPLAIN не вернул результат")
    
    # Результат может быть кортежем, берем первый элемент
    plan_data = result[0] if isinstance(result, (tuple, list)) else result
    
    # Парсим и нормализуем план
    normalized_plan = parser.parse_explain_result(plan_data)
    
    return normalized_plan


def _get_server_param_names() -> list[str]:
    """Возвращает список имен параметров сервера для получения через SHOW"""
    return [
        "random_page_cost",
        "seq_page_cost",
        "cpu_tuple_cost",
        "cpu_index_tuple_cost",
        "cpu_operator_cost",
        "effective_cache_size",
        "work_mem",
        "max_parallel_workers_per_gather",
        "jit",
        "shared_buffers",
        "maintenance_work_mem"
    ]


def _get_additional_server_params(parser: PlanParser, dsn: str) -> Dict[str, Any]:
    """Получает дополнительные параметры сервера через SHOW"""
    params = {}
    param_names = _get_server_param_names()
    
    # Используем синхронное подключение для получения параметров
    conn = psycopg.connect(dsn)
    try:
        with conn.cursor() as cur:
            for param_name in param_names:
                try:
                    # Используем psycopg.sql для безопасного форматирования
                    query = sql.SQL("SHOW {}").format(sql.Identifier(param_name))
                    # Просто выполняем SQL строку напрямую
                    cur.execute(query)
                    result = cur.fetchone()
                    if result:
                        value = result[0]
                        params[param_name] = parser._normalize_setting_value(value)
                except Exception:
                    # Если параметр не найден, пропускаем
                    continue
    finally:
        conn.close()
        
    return params


class QueryPlanService:
    """Сервис для получения и обработки query plans из PostgreSQL"""
    
    def __init__(self, dsn: str):
        """
        Инициализация сервиса
        
        Args:
            dsn: Строка подключения к PostgreSQL
        """
        self.dsn = dsn
        self.parser = PlanParser()
    
    async def get_plan(
        self,
        query: str,
        analyze: bool = False,
        format_json: bool = True,
        include_buffers: bool = True,
        include_settings: bool = True,
        include_timing: bool = True
    ) -> Dict[str, Any]:
        """
        Получает план выполнения запроса
        
        Args:
            query: SQL запрос
            analyze: Выполнить ANALYZE (фактическое выполнение)
            format_json: Получить результат в формате JSON
            include_buffers: Включить информацию о буферах
            include_settings: Включить настройки сервера
            include_timing: Включить информацию о времени
            
        Returns:
            Нормализованный план выполнения
        """
        explain_sql = _build_explain_sql(
            query, analyze, format_json, include_buffers, include_settings, include_timing
        )
        
        async with self._get_connection() as cur:
            await cur.execute(explain_sql)  # type: ignore
            result = await cur.fetchone()
            return _process_plan_result(self.parser, result)
    
    async def get_plan_with_server_params(
        self,
        query: str,
        analyze: bool = False
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Получает план выполнения вместе с параметрами сервера
        
        Args:
            query: SQL запрос
            analyze: Выполнить ANALYZE
            
        Returns:
            Кортеж (план, параметры_сервера)
        """
        # Получаем план с настройками
        plan = await self.get_plan(query, analyze=analyze, include_settings=True)
        
        # Извлекаем параметры сервера из плана
        server_params = self.parser.extract_server_params(plan)
        
        # Если параметров мало, получаем дополнительно через SHOW
        if len(server_params) < 5:
            additional_params = await self._get_additional_server_params()
            server_params.update(additional_params)
        
        return plan, server_params
    
    async def validate_query(self, query: str) -> bool:
        """
        Проверяет корректность SQL запроса без его выполнения
        
        Args:
            query: SQL запрос
            
        Returns:
            True если запрос корректный, False иначе
        """
        try:
            # Используем EXPLAIN без ANALYZE для проверки синтаксиса
            await self.get_plan(query, analyze=False)
            return True
        except Exception:
            return False
    
    @asynccontextmanager
    async def _get_connection(self):
        """Создает асинхронное соединение с автоматическим rollback"""
        conn = await psycopg.AsyncConnection.connect(self.dsn)
        try:
            async with conn.cursor() as cur:
                yield cur
        finally:
            # Автоматический rollback при закрытии соединения
            await conn.close()
    
    async def _get_additional_server_params(self) -> Dict[str, Any]:
        """Получает дополнительные параметры сервера через SHOW"""
        params = {}
        param_names = _get_server_param_names()
        
        async with self._get_connection() as cur:
            for param_name in param_names:
                try:
                    # Используем psycopg.sql для безопасного форматирования
                    query = sql.SQL("SHOW {}").format(sql.Identifier(param_name))
                    # Просто выполняем SQL строку напрямую
                    await cur.execute(query)
                    result = await cur.fetchone()
                    if result:
                        value = result[0]
                        params[param_name] = self.parser._normalize_setting_value(value)
                except Exception:
                    # Если параметр не найден, пропускаем
                    continue
        
        return params


class SyncQueryPlanService:
    """Синхронная версия сервиса для использования в экспериментах"""
    
    def __init__(self, dsn: str):
        """
        Инициализация сервиса
        
        Args:
            dsn: Строка подключения к PostgreSQL
        """
        self.dsn = dsn
        self.parser = PlanParser()
        self._connection = None
        self._current_timeout = None
    
    def get_plan(
        self,
        query: str,
        analyze: bool = False,
        format_json: bool = True,
        include_buffers: bool = True,
        include_settings: bool = True,
        include_timing: bool = True,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Синхронно получает план выполнения запроса
        
        Args:
            query: SQL запрос
            analyze: Выполнить ANALYZE (фактическое выполнение)
            format_json: Получить результат в формате JSON
            include_buffers: Включить информацию о буферах
            include_settings: Включить настройки сервера
            include_timing: Включить информацию о времени
            timeout: Таймаут для запроса в секундах
            
        Returns:
            Нормализованный план выполнения
        """
        explain_sql = _build_explain_sql(
            query, analyze, format_json, include_buffers, include_settings, include_timing
        )
        
        with self._get_connection_with_timeout(timeout) as (conn, cur):
            try:
                # Просто выполняем SQL строку напрямую
                cur.execute(explain_sql)
                result = cur.fetchone()
                
                if result is None:
                    raise ValueError("EXPLAIN не вернул результат")
                
                # Результат может быть кортежем, берем первый элемент
                plan_data = result[0] if isinstance(result, (tuple, list)) else result
                
                # Парсим и нормализуем план
                normalized_plan = self.parser.parse_explain_result(plan_data)
                
                return normalized_plan
            except Exception as e:
                # Автоматический rollback при ошибке
                conn.rollback()
                raise e
    
    def get_plan_with_server_params(
        self,
        query: str,
        analyze: bool = False,
        timeout: Optional[int] = None
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Получает план выполнения вместе с параметрами сервера
        
        Args:
            query: SQL запрос
            analyze: Выполнить ANALYZE
            timeout: Таймаут для запроса в секундах
            
        Returns:
            Кортеж (план, параметры_сервера)
        """
        # Получаем план с настройками
        plan = self.get_plan(query, analyze=analyze, include_settings=True, timeout=timeout)
        
        # Извлекаем параметры сервера из плана
        server_params = self.parser.extract_server_params(plan)
        
        # Если параметров мало, получаем дополнительно через SHOW
        if len(server_params) < 5:
            additional_params = self._get_additional_server_params()
            server_params.update(additional_params)
        
        return plan, server_params
    
    def _get_connection(self):
        """Создает синхронное соединение"""
        from contextlib import contextmanager
        
        @contextmanager
        def connection_context():
            conn = psycopg.connect(self.dsn)
            try:
                with conn.cursor() as cur:
                    yield cur
            finally:
                conn.close()
        
        return connection_context()
    
    def _get_connection_with_timeout(self, timeout: Optional[int] = None):
        """Создает синхронное соединение с поддержкой timeout"""
        from contextlib import contextmanager
        
        @contextmanager
        def connection_context():
            conn = psycopg.connect(self.dsn)
            try:
                with conn.cursor() as cur:
                    # Устанавливаем statement timeout если указан
                    if timeout is not None:
                        # Просто выполняем SQL строку напрямую
                        cur.execute(f"SET statement_timeout TO '{timeout}s'")
                    
                    yield conn, cur
            finally:
                conn.close()
        
        return connection_context()
    
    def _get_additional_server_params(self) -> Dict[str, Any]:
        """Получает дополнительные параметры сервера через SHOW"""
        return _get_additional_server_params(self.parser, self.dsn)