"""
Построение датасетов для обучения моделей с использованием xxplain классов.
Аналог lab/build_datasets.py с современной архитектурой.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from ml.experiments.dataset import DatasetBuilder
from ml.features.config import FeatureConfig, FULL_CONFIG


def setup_environment() -> str:
    """
    Настройка окружения и получение строки подключения к БД
    
    Returns:
        Строка подключения DSN
    """
    # Загружаем переменные окружения из .env.postgres
    dotenv_path = Path(".env.postgres")
    load_dotenv(dotenv_path=dotenv_path)
    
    # Формируем строку подключения
    dsn = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )
    
    return dsn


def read_queries_from_source(source_path: str) -> List[str]:
    """
    Читает SQL-запросы из файла или всех .sql файлов в директории.
    Совместима с оригинальной функцией из lab/build_datasets.py
    
    Args:
        source_path: Путь к файлу или директории с SQL запросами
        
    Returns:
        Список строк (запросов)
    """
    queries = []
    path_obj = Path(source_path)
    
    if path_obj.is_file():
        # Если это файл, читаем его
        with open(path_obj, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            queries = [q for q in lines if q and not q.startswith("--")]
    
    elif path_obj.is_dir():
        # Если это директория, читаем все .sql файлы
        sql_files = sorted(path_obj.glob("*.sql"))
        for file_path in sql_files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip().replace("\n", " ")
                if content and not content.startswith("--"):
                    queries.append(content)
    else:
        raise ValueError(f"Путь {source_path} не является файлом или директорией.")
    
    return queries


def create_feature_config() -> FeatureConfig:
    """
    Создает конфигурацию для извлечения признаков.
    Аналогична функционалу extract_features из оригинального скрипта.
    
    Returns:
        Конфигурация признаков
    """
    # Используем полную конфигурацию как основу и настраиваем под наши нужды
    config = FeatureConfig(
        # Базовые признаки (аналог original)
        extract_basic_stats=True,
        extract_node_counts=True,
        extract_join_info=True,
        extract_cost_estimates=True,
        
        # Признаки запроса
        extract_query_features=True,
        
        # Параметры сервера
        extract_pg_config=True,
        
        # Дополнительные признаки
        extract_parallel_info=True,
        extract_filter_info=True,
        extract_aggregation_info=True,
        
        # Нормализация отключена для совместимости с оригиналом
        normalize_costs=False,
        normalize_rows=False,
        
        # Префикс пустой
        feature_prefix=""
    )
    
    return config


def build_dataset_with_xxplain(
    queries: List[str],
    dsn: str,
    output_csv_path: str,
    timeout: int = 30,
    analyze: bool = True
) -> Dict[str, Any]:
    """
    Строит датасет используя xxplain классы
    
    Args:
        queries: Список SQL запросов
        dsn: Строка подключения к БД
        output_csv_path: Путь для сохранения CSV файла
        timeout: Таймаут для запросов в секундах
        analyze: Выполнять ANALYZE для получения фактического времени
        
    Returns:
        Статистика построения датасета
    """
    print(f"Инициализация DatasetBuilder...")
    
    # Создаем конфигурацию признаков
    feature_config = create_feature_config()
    
    # Создаем билдер датасета
    builder = DatasetBuilder(dsn=dsn, feature_config=feature_config)
    
    print(f"Начинаем построение датасета из {len(queries)} запросов...")
    print(f"Анализ: {'включен' if analyze else 'отключен'}")
    print(f"Таймаут: {timeout} секунд")
    print("." * 50)
    
    try:
        # Строим датасет с автоматическим timeout и rollback
        # Timeout устанавливается для каждого запроса через statement_timeout
        # Rollback выполняется автоматически при ошибках в SyncQueryPlanService
        dataset = builder.build_from_queries(
            queries=queries,
            output_path=output_csv_path,
            analyze=analyze,
            timeout=timeout
        )
        
        # Получаем статистику
        stats = builder.get_dataset_statistics(dataset)
        
        print(f"\nДатасет сохранен: {output_csv_path}")
        print(f"Общее количество записей: {stats.get('total_records', 0)}")
        print(f"Записи с целевой переменной: {stats.get('records_with_target', 0)}")
        print(f"Покрытие целевой переменной: {stats.get('target_coverage', 0):.2%}")
        
        if 'target_stats' in stats and stats['target_stats']:
            target_stats = stats['target_stats']
            print(f"Статистика времени выполнения:")
            print(f"  Среднее: {target_stats.get('mean', 0):.2f} мс")
            print(f"  Медиана: {target_stats.get('median', 0):.2f} мс")
            print(f"  Мин: {target_stats.get('min', 0):.2f} мс")
            print(f"  Макс: {target_stats.get('max', 0):.2f} мс")
        
        if 'feature_stats' in stats:
            feature_stats = stats['feature_stats']
            print(f"Статистика признаков:")
            print(f"  Всего признаков: {feature_stats.get('total_features', 0)}")
            print(f"  Среднее на запись: {feature_stats.get('avg_features_per_record', 0):.1f}")
        
        return {
            "success": True,
            "records_processed": len(dataset),
            "output_file": output_csv_path,
            "statistics": stats
        }
        
    except Exception as e:
        print(f"\nОшибка при построении датасета: {e}")
        return {
            "success": False,
            "error": str(e),
            "records_processed": 0
        }


def main():
    """Основная функция для запуска построения датасетов"""
    
    print("=== Построение датасетов с использованием xxplain ===\n")
    
    # Настройка окружения
    try:
        dsn = setup_environment()
        print(f"Подключение к БД настроено")
    except Exception as e:
        print(f"Ошибка настройки подключения к БД: {e}")
        return
    
    # Конфигурация источников данных (аналогично оригинальному скрипту)
    TRAIN_SQL_FILE = "benchmarks/tpc-h/generated/samples.sql"
    TEST_SQL_FOLDER = "benchmarks/tpc-h/queries/"
    
    # Параметры построения
    TRAIN_TIMEOUT = 20  # секунд
    TEST_TIMEOUT = 100  # секунд
    
    datasets_to_build = [
        {
            "name": "training",
            "source": TRAIN_SQL_FILE,
            "output": "datasets/train.csv",
            "timeout": TRAIN_TIMEOUT,
            "analyze": True
        },
        {
            "name": "test", 
            "source": TEST_SQL_FOLDER,
            "output": "datasets/test.csv",
            "timeout": TEST_TIMEOUT,
            "analyze": True
        }
    ]
    
    # Создаем директорию для датасетов
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Строим каждый датасет
    results = {}
    
    for dataset_config in datasets_to_build:
        print(f"\n--- Построение {dataset_config['name']} датасета ---")
        
        try:
            # Читаем запросы
            print(f"Чтение запросов из: {dataset_config['source']}")
            queries = read_queries_from_source(dataset_config['source'])
            print(f"Загружено {len(queries)} запросов")
            
            if not queries:
                print(f"Не найдено запросов в {dataset_config['source']}")
                results[dataset_config['name']] = {"success": False, "error": "No queries found"}
                continue
            
            # Строим датасет
            result = build_dataset_with_xxplain(
                queries=queries,
                dsn=dsn,
                output_csv_path=dataset_config['output'],
                timeout=dataset_config['timeout'],
                analyze=dataset_config['analyze']
            )
            
            results[dataset_config['name']] = result
            
        except Exception as e:
            print(f"Ошибка при обработке {dataset_config['name']} датасета: {e}")
            results[dataset_config['name']] = {"success": False, "error": str(e)}
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СВОДКА:")
    print(f"{'='*60}")
    
    total_success = 0
    total_records = 0
    
    for name, result in results.items():
        status = "✓ Успешно" if result.get("success") else "✗ Ошибка"
        records = result.get("records_processed", 0)
        
        print(f"{name.upper()} датасет: {status}")
        if result.get("success"):
            print(f"  Записей обработано: {records}")
            print(f"  Файл: {result.get('output_file', 'N/A')}")
            total_success += 1
            total_records += records
        else:
            print(f"  Ошибка: {result.get('error', 'Unknown error')}")
        print()
    
    print(f"Успешно построено датасетов: {total_success}/{len(results)}")
    print(f"Общее количество записей: {total_records}")
    
    if total_success == len(results):
        print("Все датасеты успешно построены! 🎉")
    else:
        print("Некоторые датасеты построить не удалось. Проверьте ошибки выше.")


if __name__ == "__main__":
    main()
