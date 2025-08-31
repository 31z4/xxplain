#!/usr/bin/env python3
"""
Скрипт для обучения моделей предсказания времени выполнения PostgreSQL запросов.
Использует новые классы из пакета xxplain.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Импортируем новые классы из xxplain
from ml.experiments import ModelTrainer
from ml.features import FeatureConfig
from ml.evaluation import calculate_metrics, ModelEvaluator


def load_csv_data(train_path: str, test_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Загружает данные из CSV файлов
    
    Args:
        train_path: Путь к файлу с обучающими данными
        test_path: Путь к файлу с тестовыми данными (опционально)
        
    Returns:
        Словарь с загруженными данными
    """
    print(f"Загрузка обучающих данных из {train_path}")
    train_df = pd.read_csv(train_path, delimiter='\t')
    
    # Проверяем наличие необходимых колонок
    required_columns = ['query', 'plan', 'features', 'target']
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
    
    print(f"Загружено {len(train_df)} записей для обучения")
    
    data = {'train': train_df}
    
    if test_path:
        print(f"Загрузка тестовых данных из {test_path}")
        test_df = pd.read_csv(test_path, delimiter='\t')
        print(f"Загружено {len(test_df)} записей для тестирования")
        data['test'] = test_df
    
    return data


def prepare_data_from_features(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Подготавливает данные из CSV с готовыми признаками для ModelTrainer
    
    Args:
        df: DataFrame с данными
        
    Returns:
        Список словарей в формате для ModelTrainer
    """
    data_list = []
    
    for idx, row in df.iterrows():
        try:
            # Парсим готовые признаки
            features = json.loads(row['features'])
            
            # Парсим план запроса
            plan = json.loads(row['plan'])
            
            # Извлекаем параметры PostgreSQL из признаков
            server_params = {}
            for key, value in features.items():
                if key.startswith('pgconf_'):
                    param_name = key.replace('pgconf_', '')
                    server_params[param_name] = value
            
            # Создаем элемент данных
            data_item = {
                'plan': plan,
                'query_text': row['query'],
                'server_params': server_params,
                'target': float(row['target']),
                'features': features  # Добавляем готовые признаки
            }
            
            data_list.append(data_item)
            
        except Exception as e:
            print(f"Ошибка обработки строки {idx}: {e}")
            continue
    
    print(f"Подготовлено {len(data_list)} записей для обучения")
    return data_list


def prepare_features_directly(data_list: List[Dict[str, Any]]) -> tuple:
    """
    Извлекает готовые признаки напрямую из данных для быстрого обучения
    
    Args:
        data_list: Список с данными
        
    Returns:
        Кортеж (X, y, feature_names)
    """
    features_list = []
    targets = []
    
    for item in data_list:
        if 'features' in item and 'target' in item:
            features_list.append(item['features'])
            targets.append(item['target'])
    
    # Преобразуем в DataFrame
    features_df = pd.DataFrame(features_list).fillna(0.0)
    
    # Целевая переменная (логарифм для лучшего обучения)
    y = np.log1p(np.array(targets))
    X = features_df.values
    
    return X, y, list(features_df.columns)


class DirectModelTrainer:
    """
    Упрощенный тренер моделей для работы с готовыми признаками
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        
        # Доступные модели
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        self.available_models = {
            'Linear': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'RandomForest': RandomForestRegressor,
            'GradientBoosting': GradientBoostingRegressor
        }
    
    def train_models(self, X, y, feature_names, test_size=0.2):
        """Обучает все модели"""
        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Нормализация
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_names = feature_names
        
        print(f"Данные подготовлены:")
        print(f"  Train: {X_train_scaled.shape}")
        print(f"  Test: {X_test_scaled.shape}")
        print(f"  Признаков: {X_train_scaled.shape[1]}")
        
        # Обучаем модели
        trained_models = {}
        test_results = {}
        y_test_true = np.expm1(y_test)  # Вычисляем один раз для всех моделей
        
        for model_name, model_class in self.available_models.items():
            print(f"\nОбучение модели {model_name}...")
            
            # Параметры по умолчанию
            model_params = {}
            if hasattr(model_class(), 'random_state'):
                model_params['random_state'] = self.random_state
            if model_name == 'Ridge':
                model_params['alpha'] = 1.0
            elif model_name == 'Lasso':
                model_params['alpha'] = 0.1
            elif model_name in ['RandomForest', 'GradientBoosting']:
                model_params['n_estimators'] = 100
            
            # Создаем и обучаем модель
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            
            # Предсказания на тесте
            y_test_pred_log = model.predict(X_test_scaled)
            y_test_pred = np.expm1(y_test_pred_log)
            y_test_pred = np.maximum(y_test_pred, 0)  # Ограничиваем снизу нулем
            
            # Метрики
            test_metrics = calculate_metrics(y_test_true, y_test_pred)
            
            trained_models[model_name] = {
                'model': model,
                'model_params': model_params,
                'test_metrics': test_metrics
            }
            
            test_results[model_name] = {
                'predictions': y_test_pred,
                'metrics': test_metrics
            }
            
            # Выводим основные метрики
            mae = test_metrics.get('MAE', 'N/A')
            q_error = test_metrics.get('Q_Error_Median', 'N/A')
            r2 = test_metrics.get('R2', 'N/A')
            print(f"  MAE: {mae:.4f}")
            print(f"  Q-Error Median: {q_error:.4f}")
            print(f"  R²: {r2:.4f}")
        
        return trained_models, test_results, (X_test_scaled, y_test, y_test_true)
    
    def save_models(self, trained_models, output_dir="models"):
        """Сохраняет обученные модели"""
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем каждую модель
        for model_name, model_data in trained_models.items():
            model_path = output_path / f"{model_name.lower()}_model.pkl"
            joblib.dump(model_data['model'], model_path)
            print(f"Модель {model_name} сохранена: {model_path}")
        
        # Сохраняем скейлер
        if self.scaler:
            scaler_path = output_path / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"Скейлер сохранен: {scaler_path}")
        
        # Сохраняем метаданные
        metadata = {
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "models_available": list(trained_models.keys()),
            "random_state": self.random_state
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Метаданные сохранены: {metadata_path}")
        
        return output_path


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Обучение моделей предсказания времени выполнения PostgreSQL запросов')
    parser.add_argument('--train', default='datasets/train.csv', help='Путь к обучающим данным')
    parser.add_argument('--test', default='datasets/test.csv', help='Путь к тестовым данным (опционально)')
    parser.add_argument('--output', default='models', help='Директория для сохранения моделей')
    parser.add_argument('--test-size', type=float, default=0.2, help='Размер тестовой выборки')
    parser.add_argument('--random-state', type=int, default=42, help='Сид для воспроизводимости')
    parser.add_argument('--use-extractor', action='store_true', help='Использовать FeatureExtractor вместо готовых признаков')
    
    args = parser.parse_args()
    
    try:
        # Загружаем данные
        data = load_csv_data(args.train, args.test if Path(args.test).exists() else None)
        train_df = data['train']
        
        if args.use_extractor:
            print("\nИспользуем FeatureExtractor для извлечения признаков...")
            
            # Подготавливаем данные для ModelTrainer
            train_data = prepare_data_from_features(train_df)
            
            # Создаем конфигурацию признаков (аналогично lab/predict_model.py)
            feature_config = FeatureConfig(
                extract_basic_stats=True,
                extract_node_counts=True,
                extract_join_info=True,
                extract_cost_estimates=True,
                extract_query_features=True,
                extract_pg_config=True,
                extract_parallel_info=True,
                extract_filter_info=True,
                extract_aggregation_info=True
            )
            
            # Создаем и используем ModelTrainer
            trainer = ModelTrainer(feature_config=feature_config, random_state=args.random_state)
            
            # Подготавливаем данные
            X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
                train_data, test_size=args.test_size, validation_size=0.1
            )
            
            # Обучаем модели
            trained_models = trainer.train_multiple_models(X_train, y_train, X_val, y_val)
            
            # Оцениваем на тесте
            test_results = trainer.evaluate_on_test(trained_models, X_test, y_test)
            
            # Сохраняем модели
            trainer.save_trained_models(trained_models, args.output)
            
        else:
            print("\nИспользуем готовые признаки из CSV...")
            
            # Подготавливаем данные с готовыми признаками
            train_data = prepare_data_from_features(train_df)
            X, y, feature_names = prepare_features_directly(train_data)
            
            # Создаем упрощенный тренер
            trainer = DirectModelTrainer(random_state=args.random_state)
            
            # Обучаем модели
            trained_models, test_results, test_data = trainer.train_models(
                X, y, feature_names, test_size=args.test_size
            )
            
            # Сохраняем модели
            output_path = trainer.save_models(trained_models, args.output)
        
        print(f"\n{'='*60}")
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print(f"{'='*60}")
        
        # Выводим сравнение моделей
        model_comparison = []
        for model_name in trained_models.keys():
            if args.use_extractor:
                metrics = test_results[model_name]['test_metrics']
            else:
                metrics = trained_models[model_name]['test_metrics']
            
            model_comparison.append({
                'Модель': model_name,
                'MAE': f"{metrics.get('MAE', 0):.4f}",
                'Q-Error Median': f"{metrics.get('Q_Error_Median', 0):.4f}",
                'R²': f"{metrics.get('R2', 0):.4f}",
                'Within 2x': f"{metrics.get('Within_2.0x', 0):.1f}%"
            })
        
        # Сортируем по MAE
        model_comparison.sort(key=lambda x: float(x['MAE']))
        
        # Выводим таблицу
        print(f"{'Модель':<15} {'MAE':<10} {'Q-Error':<12} {'R²':<8} {'Within 2x':<10}")
        print("-" * 60)
        for result in model_comparison:
            print(f"{result['Модель']:<15} {result['MAE']:<10} {result['Q-Error Median']:<12} {result['R²']:<8} {result['Within 2x']:<10}")
        
        print(f"\nЛучшая модель по MAE: {model_comparison[0]['Модель']}")
        print(f"Модели сохранены в: {args.output}")
        
        # Тестируем загрузку моделей
        print(f"\n{'='*60}")
        print("ТЕСТИРОВАНИЕ ЗАГРУЗКИ МОДЕЛЕЙ")
        print(f"{'='*60}")
        
        try:
            from ml.models import ModelLoader
            
            loader = ModelLoader(args.output)
            available_models = loader.get_available_models()
            print(f"Успешно загружено моделей: {len(available_models)}")
            print(f"Доступные модели: {available_models}")
            
            # Тестируем предсказание на примере
            if available_models and hasattr(loader, 'get_feature_names'):
                feature_names = loader.get_feature_names()
                if feature_names:
                    # Создаем тестовые признаки (нули)
                    test_features = {name: 0.0 for name in feature_names}
                    # Устанавливаем несколько реалистичных значений
                    test_features.update({
                        'n_nodes': 4.0,
                        'max_depth': 3.0,
                        'sum_est_total_cost': 1000.0,
                        'root_total_cost': 500.0,
                        'query_length': 100.0
                    })
                    
                    for model_name in available_models[:2]:  # Тестируем первые 2 модели
                        try:
                            prediction = loader.predict(test_features, model_name)
                            print(f"Тестовое предсказание {model_name}: {prediction:.4f} мс")
                        except Exception as e:
                            print(f"Ошибка предсказания {model_name}: {e}")
            
        except Exception as e:
            print(f"Ошибка при тестировании загрузки: {e}")
        
        print("\nОбучение завершено успешно!")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()