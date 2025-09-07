from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib
from pathlib import Path
import json

from ml.metrics import calculate_metrics


def load_dataset(dataset_path: str):
    return pd.read_csv(dataset_path)


def prepare_data(
    dataset,
    target_column: str = 'time',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Подготавливает данные для обучения

    Args:
        data: Список словарей с features, target (time или size)
        target_column: Название колонки с целевой переменной
        test_size: Размер тестовой выборки
        val_size: Размер валидационной выборки

    Returns:
        Кортеж (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    features_df = pd.json_normalize(dataset['features'].apply(json.loads))
    features_df = features_df.fillna(0.0)
    features_df = features_df.astype(float)

    # Целевая переменная (логарифм для лучшего обучения)
    targets = dataset[target_column]
    y = np.log1p(np.array(targets))
    X = features_df.values
    print('>>> ', X.shape, features_df.columns)

    # Разделяем на train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = X_temp, np.array([]), y_temp, np.array([])

    print("Данные подготовлены:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape if val_size > 0 else 'None'}")
    print(f"  Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_train_data(
    data,
    target_column: str = 'time',
    val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Подготавливает обучающие данные (train/val)

    Args:
        data: Список словарей с features, target (time или size)
        target_column: Название колонки с целевой переменной
        val_size: Размер валидационной выборки

    Returns:
        Кортеж (X_train, X_val, y_train, y_val)
    """
    features_list = []
    targets = []
    for item in data:
        try:
            target = item.get(target_column)
            features = json.loads(item.get('features'))

            if target is None:
                continue

            features_list.append(features)
            targets.append(float(target))

        except Exception as e:
            print(f"Ошибка обработки элемента: {e}")
            continue

    if not features_list:
        raise ValueError("Не удалось извлечь признаки")

    # Преобразуем в DataFrame
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0.0)

    # Целевая переменная (логарифм для лучшего обучения)
    y = np.log1p(np.array(targets))
    X = features_df.values

    # Разделяем на train/val
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = X, np.array([]), y, np.array([])

    print("Обучающие данные подготовлены:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape if val_size > 0 else 'None'}")

    return X_train, X_val, y_train, y_val, features_df.columns.tolist()


def prepare_test_data(
    data,
    train_columns: List[str],
    target_column: str = 'time'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает тестовые данные

    Args:
        data: Список словарей с features, target
        train_columns: Список колонок из обучающих данных
        target_column: Название колонки с целевой переменной

    Returns:
        Кортеж (X_test, y_test)
    """
    features_list = []
    targets = []
    for item in data:
        try:
            target = item.get(target_column)
            features = json.loads(item.get('features'))

            if target is None:
                continue

            features_list.append(features)
            targets.append(float(target))

        except Exception as e:
            print(f"Ошибка обработки элемента: {e}")
            continue

    if not features_list:
        raise ValueError("Не удалось извлечь признаки из тестовых данных")

    # Преобразуем в DataFrame
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0.0)

    # Выравниваем колонки с обучающими данными
    for col in train_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df = features_df[train_columns]

    # Целевая переменная
    y = np.log1p(np.array(targets))
    X = features_df.values

    print("Тестовые данные подготовлены:")
    print(f"  Test: {X.shape}")

    return X, y


def create_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Создает и обучает StandardScaler

    Args:
        X_train: Обучающие признаки

    Returns:
        Обученный scaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Применяет scaler к данным

    Args:
        scaler: Обученный scaler
        X_train: Обучающие признаки
        X_val: Валидационные признаки
        X_test: Тестовые признаки

    Returns:
        Кортеж масштабированных данных
    """
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = np.array([])

    return X_train_scaled, X_val_scaled, X_test_scaled


def get_model_class(model_type: str):
    """Возвращает класс модели по типу"""
    models = {
        'Ridge': Ridge,
        'Lasso': Lasso,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': XGBRegressor,
        'CatBoost': CatBoostRegressor
    }

    if model_type not in models:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

    return models[model_type]


def get_param_grids() -> Dict[str, List[Dict[str, Any]]]:
    """Возвращает сетки параметров для тюнинга"""
    return {
        'Ridge': [
            {'alpha': 0.1},
            {'alpha': 1.0},
            {'alpha': 10.0},
            {'alpha': 100.0}
        ],
        'Lasso': [
            {'alpha': 0.01},
            {'alpha': 0.1},
            {'alpha': 1.0},
            {'alpha': 10.0}
        ],
        'RandomForest': [
            {'n_estimators': 50, 'max_depth': None},
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': None},
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 100, 'max_depth': 20}
        ],
        'GradientBoosting': [
            {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
        ],
        'XGBoost': [
            {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'random_state': 42}
        ],
        'CatBoost': [
            {'iterations': 50, 'depth': 4, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': 0},
            {'iterations': 100, 'depth': 4, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': 0},
            {'iterations': 100, 'depth': 6, 'learning_rate': 0.1, 'random_seed': 42, 'verbose': 0},
            {'iterations': 100, 'depth': 4, 'learning_rate': 0.01, 'random_seed': 42, 'verbose': 0},
            {'iterations': 200, 'depth': 5, 'learning_rate': 0.05, 'random_seed': 42, 'verbose': 0}
        ]
    }


def tune_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    """
    Проводит тюнинг гиперпараметров модели

    Args:
        model_type: Тип модели
        X_train: Обучающие признаки
        y_train: Обучающие цели
        X_val: Валидационные признаки
        y_val: Валидационные цели

    Returns:
        Кортеж (лучшая_модель, лучшие_параметры)
    """
    model_class = get_model_class(model_type)
    param_grid = get_param_grids().get(model_type, [{}])

    best_score = float('inf')
    best_model = None
    best_params = {}

    print(f"Тюнинг {model_type} на {len(param_grid)} комбинациях...")

    for params in param_grid:
        try:
            # Для XGBoost и CatBoost параметры уже включают random_state/random_seed
            if model_type in ['XGBoost', 'CatBoost']:
                model = model_class(**params)
            else:
                model = model_class(random_state=42, **params)
            model.fit(X_train, y_train)

            if len(X_val) > 0:
                y_pred_log = model.predict(X_val)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_val)
                score = calculate_metrics(y_true, y_pred)['Q_Error_Mean']
            else:
                # Если нет валидации, используем train
                y_pred_log = model.predict(X_train)
                y_pred = np.expm1(y_pred_log)
                y_true = np.expm1(y_train)
                score = calculate_metrics(y_true, y_pred)['Q_Error_Mean']

            if score < best_score:
                best_score = score
                best_model = model
                best_params = params

        except Exception as e:
            print(f"Ошибка с параметрами {params}: {e}")
            continue

    if best_model is None:
        # Fallback на дефолтную модель
        if model_type in ['XGBoost', 'CatBoost']:
            if model_type == 'XGBoost':
                best_model = model_class(random_state=42)
            else:  # CatBoost
                best_model = model_class(random_seed=42, verbose=0)
        else:
            best_model = model_class(random_state=42)
        best_model.fit(X_train, y_train)

    print(f"Лучшие параметры для {model_type}: {best_params}")
    return best_model, best_params


def train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Tuple[Any, Dict[str, Any]]:
    """
    Обучает модель с тюнингом

    Args:
        model_type: Тип модели
        X_train: Обучающие признаки
        y_train: Обучающие цели
        X_val: Валидационные признаки
        y_val: Валидационные цели

    Returns:
        Кортеж (модель, параметры)
    """
    print(f"Обучение модели {model_type}...")

    # Тюнинг на 10% данных (используем валидационную выборку)
    if len(X_val) > 0:
        model, params = tune_model(model_type, X_train, y_train, X_val, y_val)
    else:
        # Если нет валидации, обучаем с дефолтными параметрами
        model_class = get_model_class(model_type)
        if model_type in ['XGBoost', 'CatBoost']:
            if model_type == 'XGBoost':
                model = model_class(random_state=42)
            else:  # CatBoost
                model = model_class(random_seed=42, verbose=0)
        else:
            model = model_class(random_state=42)
        model.fit(X_train, y_train)
        params = {}

    return model, params


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Оценивает модели на тестовой выборке

    Args:
        models: Словарь {тип_модели: модель}
        X_test: Тестовые признаки
        y_test: Тестовые цели

    Returns:
        Результаты оценки
    """
    results = {}
    y_test_true = np.expm1(y_test)

    for model_type, model in models.items():
        try:
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_pred = np.maximum(y_pred, 0)  # Ограничиваем снизу нулем

            metrics = calculate_metrics(y_test_true, y_pred)

            results[model_type] = metrics

        except Exception as e:
            print(f"Ошибка оценки {model_type}: {e}")
            results[model_type] = {}

    # Отображаем все метрики в виде таблицы
    if results:
        print("\n" + "="*80)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛЕЙ")
        print("="*80)

        # Создаем DataFrame из результатов
        df_results = pd.DataFrame.from_dict(results, orient='index')

        # Округляем числовые значения для лучшей читаемости
        numeric_columns = df_results.select_dtypes(include=[np.number]).columns
        df_results[numeric_columns] = df_results[numeric_columns].round(4)

        # Выводим таблицу
        print(df_results.to_string())
        print("="*80)

        # Также печатаем ключевые метрики для быстрого обзора
        print("\nКЛЮЧЕВЫЕ МЕТРИКИ:")
        for model_type, metrics in results.items():
            mae = metrics.get('MAE', 'N/A')
            q_error_median = metrics.get('Q_Error_Median', 'N/A')
            r2 = metrics.get('R2', 'N/A')
            print(f"{model_type}: MAE={mae}, Q-Error Median={q_error_median}, R²={r2}")

    return results


def save_models(
    models: Dict[str, Any],
    scaler: StandardScaler,
    output_dir: str = "models",
) -> Dict[str, str]:
    """
    Сохраняет модели и scaler

    Args:
        models: Словарь моделей
        scaler: Обученный scaler
        output_dir: Директория для сохранения

    Returns:
        Пути к сохраненным файлам
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # Сохраняем scaler
    scaler_path = output_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    saved_paths['scaler'] = str(scaler_path)

    # Сохраняем модели
    for model_type, model in models.items():
        model_path = output_path / f"{model_type.lower()}_model.pkl"
        joblib.dump(model, model_path)
        saved_paths[model_type] = str(model_path)

    # Сохраняем метаданные
    metadata = {
        "models_available": list(models.keys()),
        "scaler_path": str(scaler_path)
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_paths['metadata'] = str(metadata_path)

    print(f"Модели сохранены в {output_path}")
    return saved_paths


def run_training_pipeline(
    train_dataset_path: str,
    test_dataset_path: str,
    target_column: str = 'time',
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    Основная функция обучения

    Args:
        train_dataset_path: Путь к обучающему датасету
        test_dataset_path: Путь к тестовому датасету
        target_column: Название колонки с целевой переменной
        output_dir: Директория для сохранения

    Returns:
        Результаты обучения
    """
    model_types = ['Ridge', 'Lasso', 'RandomForest', 'GradientBoosting', 'XGBoost', 'CatBoost']

    # 1. Загружаем данные
    print("Загрузка обучающего датасета...")
    train_data = load_dataset(train_dataset_path)

    print("Загрузка тестового датасета...")
    test_data = load_dataset(test_dataset_path)

    # 2. Подготавливаем данные
    print("Подготовка обучающих данных...")
    X_train, X_val, y_train, y_val, train_columns = prepare_train_data(train_data, target_column)

    print("Подготовка тестовых данных...")
    X_test, y_test = prepare_test_data(test_data, train_columns, target_column)

    # 3. Создаем и применяем scaler
    print("Создание scaler...")
    scaler = create_scaler(X_train)
    X_train_scaled, X_val_scaled, X_test_scaled = apply_scaler(
        scaler, X_train, X_val, X_test
    )

    # 4. Обучаем модели
    trained_models = {}
    for model_type in model_types:
        model, params = train_model(
            model_type, X_train_scaled, y_train, X_val_scaled, y_val
        )
        trained_models[model_type] = model

    # 5. Оцениваем модели
    print("Оценка моделей...")
    evaluation_results = evaluate_models(trained_models, X_test_scaled, y_test)

    # 6. Сохраняем модели
    print("Сохранение моделей...")
    saved_paths = save_models(trained_models, scaler, output_dir)

    return {
        'models': trained_models,
        'evaluation_results': evaluation_results,
        'saved_paths': saved_paths,
        'scaler': scaler
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Использование: python scripts/c_train_models.py <train_dataset_path> <test_dataset_path>")
        sys.exit(1)

    train_dataset_path = sys.argv[1]
    test_dataset_path = sys.argv[2]

    # Пример использования
    result = run_training_pipeline(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        target_column="time",
        output_dir="time_models",
    )

    result = run_training_pipeline(
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        target_column="size",
        output_dir="size_models",
    )
    print("Обучение завершено!")