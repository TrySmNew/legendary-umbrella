import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)



# ЛОГИРОВАНИЕ
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настраивает логгер для проекта.
    
    Args:
        name: имя логгера
        level: уровень логирования
    
    Returns:
        настроенный Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛЕЙ
def save_model(model, path: str) -> None:
    """
    Сохраняет модель в файл.
    
    Args:
        model: обученная модель
        path: путь для сохранения
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена: {path}")


def load_model(path: str):
    """
    Загружает модель из файла.
    
    Args:
        path: путь к файлу модели
    
    Returns:
        загруженная модель
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"Модель загружена: {path}")
    return model


def save_models(models: dict, directory: str = 'models/') -> None:
    """
    Сохраняет несколько моделей в указанную директорию.
    
    Args:
        models: dict с именами моделей и самими моделями
        directory: директория для сохранения
    """
    os.makedirs(directory, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(directory, f'{name}.pkl')
        save_model(model, path)


# ВИЗУАЛИЗАЦИЯ МЕТРИК
def plot_feature_importance(model, feature_names: list, top_n: int = 20, 
                           figsize: tuple = (12, 8)) -> None:
    """
    Визуализирует важность признаков.
    
    Args:
        model: обученная модель
        feature_names: список названий признаков
        top_n: количество топ признаков для отображения
        figsize: размер графика
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        importances = model.get_feature_importance()
    else:
        print("Модель не поддерживает feature importance")
        return
    
    feature_imp = pd.Series(importances, index=feature_names)
    feature_imp = feature_imp.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_imp.values, y=feature_imp.index)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (14, 10)) -> None:
    """
    Визуализирует корреляционную матрицу.
    
    Args:
        df: DataFrame для анализа
        figsize: размер графика
    """
    corr = df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()


def plot_target_correlation(df: pd.DataFrame, target: str = 'retention', 
                           top_n: int = 15) -> None:
    """
    Визуализирует корреляцию признаков с целевой переменной.
    
    Args:
        df: DataFrame с признаками и таргетом
        target: название целевой переменной
        top_n: количество топ признаков
    """
    corr_with_target = df.corr()[target].drop(target).sort_values(ascending=False)
    top_corr = corr_with_target.head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
    plt.title(f'Top {top_n} Features Correlated with {target}')
    plt.xlabel('Correlation')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# АНАЛИЗ ДАННЫХ
def data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает сводную статистику по DataFrame.
    
    Args:
        df: DataFrame для анализа
    
    Returns:
        DataFrame со сводной статистикой
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'nulls': df.isnull().sum(),
        'nulls_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'nunique': df.nunique(),
    })
    return summary


def check_class_balance(y: pd.Series) -> dict:
    """
    Проверяет баланс классов.
    
    Args:
        y: Series с метками классов
    
    Returns:
        dict с информацией о балансе
    """
    counts = y.value_counts()
    ratio = counts.iloc[0] / counts.iloc[1] if len(counts) == 2 else None
    
    info = {
        'class_counts': counts.to_dict(),
        'ratio': ratio,
        'is_balanced': 0.5 <= ratio <= 2.0 if ratio else False,
    }
    
    print(f"Распределение классов:")
    for cls, count in info['class_counts'].items():
        print(f"  Класс {cls}: {count} ({count/len(y)*100:.1f}%)")
    print(f"Соотношение: {ratio:.2f}" if ratio else "Многоклассовая задача")
    print(f"Сбалансирован: {'Да' if info['is_balanced'] else 'Нет'}")
    
    return info


# ВОСПРОИЗВОДИМОСТЬ
def set_seed(seed: int = 42) -> None:
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed: значение seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    print(f"Seed установлен: {seed}")