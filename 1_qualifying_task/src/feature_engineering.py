import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт новые признаки на основе сырых данных.
    
    Args:
        df: DataFrame с исходными признаками (id, sessions_count, avg_session_time, 
            days_since_last_activity, purchases_count, avg_purchase_value, 
            active_days, session_std, is_weekend_user)
    
    Returns:
        DataFrame с добавленными признаками (без avg_purchase_value)
    """
    df = df.copy()
    
    # БАЗОВАЯ АКТИВНОСТЬ
    df['sessions_per_active_day'] = df['sessions_count'] / df['active_days'].replace(0, 1)
    df['purchase_rate'] = df['purchases_count'] / df['sessions_count'].replace(0, 1)
    df['recency_activity_ratio'] = df['active_days'] / (df['days_since_last_activity'] + 1)
    
    # СЕГМЕНТАЦИЯ (WEEKEND EFFECT)
    df['weekend_x_session_per_day'] = df['is_weekend_user'] * df['sessions_per_active_day']

    # СТАБИЛЬНОСТЬ И РАЗБРОС
    df['session_cv'] = df['session_std'] / df['avg_session_time'].replace(0, 1)
    df['session_stability'] = 1 - (df['session_std'] / (df['avg_session_time'] + 1))
    
    # ЦЕННОСТЬ И КОНВЕРСИЯ
    df['spend_intensity'] = df['avg_purchase_value'] * df['purchase_rate']
    df['value_per_session'] = (
        (df['avg_purchase_value'] * df['purchases_count']) / (df['sessions_count'] + 1)
    )
    df['purchase_value_per_active_day'] = (
        (df['avg_purchase_value'] * df['purchases_count']) / (df['active_days'] + 1)
    )
    df['purchase_momentum'] = df['purchases_count'] / (df['active_days'] + 1)
    df['active_x_purchase'] = df['active_days'] * df['purchases_count']
    
    # РЕЦЕНТНОСТЬ И ВРЕМЕННЫЕ ЭФФЕКТЫ
    df['recency_score'] = 1 / (df['days_since_last_activity'] + 1)
    df['activity_decay'] = df['sessions_count'] / (df['days_since_last_activity'] + 1)
    df['recency_x_sessions'] = df['recency_score'] * df['sessions_count']
    df['days_since_last_per_active'] = df['days_since_last_activity'] / (df['active_days'] + 1)
    df['time_efficiency'] = df['avg_session_time'] * df['sessions_count']
    df['days_per_session'] = df['days_since_last_activity'] / (df['sessions_count'] + 1)
    df['weekend_x_days_per_session'] = df['is_weekend_user'] * df['days_per_session']

    # КВАДРАТИЧНЫЕ ПРИЗНАКИ (нелинейные зависимости)
    df['sessions_squared'] = df['sessions_count'] ** 2
    df['purchases_squared'] = df['purchases_count'] ** 2
    df['active_days_squared'] = df['active_days'] ** 2
    
    # ЛОГАРИФМИЧЕСКИЕ ПРИЗНАКИ (сглаживание выбросов)
    for col in ['sessions_count', 'purchases_count', 'avg_purchase_value', 'active_days']:
        df[f'log_{col}'] = np.log1p(df[col])
    
    # Удаляем avg_purchase_value (он уже учтён в производных признаках)
    df = df.drop(columns=['avg_purchase_value'])
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Возвращает список колонок-признаков (без id, retention и исключённых).
    
    Args:
        df: DataFrame после add_features
        exclude: дополнительные колонки для исключения
    
    Returns:
        list of feature names
    """
    if exclude is None:
        exclude = []
    
    drop_cols = ['id', 'retention'] + exclude
    return [col for col in df.columns if col not in drop_cols]