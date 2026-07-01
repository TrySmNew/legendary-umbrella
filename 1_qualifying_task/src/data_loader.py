import pandas as pd
import numpy as np
from src.config import TRAIN_PATH, TEST_PATH, TARGET, RANDOM_STATE, DROP_COLS


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    """Загружает train и test данные"""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape: {df_test.shape}")
    
    return df_train, df_test


def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_fn):
    """
    Подготовка данных с использованием внешней функции feature engineering
    
    Args:
        df_train: train DataFrame
        df_test: test DataFrame
        feature_fn: функция для создания признаков (например, add_features)
    
    Returns:
        X_train, X_test, y_train, y_test, X_main, y_main, X_test_final, test_ids
    """
    from sklearn.model_selection import train_test_split
    
    # Применяем feature engineering
    print("Adding features to train data...")
    df_train = feature_fn(df_train)
    
    print("Adding features to test data...")
    df_test = feature_fn(df_test)
    
    # Сохраняем ID для сабмишена
    test_ids = df_test['id'].copy()
    
    # Разделяем на признаки и таргет
    y = df_train[TARGET].copy()
    X = df_train.drop(columns=DROP_COLS).copy()
    
    # Test данные (без таргета)
    X_test_final = df_test.drop(columns=['id']).copy()
    
    # Train/test split для валидации
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # X_main, y_main - все train данные для финального обучения
    X_main = X.copy()
    y_main = y.copy()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_main shape: {X_main.shape}")
    print(f"X_test_final shape: {X_test_final.shape}")
    
    return X_train, X_test, y_train, y_test, X_main, y_main, X_test_final, test_ids