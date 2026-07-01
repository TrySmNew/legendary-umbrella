from src.data_loader import load_data, prepare_data
from src.feature_engineering import add_features

df_train, df_test = load_data()

X_train, X_test, y_train, y_test, X_main, y_main, X_test_final, test_ids = prepare_data(
    df_train, df_test, feature_fn=add_features
)

print(f"Features: {X_train.columns.tolist()}")
print(f"Количество признаков: {X_train.shape[1]}")