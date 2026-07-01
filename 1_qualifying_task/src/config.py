import os

RANDOM_STATE = 42

# Пути к данным
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(DATA_DIR, 'data', 'train')
TEST_PATH = os.path.join(DATA_DIR, 'data', 'test')

# Целевая переменная
TARGET = 'retention'

# Исключаемые колонки
DROP_COLS = ['id', 'retention']