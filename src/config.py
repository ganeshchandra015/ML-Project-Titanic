import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'test.csv')