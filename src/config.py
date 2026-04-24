import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'test.csv')
DEFAULT_MODEL_BUNDLE_PATH = os.path.join(MODEL_DIR, 'titanic_logistic_bundle.pkl')
DEFAULT_SUBMISSION_PATH = os.path.join(RESULTS_DIR, 'submission.csv')