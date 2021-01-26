import os
import json


def _path(path):
    return path if path[0] == '/' else os.path.normpath(os.path.join(base_path, path))


base_path = os.path.join(os.path.dirname(__file__), "..")
settings_path = os.getenv('SETTINGS_PATH', os.path.join(base_path, "settings.json"))

with open(settings_path, 'r') as f:
    settings = json.load(f)

LIBRARY_PATH = _path(base_path)
PREPARED_PATH = _path(settings['TRAIN_PREPARED_DATA_PATH'])
WHOLE_IMG_PATH = os.path.join(PREPARED_PATH, "train_images")
DATA_PATH = _path(settings['RAW_DATA_DIR'])
LOGS_PATH = _path(settings['LOGS_DIR'])
CHECKPOINTS_PATH = _path(settings['CHECKPOINT_DIR'])
MODELS_PATH = _path(settings['MODEL_DIR'])
SUBMISSIONS_PATH = _path(settings['SUBMISSION_DIR'])
TEST_DATA_PATH = _path(settings['TEST_DATA_PATH'])
TMP_PATH = _path(settings['TEMP_DIR'])
