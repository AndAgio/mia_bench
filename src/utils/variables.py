import sys
import pathlib



this_file_path = pathlib.Path(__file__).resolve()
PATH_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PATH_REPO))
DEFAULT_LOG_FOLDER = PATH_REPO.joinpath('logs')
DEFAULT_MODELS_FOLDER = PATH_REPO.joinpath('ckpts')
DEFAULT_RESUME_CKPTS_FOLDER = PATH_REPO.joinpath('resume_ckpts')
DEFAULT_DATASETS_FOLDER = PATH_REPO.joinpath('datas')
DEFAULT_SHADOW_DATASETS_FOLDER = PATH_REPO.joinpath('shadow_datas')
DEFAULT_METRICS_FOLDER = PATH_REPO.joinpath('metrics')
DEFAULT_OUT_FOLDER = PATH_REPO.joinpath('outs')
DEFAULT_PLOTS_FOLDER = PATH_REPO.joinpath('plots')


DEFAULT_METRICS = ['auc', 'tpr', 'fpr', 'roc']

PERCENTAGE_OF_DATA_TO_USE_FOR_SHADOW_DATASETS = 0.6
PERCENTAGE_OF_DATA_TO_USE_FOR_TRAINING_SPLIT = 0.6
PERCENTAGE_OF_DATA_TO_USE_FOR_VALIDATION_SPLIT = 0.15
PERCENTAGE_OF_DATA_TO_USE_FOR_TEST_SPLIT = 0.25