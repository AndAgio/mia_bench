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

# Fraction of the defender val split that AdvReg uses as its reference set D' (the rest selects the best checkpoint).
ADV_REG_REFERENCE_FRACTION_OF_VAL_SPLIT = 0.5

# HAMP's (gamma, alpha) per dataset: the entropy threshold of the soft labels and the strength of the entropy
# regularizer, from Table III of the paper (arXiv 2307.01610, Appendix A). The authors' train-all.sh scripts use the same
# values for CIFAR-10 and Purchase100. The paper also evaluates Location30 (0.5, 0.001), which the benchmark does not
# have. Unset --defender_hamp_gamma/--defender_hamp_alpha take the dataset's values, and datasets the paper does not
# evaluate take CIFAR-10's.
HAMP_PAPER_PARAMETERS = {'cifar10': (0.95, 0.001),
                         'cifar100': (0.5, 0.005),
                         'purchase': (0.8, 0.01),
                         'texas': (0.6, 0.01)}
HAMP_FALLBACK_DATASET = 'cifar10'
