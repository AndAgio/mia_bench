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

# MemGuard's Phase I settings, from the paper (arXiv 1909.10594, Eq. 20 and Sec. 5.1) and the authors' code
# (defense_framework.py): the weights c1 and c2 of the membership and label losses, the first and largest weight c3 of
# the distortion loss, the gradient steps for each c3, and the step size beta. Only the code bounds c3.
MEMGUARD_PAPER_SETTINGS = {'c1': 1.0, 'c2': 10.0, 'c3_init': 0.1, 'c3_max': 1e5, 'max_iter': 300, 'step_size': 0.1}

# LDL's number of noisy copies per query, by noise type, from the authors' code (classifier.py): 200 with Gaussian
# noise (images), 100 with bit flips (Purchase, Texas). Unset --defender_ldl_n_queries takes the value of the dataset's
# noise.
LDL_CODE_N_QUERIES = {'normal': 200, 'bernoulli': 100}
# LDL's probability of flipping each feature of the binary datasets, from Table 1 of the paper (arXiv 2212.01688),
# which picks it by the number of features. Unset --defender_ldl_flip_prob takes the dataset's value.
LDL_PAPER_FLIP_PROBS = {'purchase': 0.0166, 'texas': 0.00016}

# Purifier's reformer settings per dataset, from the paper's appendix (arXiv 2212.00612): the width h of the first
# hidden layer, the latent size, and the training epochs and learning rate. The appendix gives 0.0005 "for
# Facescrub530 and CIFAR10" right after 0.01 for CIFAR10, which we read as CIFAR100. Unset
# --defender_purifier_reformer_* options take the dataset's values, and datasets the paper does not evaluate take
# CIFAR-10's.
PURIFIER_PAPER_REFORMER = {
    'cifar10': {'reformer_hidden_dim': 32, 'reformer_latent_dim': 2, 'reformer_epochs': 100, 'reformer_lr': 0.01},
    'cifar100': {'reformer_hidden_dim': 512, 'reformer_latent_dim': 100, 'reformer_epochs': 300,
                 'reformer_lr': 0.0005},
    'purchase': {'reformer_hidden_dim': 128, 'reformer_latent_dim': 20, 'reformer_epochs': 150,
                 'reformer_lr': 0.0001},
    'texas': {'reformer_hidden_dim': 128, 'reformer_latent_dim': 20, 'reformer_epochs': 100, 'reformer_lr': 0.0001},
}
PURIFIER_FALLBACK_DATASET = 'cifar10'
