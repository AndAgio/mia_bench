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

# MIST's number of sub-models C and weight lambda of the cross-difference loss per dataset, from Table 7 of the paper
# (arXiv 2311.00919), with the ResNet-18 rows for CIFAR: the benchmark has no AlexNet or DenseNet, whose rows differ.
# Unset --defender_mist_num_submodels/--defender_mist_lambda take the dataset's values, and datasets the paper does not
# evaluate take CIFAR-10's.
MIST_PAPER_PARAMETERS = {'cifar10': {'num_submodels': 2, 'lmbd': 3.5},
                         'cifar100': {'num_submodels': 2, 'lmbd': 13.0},
                         'purchase': {'num_submodels': 4, 'lmbd': 40.0},
                         'texas': {'num_submodels': 4, 'lmbd': 25.0}}
MIST_FALLBACK_DATASET = 'cifar10'

# RelaxLoss's target loss alpha and bound on the ground-truth score during posterior flattening, per dataset, from the
# official configs (github.com/DingfanChen/RelaxLoss, source/*/defense/configs): the paper gives no alpha, and bounds
# the score on non-image data only (App. B.3). CIFAR takes the ResNet-20 values for every model; the configs also have
# VGG11-bn ones. Unset --defender_relax_loss_alpha/--defender_relax_loss_upper take the dataset's values, other image
# datasets CIFAR-10's, and other non-image datasets Purchase's.
RELAXLOSS_CODE_PARAMETERS = {'cifar10': {'relax_alpha': 1.0, 'relax_upper': 1.0},
                             'cifar100': {'relax_alpha': 3.0, 'relax_upper': 1.0},
                             'purchase': {'relax_alpha': 0.8, 'relax_upper': 0.3},
                             'texas': {'relax_alpha': 2.5, 'relax_upper': 0.1}}
RELAXLOSS_IMAGE_FALLBACK_DATASET = 'cifar10'
RELAXLOSS_NONIMAGE_FALLBACK_DATASET = 'purchase'

# Weighted smoothing's epochs of normal training before the smoothing starts, from Sec. 5.4 of the paper
# (doi.org/10.1145/3627106.3627189): 2 on image datasets, 1 on tabular ones. Unset
# --defender_weighted_smoothing_warmup_epochs takes the value for the dataset's kind; News counts as non-image.
WEIGHTED_SMOOTHING_PAPER_WARMUP_EPOCHS = {'image': 2, 'non_image': 1}

# AdvReg's adversarial regularization factor lambda per dataset, from Table 3 of the paper (arXiv 1807.05852). The
# authors' code (github.com/SPIN-UMass/ML-Privacy-Regulization) agrees on Purchase only, and uses other penalties. Unset
# --defender_adv_reg_lambda takes the dataset's value, other image datasets CIFAR-100's, and other non-image datasets
# Purchase's.
ADV_REG_PAPER_LAMBDAS = {'purchase': 3.0, 'texas': 2.0, 'cifar100': 6.0}
ADV_REG_IMAGE_FALLBACK_DATASET = 'cifar100'
ADV_REG_NONIMAGE_FALLBACK_DATASET = 'purchase'
# AdvReg's learning rate of the attack model's Adam optimizer, from the authors' Purchase code. The paper's 0.001
# (Sec. 5.3) lets the attack model die, and the code's CIFAR value 0.00001 is too slow (docs/defenses/adv_reg.md).
# Unset --defender_adv_reg_attacker_lr takes it on every dataset.
ADV_REG_CODE_ATTACKER_LR = 0.0001
# AdvReg's penalty on the attack model's membership probability h of members in the classifier loss: 'linear' is the
# authors' Purchase and Texas code, lambda·mean(h), and 'log' the paper's lambda·log h (Algorithm 1), which has no lower
# bound and collapses the classifier (docs/defenses/adv_reg.md). Unset --defender_adv_reg_penalty takes it.
ADV_REG_CODE_PENALTY = 'linear'
