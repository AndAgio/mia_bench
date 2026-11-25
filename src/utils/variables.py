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

# How membership decisions are made from an attack's scores (see src/mia/helpers/decision_rules.py).
DECISION_RULES = ('attack', 'calibrated', 'oracle')

# Fraction of the defender val split that AdvReg uses as its reference set D' (the rest selects the best checkpoint).
ADV_REG_REFERENCE_FRACTION_OF_VAL_SPLIT = 0.5
# Fraction of the defender val split that MMD uses as its reference non-members (the rest selects the best checkpoint).
MMD_REFERENCE_FRACTION_OF_VAL_SPLIT = 0.5

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
# --defender_weighted_smoothing_plain_epochs takes the value for the dataset's kind; News counts as non-image.
WEIGHTED_SMOOTHING_PAPER_PLAIN_EPOCHS = {'image': 2, 'non_image': 1}

# AdvReg's adversarial regularization factor lambda per dataset, from Table 3 of the paper (arXiv 1807.05852). The
# authors' code (github.com/SPIN-UMass/ML-Privacy-Regulization) agrees on Purchase only, and uses other penalties. Unset
# --defender_adv_reg_lambda takes the dataset's value, other image datasets CIFAR-100's, and other non-image datasets
# Purchase's.
ADV_REG_PAPER_LAMBDAS = {'purchase': 3.0, 'texas': 2.0, 'cifar100': 6.0}
ADV_REG_IMAGE_FALLBACK_DATASET = 'cifar100'
ADV_REG_NONIMAGE_FALLBACK_DATASET = 'purchase'

# MMD's loss weight lambda per dataset, from Table 5 of the paper (arXiv 2002.12062): the largest weight whose test
# accuracy stays within 1% of the undefended model's. On CIFAR, the ResNet-20 values, the paper's model closest to
# ResNet-18. Unset --defender_mmd_lambda takes the dataset's value, other image datasets CIFAR-10's, and other non-image
# datasets Purchase's.
MMD_PAPER_LAMBDAS = {'cifar10': 0.5, 'cifar100': 4.0, 'purchase': 0.1, 'texas': 1.0}
MMD_IMAGE_FALLBACK_DATASET = 'cifar10'
MMD_NONIMAGE_FALLBACK_DATASET = 'purchase'
# An epoch adds the MMD pass only when the training accuracy exceeds the reference accuracy by this much, as in the
# authors' code (github.com/colaalex111/MMD-mixup-Defense), which finds MMD in every epoch unstable.
MMD_CODE_MIN_ACCURACY_GAP = 0.03
# AdvReg's learning rate of the attack model's Adam optimizer, from the authors' Purchase code. The paper's 0.001
# (Sec. 5.3) lets the attack model die, and the code's CIFAR value 0.00001 is too slow (docs/defenses/adv_reg.md).
# Unset --defender_adv_reg_attacker_lr takes it on every dataset.
ADV_REG_CODE_ATTACKER_LR = 0.0001
# AdvReg's penalty on the attack model's membership probability h of members in the classifier loss: 'linear' is the
# authors' Purchase and Texas code, lambda·mean(h), and 'log' the paper's lambda·log h (Algorithm 1), which has no lower
# bound and collapses the classifier (docs/defenses/adv_reg.md). Unset --defender_adv_reg_penalty takes it.
ADV_REG_CODE_PENALTY = 'linear'

# Data augmentation's probability of zeroing each feature of the training samples of tabular datasets, where the image
# transforms do not apply: our choice, from a check on Purchase (docs/defenses/data_augmentation.md). Unset
# --defender_augment_feature_mask_prob takes it.
DATA_AUGMENTATION_FEATURE_MASK_PROB = 0.2

# DP-SGD's delta, at which the privacy budget epsilon is reported and targeted: the value of every experiment of Abadi
# et al. (arXiv 1607.00133, Sec. 5). Unset --defender_dp_delta takes it.
DP_PAPER_DELTA = 1e-5

# DP-SAT's radius of the weight perturbation, which the dp defense uses with --defender_optimizer sam: the value of
# Park et al. without tuning, for all their models but one (arXiv 2306.05651, Appendix G.2, Table 10). Their tuned
# values range from 0.01 to 0.1 (Table 5). Unset --defender_sam_rho takes it.
DP_SAT_PAPER_RHO = 0.03

# Quantile regression attack (src/mia/attacks/black_box/quantile.py, docs/attacks/quantile.md).
# Share of the attacker's data held out to choose the regression model's epoch when --attacker_quantile_val_split is
# unset.
QUANTILE_DEFAULT_VAL_SPLIT = 0.2
# Levels always added to the pinball grid: log-spaced from 0.9 to 0.9999, where small alphas decide. The loss averages
# over the levels, so a grid crowded near 1 fits that tail better: on CIFAR-10 at alpha = 0.01, 100 even levels let
# 4.8% of non-members through, and these 50 extra levels brought it to 1.8%, as the authors' 41-level grid did.
QUANTILE_TAIL_LEVELS, QUANTILE_TAIL_LOW, QUANTILE_TAIL_HIGH = 50, 0.9, 0.9999
# After training, a warning when the share of validation samples (non-members) above their threshold is this
# unlikely under a false-positive rate of alpha (one-sided binomial test).
QUANTILE_VALIDATION_WARNING_P_VALUE = 0.01
# The authors' training recipe for the regression model, used instead of the --attacker_* training settings, which
# stay for the shadow models of the other attacks: the defaults of quantile-mia's train_mia.py (AdamW, learning rate
# 1e-4, weight decay 1e-4, gradient clipping 1.0, no scheduler, 30 epochs), with batch 64, the effective batch the
# paper's tuning consistently chose (App. D). Under the attacker's SGD settings, the Gaussian head learned nothing.
QUANTILE_PAPER_OPTIMIZER = dict(name='adamw', lr=1e-4, weight_decay=1e-4)
QUANTILE_PAPER_EPOCHS = 30
QUANTILE_PAPER_BATCH_SIZE = 64
QUANTILE_PAPER_GRAD_CLIP = 1.0
# Query budget of HopSkipJump or QEBA for each sample, when --attacker_boundary_n_queries is unset: SBA takes the
# paper's 2,500 (Choquette-Choo et al., App. B.1); UBA keeps the library's earlier value until its review.
SBA_PAPER_N_QUERIES = 2500
UBA_DEFAULT_N_QUERIES = 1000

# Noise levels the noise robustness attack tries on its shadow model when --attacker_noise_robust_sigmas is unset.
# Images: Gaussian standard deviations in pixels, spanning the authors' code's 0.02 to 0.1 (Choquette-Choo et al.).
# Binary features (purchase, texas): flip probabilities of 1/d, 2/d, 3/d, 5/d and 10/d for d features, as in the
# authors' code. Other tabular features: Gaussian standard deviations around the best one for News's TF-IDF features in
# a test.
NOISE_ROBUSTNESS_IMAGE_SIGMAS = [0.02, 0.03, 0.05, 0.07, 0.1]
NOISE_ROBUSTNESS_CODE_FLIPPED_FEATURES = [1, 2, 3, 5, 10]
NOISE_ROBUSTNESS_TABULAR_SIGMAS = [0.002, 0.005, 0.01, 0.02]
