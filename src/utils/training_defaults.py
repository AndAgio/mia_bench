"""Per-dataset defaults of the training options of the target model (--defender_*) and the shadow models (--attacker_*).

A training option left unset on the command line takes its dataset's value from TRAINING_DEFAULTS. The value is
filled in right after parsing (see SettingsParser in src/utils/settings.py), so it enters the run hashes like a value
given on the command line: changing a default here gives the affected runs new hashes.
"""

# The options with a per-dataset default, without their --defender_ or --attacker_ prefix.
TRAINING_OPTIONS = ('optimizer', 'lr', 'lr_sched', 'weight_decay', 'momentum', 'epochs', 'batch_size')

# Image datasets: SGD with momentum and a cosine schedule, the benchmark's defaults for its ResNets.
# TODO: check each image dataset as the tabular ones were (train and test accuracy per epoch, against e.g. SGD at 0.1),
# and give each its own values: Fashion-MNIST already fits 99.6% of its training data by epoch 10 of 100.
IMAGE_TRAINING = dict(optimizer='sgd', lr=0.01, lr_sched='cosine', weight_decay=5e-4, momentum=0.9, epochs=100,
                      batch_size=256)

# The tabular datasets use the tabular MLP, on which SGD at 0.01 barely trains (Texas: 22.6% train and 22.5% test
# accuracy after 30 epochs). Each takes the better of two recipes, compared on 20,000 training records (5,000 on News):
# AdamW at 1e-3 for 50 epochs, and the RMIA authors' Purchase recipe (privacy_meter's configs/purchase.yaml: SGD at
# 0.1, no momentum or weight decay, 100 epochs), here with a cosine schedule. Train / test accuracy:
#   Purchase: AdamW 99.8% / 75.6%, SGD 99.7% / 82.7%
#   Texas:    AdamW 100% / 55.0%,  SGD 74.5% / 55.6% (still under-fit)
#   News:     AdamW 100% / 76.8%,  SGD 5.7% / 5.3% (does not learn)
SGD_PURCHASE_TRAINING = dict(optimizer='sgd', lr=0.1, lr_sched='cosine', weight_decay=0., momentum=0., epochs=100,
                             batch_size=256)
# momentum is not used by AdamW.
ADAMW_TABULAR_TRAINING = dict(optimizer='adamw', lr=1e-3, lr_sched='const', weight_decay=5e-4, momentum=0.9, epochs=50,
                              batch_size=256)

# One entry per dataset of --dataset, so that each can be changed on its own.
TRAINING_DEFAULTS = {
    'cifar10': {**IMAGE_TRAINING},
    'cifar100': {**IMAGE_TRAINING},
    'cinic10': {**IMAGE_TRAINING},
    'svhn': {**IMAGE_TRAINING},
    'fmnist': {**IMAGE_TRAINING},
    'gtsrb': {**IMAGE_TRAINING},
    'tinyimagenet': {**IMAGE_TRAINING},
    'imagenet': {**IMAGE_TRAINING},
    'purchase': {**SGD_PURCHASE_TRAINING},
    'texas': {**ADAMW_TABULAR_TRAINING},
    'news': {**ADAMW_TABULAR_TRAINING},
}


def apply_training_defaults(settings):
    """Give every unset training option of the target and the shadow models its dataset's default."""
    defaults = TRAINING_DEFAULTS[settings.dataset]
    for role in ('defender', 'attacker'):
        for option in TRAINING_OPTIONS:
            if getattr(settings, f'{role}_{option}', None) is None:
                setattr(settings, f'{role}_{option}', defaults[option])
    return settings
