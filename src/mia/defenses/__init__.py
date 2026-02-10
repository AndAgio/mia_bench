from src.mia.defenses.vanilla import VanillaDefender
from src.mia.defenses.training.differential_privacy import DifferentialPrivacyDefender
from src.mia.defenses.training.relax_loss import RelaxLossDefender
from src.mia.defenses.training.adversarial_regularization import AdvRegDefender
from src.mia.defenses.training.mixup import MixupDefender
from src.mia.defenses.training.selena import SelenaDefender
from src.mia.defenses.training.mist import MistDefender
from src.mia.defenses.training.weigthed_smoothing import WeightedSmoothingDefender
from src.mia.defenses.training.mmd import MmdDefender
from src.mia.defenses.post_hoc.mem_guard import MemGuardDefender
from src.mia.defenses.post_hoc.purifier import PurifierDefender
from src.mia.defenses.hybrid.hamp import HampDefender



def get_defender_class(defense_mode: str):
    if defense_mode in ['none', 'no', 'vanilla']:
        return VanillaDefender
    elif defense_mode in ['dp', 'differential_privacy', 'differential-privacy']:
        return DifferentialPrivacyDefender
    elif defense_mode in ['mem_guard', 'memguard', 'mem-guard']:
        return MemGuardDefender
    elif defense_mode in ['relax_loss', 'relaxloss', 'relax-loss']:
        return RelaxLossDefender
    elif defense_mode in ['adv_reg', 'advreg', 'adv-reg']:
        return AdvRegDefender
    elif defense_mode in ['mixup']:
        return MixupDefender
    elif defense_mode in ['selena']:
        return SelenaDefender
    elif defense_mode in ['hamp', 'hamp_train', 'hamp_test', 'hamp_full']:
        return HampDefender
    elif defense_mode in ['mist', 'mist_mixup', 'mist-mixup']:
        return MistDefender
    elif defense_mode in ['weighted_smoothing', 'weighted-smoothing', 'weighted_smooth', 'weighted-smooth', 'weightedsmoothing', 'weightedsmooth', 'ws']:
        return WeightedSmoothingDefender
    elif defense_mode in ['purifier']:
        return PurifierDefender
    elif defense_mode in ['mmd', 'mmd_mixup', 'mmd-mixup']:
        return MmdDefender
    else:
        raise ValueError('Defense "{}" not found or not implemented yet! Double check your settings please!'.format(defense_mode))