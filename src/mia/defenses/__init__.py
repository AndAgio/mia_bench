from src.mia.defenses.vanilla import VanillaDefender
from src.mia.defenses.training.differential_privacy import DifferentialPrivacyDefender
from src.mia.defenses.post_hoc.mem_guard import MemGuardDefender



def get_defender_class(defense_mode: str):
    if defense_mode in ['none', 'no', 'vanilla']:
        return VanillaDefender
    elif defense_mode == 'dp':
        return DifferentialPrivacyDefender
    elif defense_mode in ['mem_guard', 'memguard', 'mem-guard']:
        return MemGuardDefender
    else:
        raise ValueError('Defense "{}" not found or not implemented yet! Double check your settings please!'.format(defense_mode))